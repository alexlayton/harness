//! GitHub Releases version checks and self-update support.

use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;
use sha2::Digest;
use std::path::{Path, PathBuf};
use std::process::Command as ProcessCommand;
use std::time::Duration;

const RELEASE_API: &str = "https://api.github.com/repos/alexlayton/harness/releases/latest";
const RELEASES: &str = "https://github.com/alexlayton/harness/releases";

#[derive(Deserialize)]
struct Release {
    tag_name: String,
    assets: Vec<ReleaseAsset>,
}

#[derive(Deserialize)]
struct ReleaseAsset {
    name: String,
    browser_download_url: String,
}

fn is_newer(current: &str, latest: &str) -> bool {
    let Ok(current) = semver::Version::parse(current) else {
        return false;
    };
    let tag = latest.strip_prefix('v').unwrap_or(latest);
    let Ok(latest) = semver::Version::parse(tag) else {
        return false;
    };
    latest.pre.is_empty() && latest > current
}

fn host_target(os: &str, arch: &str) -> Option<&'static str> {
    match (os, arch) {
        ("Linux", "x86_64" | "amd64") => Some("x86_64-unknown-linux-gnu"),
        ("Darwin", "arm64" | "aarch64") => Some("aarch64-apple-darwin"),
        ("Darwin", "x86_64") => Some("x86_64-apple-darwin"),
        _ => None,
    }
}

fn update_notice(current: &str, latest: &str) -> Option<String> {
    is_newer(current, latest)
        .then(|| format!("New version available: {latest} (run `harness update`)"))
}

fn current_host_target() -> Option<&'static str> {
    host_target(std::env::consts::OS, std::env::consts::ARCH)
}

async fn latest_release() -> Result<Release> {
    // Update can run before a provider client has initialized rustls.
    let _ = rustls::crypto::ring::default_provider().install_default();
    let client = reqwest::Client::builder()
        .user_agent(concat!("Harness/", env!("CARGO_PKG_VERSION")))
        .timeout(Duration::from_secs(8))
        .build()
        .context("create GitHub release client")?;
    client
        .get(RELEASE_API)
        .send()
        .await
        .context("contact GitHub Releases")?
        .error_for_status()
        .context("GitHub Releases returned an error")?
        .json()
        .await
        .context("parse GitHub's latest release response")
}

/// Check GitHub without making startup failure depend on network availability.
pub(crate) async fn check_latest() -> Option<String> {
    let result = tokio::time::timeout(Duration::from_secs(2), latest_release()).await;
    let Ok(Ok(release)) = result else {
        return None;
    };
    update_notice(env!("CARGO_PKG_VERSION"), &release.tag_name)
}

/// Install the latest official release for this host, after validating its
/// checksum and version. The existing executable is replaced only at the final
/// atomic rename, after the full download has been verified.
pub(crate) async fn run() -> Result<()> {
    let target = current_host_target().ok_or_else(|| {
        anyhow::anyhow!(
            "self-update is not supported on {} {}; download a compatible release from {RELEASES}",
            std::env::consts::OS,
            std::env::consts::ARCH
        )
    })?;
    let release = latest_release().await?;
    let version = release
        .tag_name
        .strip_prefix('v')
        .unwrap_or(&release.tag_name);
    let parsed =
        semver::Version::parse(version).context("latest release has an invalid version tag")?;
    ensure!(
        parsed.pre.is_empty(),
        "latest release is a prerelease, so it cannot be installed automatically"
    );
    ensure!(
        is_newer(env!("CARGO_PKG_VERSION"), &release.tag_name),
        "Harness {version} is not newer than the running version {}",
        env!("CARGO_PKG_VERSION")
    );

    let archive_name = format!("harness-{}-{target}.tar.gz", release.tag_name);
    let checksum_name = "SHA256SUMS";
    let archive_url = release
        .assets
        .iter()
        .find(|asset| asset.name == archive_name)
        .map(|asset| asset.browser_download_url.as_str())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "release {} has no compatible asset {archive_name}",
                release.tag_name
            )
        })?;
    let checksum_url = release
        .assets
        .iter()
        .find(|asset| asset.name == checksum_name)
        .map(|asset| asset.browser_download_url.as_str())
        .ok_or_else(|| {
            anyhow::anyhow!("release {} does not include SHA256SUMS", release.tag_name)
        })?;

    let client = reqwest::Client::builder()
        .user_agent(concat!("Harness/", env!("CARGO_PKG_VERSION")))
        .timeout(Duration::from_secs(120))
        .build()?;
    let archive = client
        .get(archive_url)
        .send()
        .await?
        .error_for_status()?
        .bytes()
        .await?;
    let sums = client
        .get(checksum_url)
        .send()
        .await?
        .error_for_status()?
        .text()
        .await?;
    let expected = checksum_for(&sums, &archive_name)?;
    let actual = format!("{:x}", sha2::Sha256::digest(&archive));
    ensure!(
        actual == expected,
        "checksum verification failed for {archive_name}"
    );

    let temp = TempUpdate::new()?;
    let archive_path = temp.path.join(&archive_name);
    std::fs::write(&archive_path, &archive).context("save downloaded release archive")?;
    let package = format!("harness-{}-{target}", release.tag_name);
    validate_archive(&archive_path, &package)?;
    let extract = ProcessCommand::new("tar")
        .args(["-xzf"])
        .arg(&archive_path)
        .arg("-C")
        .arg(&temp.path)
        .status()
        .context("extract release archive (tar is required)")?;
    ensure!(extract.success(), "could not extract release archive");
    let downloaded = temp.path.join(&package).join("harness");
    ensure!(
        downloaded.is_file() && !downloaded.is_symlink(),
        "release archive does not contain a regular harness executable"
    );
    let reported = ProcessCommand::new(&downloaded)
        .arg("--version")
        .output()
        .context("validate downloaded Harness executable")?;
    ensure!(
        reported.status.success(),
        "downloaded Harness executable failed its version check"
    );
    let reported = String::from_utf8_lossy(&reported.stdout).trim().to_owned();
    ensure!(
        reported == format!("harness {version}"),
        "downloaded executable reported unexpected version: {reported}"
    );

    let running = std::env::current_exe().context("locate the running Harness executable")?;
    let parent = running
        .parent()
        .ok_or_else(|| anyhow::anyhow!("running executable has no parent directory"))?;
    let install_path = parent.join("harness");
    // Do not replace a different path (for example, a symlinked alias) or an
    // installation whose directory is not writable.
    let running_real = std::fs::canonicalize(&running).context("resolve running executable")?;
    let install_real =
        std::fs::canonicalize(&install_path).context("resolve installed executable")?;
    ensure!(
        running_real == install_real,
        "the running executable is not the `harness` binary at {}; refusing to replace it",
        install_path.display()
    );
    let staged = parent.join(format!(".harness.update.{}.tmp", std::process::id()));
    let _ = std::fs::remove_file(&staged);
    std::fs::copy(&downloaded, &staged)
        .context("stage verified update beside existing installation")?;
    let mut permissions = std::fs::metadata(&staged)?.permissions();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        permissions.set_mode(0o755);
    }
    std::fs::set_permissions(&staged, permissions)?;
    if let Err(error) = std::fs::rename(&staged, &install_path) {
        let _ = std::fs::remove_file(&staged);
        return Err(error).context("atomically replace installed Harness executable");
    }
    println!("Updated Harness to {version} at {}", install_path.display());
    Ok(())
}

fn checksum_for(checksums: &str, archive_name: &str) -> Result<String> {
    let mut matches = checksums.lines().filter_map(|line| {
        let mut fields = line.split_whitespace();
        let hash = fields.next()?;
        let name = fields.next()?.trim_start_matches('*');
        (name == archive_name).then(|| hash.to_ascii_lowercase())
    });
    let hash = matches
        .next()
        .ok_or_else(|| anyhow::anyhow!("release checksums do not contain {archive_name}"))?;
    ensure!(
        matches.next().is_none(),
        "release checksums contain duplicate entries for {archive_name}"
    );
    ensure!(
        hash.len() == 64 && hash.bytes().all(|b| b.is_ascii_hexdigit()),
        "invalid SHA-256 checksum for {archive_name}"
    );
    Ok(hash)
}

fn validate_archive(archive: &Path, package: &str) -> Result<()> {
    let listing = ProcessCommand::new("tar")
        .arg("-tzf")
        .arg(archive)
        .output()
        .context("inspect release archive (tar is required)")?;
    ensure!(
        listing.status.success(),
        "release archive is not a valid tar.gz"
    );
    let members =
        String::from_utf8(listing.stdout).context("release archive contains invalid paths")?;
    let mut binary_count = 0;
    for member in members.lines() {
        match member {
            name if name == package
                || name == format!("{package}/")
                || name == format!("{package}/README.md")
                || name == format!("{package}/LICENSE")
                || name == format!("{package}/assets/")
                || name == format!("{package}/assets/header.png") =>
            {
                ()
            }
            name if name == format!("{package}/harness") => binary_count += 1,
            _ => bail!("unexpected release archive member: {member}"),
        }
    }
    ensure!(
        binary_count == 1,
        "release archive must contain exactly one harness executable"
    );
    Ok(())
}

struct TempUpdate {
    path: PathBuf,
}
impl TempUpdate {
    fn new() -> Result<Self> {
        let path = std::env::temp_dir().join(format!(
            "harness-update-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
        ));
        std::fs::create_dir(&path).context("create temporary update directory")?;
        Ok(Self { path })
    }
}
impl Drop for TempUpdate {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn newer_release_version_is_detected_without_a_v_prefix() {
        assert!(is_newer("0.5.4", "v0.5.5"));
        assert!(!is_newer("0.5.4", "v0.5.4"));
        assert!(!is_newer("0.5.4", "v0.5.3"));
    }

    #[test]
    fn invalid_or_prerelease_tags_are_not_offered_as_stable_updates() {
        assert!(!is_newer("0.5.4", "latest"));
        assert!(!is_newer("0.5.4", "v0.5.5-rc.1"));
    }

    #[test]
    fn release_assets_match_only_supported_host_targets() {
        assert_eq!(
            host_target("Linux", "x86_64"),
            Some("x86_64-unknown-linux-gnu")
        );
        assert_eq!(host_target("Darwin", "arm64"), Some("aarch64-apple-darwin"));
        assert_eq!(host_target("Darwin", "x86_64"), Some("x86_64-apple-darwin"));
        assert_eq!(host_target("Windows", "x86_64"), None);
    }

    #[test]
    fn update_notice_is_absent_when_current_or_unsupported() {
        assert_eq!(update_notice("0.5.4", "v0.5.4"), None);
        assert_eq!(
            update_notice("0.5.4", "v0.5.5"),
            Some("New version available: v0.5.5 (run `harness update`)".into())
        );
    }

    #[test]
    fn checksum_parser_requires_a_unique_valid_entry() {
        let sums = format!("{}  harness-v1-x.tar.gz\n", "a".repeat(64));
        assert_eq!(
            checksum_for(&sums, "harness-v1-x.tar.gz").unwrap(),
            "a".repeat(64)
        );
        assert!(checksum_for("bad  harness-v1-x.tar.gz", "harness-v1-x.tar.gz").is_err());
        assert!(checksum_for(&format!("{sums}{sums}"), "harness-v1-x.tar.gz").is_err());
    }
}
