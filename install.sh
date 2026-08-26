#!/bin/sh

set -eu

REPOSITORY_URL="${HARNESS_REPOSITORY_URL:-https://github.com/alexlayton/harness}"
version=""
bin_dir="${HARNESS_BIN_DIR:-}"

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

usage() {
    cat <<'EOF'
Install the latest Harness release.

Usage: install.sh [OPTIONS]

Options:
  --version <VERSION>  Install a specific release, such as v0.1.1
  --bin-dir <DIR>      Install directory (default: $HOME/.local/bin)
  -h, --help           Show this help

Environment:
  HARNESS_BIN_DIR         Alternative default install directory
  HARNESS_REPOSITORY_URL  Alternative release repository URL
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --version)
            [ "$#" -ge 2 ] || die "--version requires a value"
            version=$2
            shift 2
            ;;
        --version=*)
            version=${1#*=}
            shift
            ;;
        --bin-dir)
            [ "$#" -ge 2 ] || die "--bin-dir requires a value"
            bin_dir=$2
            shift 2
            ;;
        --bin-dir=*)
            bin_dir=${1#*=}
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

command -v curl >/dev/null 2>&1 || die "curl is required"
command -v tar >/dev/null 2>&1 || die "tar is required"
command -v mktemp >/dev/null 2>&1 || die "mktemp is required"

os=$(uname -s)
arch=$(uname -m)
case "$os:$arch" in
    Darwin:arm64|Darwin:aarch64)
        target="aarch64-apple-darwin"
        ;;
    Darwin:x86_64)
        target="x86_64-apple-darwin"
        ;;
    Linux:x86_64|Linux:amd64)
        target="x86_64-unknown-linux-gnu"
        ;;
    *)
        die "unsupported platform $os/$arch (supported: macOS arm64/x86_64, Linux x86_64)"
        ;;
esac

download() {
    curl --proto '=https' --tlsv1.2 --fail --location --silent --show-error \
        --retry 3 --output "$2" "$1"
}

if [ -z "$version" ]; then
    latest_url=$(curl --proto '=https' --tlsv1.2 --fail --location --silent \
        --show-error --retry 3 --output /dev/null --write-out '%{url_effective}' \
        "$REPOSITORY_URL/releases/latest")
    latest_url=${latest_url%/}
    version=${latest_url##*/}
fi

case "$version" in
    v*) ;;
    *) version="v$version" ;;
esac
case "$version" in
    v[0-9]*[!A-Za-z0-9._-]*|v[!0-9]*|v) die "invalid release version: $version" ;;
esac

if [ -z "$bin_dir" ]; then
    [ -n "${HOME:-}" ] || die "HOME is not set; pass --bin-dir"
    bin_dir="$HOME/.local/bin"
fi

archive="harness-$version-$target.tar.gz"
package="harness-$version-$target"
release_url="$REPOSITORY_URL/releases/download/$version"
tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/harness-install.XXXXXX")
install_tmp=""
cleanup() {
    rm -rf "$tmp_dir"
    if [ -n "$install_tmp" ]; then
        rm -f "$install_tmp"
    fi
}
trap cleanup EXIT
trap 'exit 1' HUP INT TERM

printf 'Downloading Harness %s for %s...\n' "$version" "$target"
download "$release_url/$archive" "$tmp_dir/$archive"
download "$release_url/SHA256SUMS" "$tmp_dir/SHA256SUMS"

expected=$(awk -v name="$archive" '$2 == name { print $1 }' "$tmp_dir/SHA256SUMS")
case "$expected" in
    ''|*[!0-9A-Fa-f]*) die "release checksums do not contain exactly one valid entry for $archive" ;;
esac
[ "${#expected}" -eq 64 ] || die "invalid checksum for $archive"

if command -v sha256sum >/dev/null 2>&1; then
    actual=$(sha256sum "$tmp_dir/$archive" | awk '{ print $1 }')
elif command -v shasum >/dev/null 2>&1; then
    actual=$(shasum -a 256 "$tmp_dir/$archive" | awk '{ print $1 }')
else
    die "sha256sum or shasum is required"
fi
[ "$actual" = "$expected" ] || die "checksum verification failed for $archive"

# Validate every path before extraction. Release archives contain only this
# directory and its three known files, so accepting anything else is unsafe.
tar -tzf "$tmp_dir/$archive" > "$tmp_dir/archive.list"
found_binary=0
while IFS= read -r member; do
    case "$member" in
        "$package"|"$package/"|"$package/README.md"|"$package/LICENSE") ;;
        "$package/harness") found_binary=$((found_binary + 1)) ;;
        *) die "unexpected archive member: $member" ;;
    esac
done < "$tmp_dir/archive.list"
[ "$found_binary" -eq 1 ] || die "archive does not contain exactly one harness binary"

tar -xzf "$tmp_dir/$archive" -C "$tmp_dir"
binary="$tmp_dir/$package/harness"
[ -f "$binary" ] && [ ! -L "$binary" ] || die "archive harness entry is not a regular file"
chmod 0755 "$binary"
installed_version=$("$binary" --version) || die "downloaded harness binary could not run"
[ "$installed_version" = "harness ${version#v}" ] || \
    die "downloaded binary reported unexpected version: $installed_version"

mkdir -p "$bin_dir"
install_tmp=$(mktemp "$bin_dir/.harness.install.XXXXXX")
cp "$binary" "$install_tmp"
chmod 0755 "$install_tmp"
mv -f "$install_tmp" "$bin_dir/harness"
install_tmp=""

printf 'Installed %s to %s/harness\n' "$installed_version" "$bin_dir"
case ":${PATH:-}:" in
    *":$bin_dir:"*) ;;
    *)
        printf 'warning: %s is not on PATH; add it to your shell configuration\n' "$bin_dir" >&2
        ;;
esac
