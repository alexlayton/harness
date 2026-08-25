//! Agent Skills discovery.
//!
//! A skill is a directory containing `SKILL.md` (frontmatter + instructions)
//! plus optional freeform `scripts/`, `references/`, `assets/`.  In a
//! [`SkillMode::Harness`] root (`.harness/skills`, `~/.harness/skills`), a
//! root-level `.md` file is also a skill; in [`SkillMode::Agents`] roots it
//! is not (spec compat).
//!
//! Only skill *descriptions* are ever placed in the model prompt (progressive
//! disclosure).  The model loads a skill's body by calling `read` on the
//! absolute `SKILL.md` path in `<location>` — so this
//! module also produces a **read-path allowlist** (`read_paths`) for
//! [`ReadTool`], covering every discovered `SKILL.md` plus the skill
//! directory's `scripts/`, `references/`, `assets/` so helper files are
//! readable too.  Everything else (`edit`, `write`, `bash`, `find`) stays
//! workspace-rooted.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

/// Max skill name length per Agent Skills spec.
pub const MAX_NAME_LENGTH: usize = 64;
/// Max skill description length per Agent Skills spec.
pub const MAX_DESCRIPTION_LENGTH: usize = 1024;

/// Which skill-discovery convention a root follows.  [`SkillMode::Harness`]
/// roots (`.harness/skills`) also treat a root-level `.md` file as a skill;
/// [`SkillMode::Agents`] roots (`.agents/skills`) do not.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SkillMode {
    Harness,
    Agents,
}

/// A discovered skill.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Skill {
    pub name: String,
    pub description: String,
    /// Absolute path to `SKILL.md` (or the root `.md` file).
    pub file_path: PathBuf,
    /// Directory containing the skill file (for resolving relative paths).
    pub base_dir: PathBuf,
    pub disable_model_invocation: bool,
}

/// Diagnostics collected during discovery (validation warnings, collisions).
/// Surfaced to the user via `/skills`, never placed in the model prompt.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SkillDiagnostic {
    pub severity: SkillSeverity,
    pub message: String,
    pub path: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SkillSeverity {
    Warning,
    Collision,
}

/// The result of discovering skills from a set of root directories.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SkillCatalog {
    pub skills: Vec<Skill>,
    pub diagnostics: Vec<SkillDiagnostic>,
    /// Absolute paths (files AND dirs) that `read` may access for skills:
    /// every skill's `file_path`, plus its `base_dir` (so `scripts/`
    /// `references/` `assets/` are reachable).
    pub read_paths: Vec<PathBuf>,
}

impl SkillCatalog {
    /// Model-invocable skills in deterministic (discovery) order.
    pub fn invocable(&self) -> Vec<&Skill> {
        self.skills
            .iter()
            .filter(|skill| !skill.disable_model_invocation)
            .collect()
    }

    /// UI-facing name/description pairs for every discovered skill,
    /// including ones the model may not invoke. Drives `/skills`, header
    /// rows, and completion; deliberately decoupled from [`Skill`] so the
    /// TUI never depends on this crate's filesystem types.
    pub fn entries(&self) -> Vec<SkillEntry> {
        self.skills
            .iter()
            .map(|skill| SkillEntry {
                name: skill.name.clone(),
                description: skill.description.clone(),
            })
            .collect()
    }

    /// True when the catalog has at least one model-invocable skill.
    pub fn has_invocable(&self) -> bool {
        self.invocable().into_iter().next().is_some()
    }

    pub fn is_empty(&self) -> bool {
        self.skills.is_empty()
    }
}

/// A skill as seen by the UI: name plus a short description.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SkillEntry {
    pub name: String,
    pub description: String,
}

/// Frontmatter fields we parse.  `description` is required
/// for a skill to be loaded.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Frontmatter {
    pub name: Option<String>,
    pub description: Option<String>,
    pub disable_model_invocation: bool,
}

/// Split `---\n…\n---` frontmatter from a markdown body.  Returns
/// `(None, raw)` when the file has no frontmatter block.
pub fn parse_frontmatter(raw: &str) -> (Option<Frontmatter>, &str) {
    let body = raw.strip_prefix('\u{feff}').unwrap_or(raw);
    let trimmed = body.trim_start();
    if !trimmed.starts_with("---") {
        return (None, body);
    }
    // The opening `---` must be at the very start of the (BOM-stripped) file.
    let rest = &body[body.len() - trimmed.len() + 3..];
    let Some(line_end) = rest.find('\n') else {
        return (None, body);
    };
    let inner = &rest[..line_end];
    let after = &rest[line_end + 1..];
    // Close delimiters: `---` or `...`, optionally followed by whitespace.
    let end_of_block = after
        .find("\n---")
        .or_else(|| after.find("\n..."))
        .map(|idx| idx + 1)
        .unwrap_or(0);
    let block = &after[..end_of_block];
    let body_start = after[end_of_block..]
        .strip_prefix("---")
        .or_else(|| after[end_of_block..].strip_prefix("..."))
        .map(|s| s.trim_start())
        .unwrap_or("");
    let frontmatter = parse_key_value_block(inner, block);
    (Some(frontmatter), body_start)
}

fn parse_key_value_block(leading_rest: &str, block: &str) -> Frontmatter {
    let mut frontmatter = Frontmatter::default();
    for line in block.lines().chain([leading_rest]) {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || !line.contains(':') {
            continue;
        }
        let (key, value) = line.split_once(':').unwrap();
        let key = key
            .trim()
            .trim_matches(|c| c == '"' || c == '\'')
            .to_owned();
        let raw_value = value.trim().trim_matches(|c| c == '"' || c == '\'');
        match key.as_str() {
            "name" => frontmatter.name = Some(raw_value.to_owned()),
            "description" => frontmatter.description = Some(raw_value.to_owned()),
            "disable-model-invocation" => {
                frontmatter.disable_model_invocation = matches!(
                    raw_value.trim().to_ascii_lowercase().as_str(),
                    "true" | "yes" | "1"
                );
            }
            _ => {}
        }
    }
    frontmatter
}

fn validate_name(name: &str, diagnostics: &mut Vec<SkillDiagnostic>, path: &Path) {
    if name.len() > MAX_NAME_LENGTH {
        diagnostics.push(SkillDiagnostic {
            severity: SkillSeverity::Warning,
            message: format!("name exceeds {MAX_NAME_LENGTH} characters ({})", name.len()),
            path: Some(path.to_path_buf()),
        });
    }
    if name.is_empty()
        || !name
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
    {
        diagnostics.push(SkillDiagnostic {
            severity: SkillSeverity::Warning,
            message: "name contains invalid characters (must be lowercase a-z, 0-9, hyphens only)"
                .into(),
            path: Some(path.to_path_buf()),
        });
    }
    if name.starts_with('-') || name.ends_with('-') {
        diagnostics.push(SkillDiagnostic {
            severity: SkillSeverity::Warning,
            message: "name must not start or end with a hyphen".into(),
            path: Some(path.to_path_buf()),
        });
    }
    if name.contains("--") {
        diagnostics.push(SkillDiagnostic {
            severity: SkillSeverity::Warning,
            message: "name must not contain consecutive hyphens".into(),
            path: Some(path.to_path_buf()),
        });
    }
}

/// Load a single skill file (a `SKILL.md` or a root-level `.md` in a
/// harness-mode root).  Returns `None` when the description is missing
/// (skill is dropped).
fn load_skill_from_file(file_path: &Path, diagnostics: &mut Vec<SkillDiagnostic>) -> Option<Skill> {
    let raw = match fs::read_to_string(file_path) {
        Ok(raw) => raw,
        Err(error) => {
            diagnostics.push(SkillDiagnostic {
                severity: SkillSeverity::Warning,
                message: format!("failed to read skill file: {error}"),
                path: Some(file_path.to_path_buf()),
            });
            return None;
        }
    };
    let (frontmatter, _body) = parse_frontmatter(&raw);
    let Some(fm) = frontmatter else {
        // No frontmatter at all → not a valid skill.
        diagnostics.push(SkillDiagnostic {
            severity: SkillSeverity::Warning,
            message: "skill file has no frontmatter".into(),
            path: Some(file_path.to_path_buf()),
        });
        return None;
    };
    let base_dir = file_path.parent().unwrap_or(Path::new(".")).to_path_buf();
    let parent_name = file_path
        .file_stem()
        .or_else(|| file_path.file_name())
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    // Fall back to parent dir name for SKILL.md; for a root .md use the stem.
    let fallback = if file_path.file_name().is_some_and(|n| n == "SKILL.md") {
        file_path
            .parent()
            .and_then(|p| p.file_name())
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or(parent_name)
    } else {
        parent_name
    };
    let name = fm.name.clone().unwrap_or(fallback.clone());
    validate_name(&name, diagnostics, file_path);
    let Some(description) = fm.description.clone().filter(|d| !d.trim().is_empty()) else {
        diagnostics.push(SkillDiagnostic {
            severity: SkillSeverity::Warning,
            message: "description is required".into(),
            path: Some(file_path.to_path_buf()),
        });
        return None;
    };
    if description.len() > MAX_DESCRIPTION_LENGTH {
        diagnostics.push(SkillDiagnostic {
            severity: SkillSeverity::Warning,
            message: format!(
                "description exceeds {MAX_DESCRIPTION_LENGTH} characters ({})",
                description.len()
            ),
            path: Some(file_path.to_path_buf()),
        });
    }
    // Reads: the skill file itself, plus its base dir (for resources).
    // Accumulation happens in `discover`'s post-selection loop so shadowed
    // skills never contribute paths.
    Some(Skill {
        name,
        description,
        file_path: file_path.to_path_buf(),
        base_dir,
        disable_model_invocation: fm.disable_model_invocation,
    })
}

/// Recursive skill discovery from a root.  `mode` controls whether a
/// root-level `.md` file is treated as a skill ([`SkillMode::Harness`] does;
/// [`SkillMode::Agents`] ignores it).  `root` is the discovery root and `ig`
/// the shared ignore matcher.
fn discover_dir(
    dir: &Path,
    mode: SkillMode,
    root: &Path,
    ig: &ignore::gitignore::Gitignore,
    skills: &mut Vec<Skill>,
    diagnostics: &mut Vec<SkillDiagnostic>,
) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    // A directory containing SKILL.md is a skill root — do not recurse.
    let skill_md = dir.join("SKILL.md");
    if skill_md.is_file() {
        let rel = skill_md.strip_prefix(root).unwrap_or(&skill_md);
        if !ig.matched(rel, false).is_ignore()
            && let Some(skill) = load_skill_from_file(&skill_md, diagnostics)
        {
            skills.push(skill);
        }
        return;
    }
    let mut children: Vec<_> = entries
        .filter_map(|entry| entry.ok())
        .filter(|entry| !entry.file_name().to_string_lossy().starts_with('.'))
        .collect();
    children.sort_by_key(|e| e.file_name());
    for entry in children {
        let file_name = entry.file_name();
        let path = dir.join(&file_name);
        let rel = path.strip_prefix(root).unwrap_or(&path);
        let is_dir = entry.file_type().map(|t| t.is_dir()).unwrap_or(false);
        if is_dir {
            if file_name == "node_modules" {
                continue;
            }
            if ig.matched(rel, true).is_ignore() {
                continue;
            }
            discover_dir(&path, mode, root, ig, skills, diagnostics);
            continue;
        }
        let is_file = entry.file_type().map(|t| t.is_file()).unwrap_or(false);
        if !is_file {
            continue;
        }
        let is_md = file_name.to_string_lossy().ends_with(".md");
        // Root-level .md files are skills only in harness mode; subdirectory
        // .md files are not skills by themselves (only SKILL.md inside a
        // directory is).
        if is_md
            && matches!(mode, SkillMode::Harness)
            && path.parent() == Some(root)
            && !ig.matched(rel, false).is_ignore()
            && let Some(skill) = load_skill_from_file(&path, diagnostics)
        {
            skills.push(skill);
        }
    }
}

/// Build a gitignore-style matcher from `.gitignore` / `.ignore` /
/// `.fdignore` files rooted at `root`. A single matcher is used for the whole
/// walk (per-directory matchers would be more precise; not needed for our
/// root set).
fn build_ignore(root: &Path) -> ignore::gitignore::Gitignore {
    let mut builder = ignore::gitignore::GitignoreBuilder::new(root);
    for name in [".gitignore", ".ignore", ".fdignore"] {
        let path = root.join(name);
        if path.is_file()
            && let Some(content) = fs::read_to_string(&path).ok()
        {
            for line in content.lines() {
                if !line.trim().is_empty() && !line.trim().starts_with('#') {
                    let _ = builder.add_line(Some(root.to_path_buf()), line);
                }
            }
        }
    }
    builder.build().unwrap_or_else(|_| {
        ignore::gitignore::GitignoreBuilder::new(root)
            .build()
            .unwrap()
    })
}

/// The chosen discovery entry point: given a root directory and mode, return
/// all skills (recursively).
pub fn load_skills_from_dir(root: &Path, mode: SkillMode) -> SkillCatalog {
    let mut skills = Vec::new();
    let mut diagnostics = Vec::new();
    let ig = build_ignore(root);
    discover_dir(root, mode, root, &ig, &mut skills, &mut diagnostics);
    SkillCatalog {
        skills,
        diagnostics,
        read_paths: Vec::new(),
    }
}

/// Discover skills from the given root directories (in order), deduping by
/// name with **project-beats-global** first-wins (decision 1). `roots` is a
/// list of `(path, mode)` in priority order: project dirs first (cwd →
/// ancestors), then global.
pub fn discover(roots: &[(PathBuf, SkillMode)]) -> SkillCatalog {
    let mut all_skills = Vec::new();
    let mut all_diagnostics = Vec::new();
    let mut by_name: HashSet<String> = HashSet::new();
    let mut seen_files: HashSet<PathBuf> = HashSet::new();
    for (root, mode) in roots {
        let c = load_skills_from_dir(root, *mode);
        all_diagnostics.extend(c.diagnostics);
        for skill in c.skills {
            let canonical =
                fs::canonicalize(&skill.file_path).unwrap_or_else(|_| skill.file_path.clone());
            if !seen_files.insert(canonical.clone()) {
                continue;
            }
            if by_name.contains(&skill.name) {
                all_diagnostics.push(SkillDiagnostic {
                    severity: SkillSeverity::Collision,
                    message: format!(
                        "name \"{}\" collision; keeping the earlier skill",
                        skill.name
                    ),
                    path: Some(skill.file_path.clone()),
                });
                continue;
            }
            by_name.insert(skill.name.clone());
            all_skills.push(skill);
        }
    }
    // Read-paths for every *kept* skill (its file plus base dir).  Only
    // skills that won the name collision contribute, and the list is deduped
    // (root-priority walk can repeat the same path).
    let mut read_paths = Vec::with_capacity(all_skills.len() * 2);
    let mut seen_paths: HashSet<PathBuf> = HashSet::new();
    for skill in &all_skills {
        for path in [&skill.file_path, &skill.base_dir] {
            if seen_paths.insert(path.clone()) {
                read_paths.push(path.clone());
            }
        }
    }
    SkillCatalog {
        skills: all_skills,
        diagnostics: all_diagnostics,
        read_paths,
    }
}

/// Format skills for the system prompt (Agent Skills XML): intro uses `read`,
/// per-skill `<name>/<description>/<location>`, absolute paths. Skills with
/// `disable_model_invocation` are excluded.
pub fn format_skills_prompt(catalog: &SkillCatalog) -> String {
    let visible = catalog.invocable();
    if visible.is_empty() {
        return String::new();
    }
    let mut lines = vec![
        "\n\nThe following skills provide specialized instructions for specific tasks.".to_owned(),
        "Use the read tool to load a skill's file when the task matches its description.".to_owned(),
        "When a skill file references a relative path, resolve it against the skill directory (parent of SKILL.md / dirname of the path) and use that absolute path in tool commands.".to_owned(),
        String::new(),
        "<available_skills>".to_owned(),
    ];
    for skill in &visible {
        lines.push("  <skill>".to_owned());
        lines.push(format!("    <name>{}</name>", escape_xml(&skill.name)));
        lines.push(format!(
            "    <description>{}</description>",
            escape_xml(&skill.description)
        ));
        lines.push(format!(
            "    <location>{}</location>",
            escape_xml(&skill.file_path.to_string_lossy())
        ));
        lines.push("  </skill>".to_owned());
    }
    lines.push("</available_skills>".to_owned());
    lines.join("\n")
}

fn escape_xml(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

/// Return the user's home directory on Unix and Windows.
pub(crate) fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .filter(|value| !value.is_empty())
        .or_else(|| std::env::var_os("USERPROFILE").filter(|value| !value.is_empty()))
        .map(PathBuf::from)
}

/// Resolve a leading `~` to the user's home directory.
pub fn expand_tilde(path: &Path) -> PathBuf {
    if let Ok(stripped) = path.strip_prefix("~")
        && let Some(home) = home_dir()
    {
        if stripped.as_os_str().is_empty() {
            return home;
        }
        return home.join(stripped);
    }
    path.to_path_buf()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn parses_frontmatter_and_extracts_body() {
        let raw = "---\nname: pdf-processing\ndescription: Extract text from PDFs\n---\nBody here";
        let (fm, body) = parse_frontmatter(raw);
        let fm = fm.expect("frontmatter");
        assert_eq!(fm.name.as_deref(), Some("pdf-processing"));
        assert_eq!(fm.description.as_deref(), Some("Extract text from PDFs"));
        assert_eq!(body, "Body here");

        // No frontmatter → (None, original)
        let (fm, body) = parse_frontmatter("just body");
        assert!(fm.is_none());
        assert_eq!(body, "just body");
    }

    #[test]
    fn discovers_project_and_global_roots_with_harness_and_agents_modes() {
        let root = tempdir().unwrap();
        let harness = root.path().join(".harness/skills");
        let agents = root.path().join(".agents/skills");
        fs::create_dir_all(&harness).unwrap();
        fs::create_dir_all(&agents).unwrap();
        // harness-mode root: dir-with-SKILL.md + root .md file
        fs::create_dir_all(harness.join("alpha")).unwrap();
        fs::write(
            harness.join("alpha/SKILL.md"),
            "---\nname: alpha\ndescription: Alpha skill\n---\nbody\n",
        )
        .unwrap();
        fs::write(
            harness.join("standalone.md"),
            "---\nname: standalone\ndescription: A root md\n---\nbody\n",
        )
        .unwrap();
        // agents-mode root: root .md ignored
        fs::write(
            agents.join("ignored.md"),
            "---\nname: ignored\ndescription: should not load\n---\nbody\n",
        )
        .unwrap();
        fs::create_dir_all(agents.join("beta")).unwrap();
        fs::write(
            agents.join("beta/SKILL.md"),
            "---\nname: beta\ndescription: Beta skill\n---\nbody\n",
        )
        .unwrap();

        let catalog = discover(&[
            (harness.clone(), SkillMode::Harness),
            (agents.clone(), SkillMode::Agents),
        ]);
        let names: Vec<_> = catalog.skills.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"alpha"));
        assert!(names.contains(&"standalone"));
        assert!(names.contains(&"beta"));
        assert!(!names.contains(&"ignored"));
    }

    #[test]
    fn missing_description_drops_skill_and_read_paths_are_populated() {
        let root = tempdir().unwrap();
        let harness = root.path().join(".harness/skills");
        fs::create_dir_all(harness.join("nodesc")).unwrap();
        fs::write(
            harness.join("nodesc/SKILL.md"),
            "---\nname: nodesc\n---\nbody\n",
        )
        .unwrap();
        // A valid skill alongside it: only *its* paths belong in read_paths.
        fs::create_dir_all(harness.join("ok")).unwrap();
        let ok_skill = harness.join("ok/SKILL.md");
        fs::write(
            &ok_skill,
            "---\nname: ok\ndescription: A good skill\n---\nbody\n",
        )
        .unwrap();
        let catalog = discover(&[(harness.clone(), SkillMode::Harness)]);
        assert_eq!(catalog.skills.len(), 1);
        assert_eq!(catalog.skills[0].name, "ok");
        assert!(
            catalog
                .diagnostics
                .iter()
                .any(|d| d.message.contains("description")),
            "expected a description diagnostic"
        );
        // The dropped skill's paths must not appear in read_paths.
        assert!(
            !catalog
                .read_paths
                .contains(&harness.join("nodesc/SKILL.md")),
            "dropped skill file must not be in read_paths"
        );
        assert!(
            !catalog.read_paths.contains(&harness.join("nodesc")),
            "dropped skill dir must not be in read_paths"
        );
        // The kept skill's file + base dir are present exactly once.
        let file_count = catalog
            .read_paths
            .iter()
            .filter(|p| *p == &ok_skill)
            .count();
        let dir_count = catalog
            .read_paths
            .iter()
            .filter(|p| *p == &harness.join("ok"))
            .count();
        assert_eq!(file_count, 1, "kept skill file present once");
        assert_eq!(dir_count, 1, "kept skill dir present once");
    }

    #[test]
    fn shadowed_skill_paths_are_absent_from_read_paths() {
        let root = tempdir().unwrap();
        let project = root.path().join("proj/.harness/skills");
        let global = root.path().join("global/.harness/skills");
        fs::create_dir_all(project.join("dup")).unwrap();
        fs::create_dir_all(global.join("dup")).unwrap();
        let project_md = project.join("dup/SKILL.md");
        let global_md = global.join("dup/SKILL.md");
        fs::write(
            &project_md,
            "---\nname: dup\ndescription: project\n---\nbody\n",
        )
        .unwrap();
        fs::write(
            &global_md,
            "---\nname: dup\ndescription: global\n---\nbody\n",
        )
        .unwrap();
        let catalog = discover(&[
            (project.clone(), SkillMode::Harness),
            (global.clone(), SkillMode::Harness),
        ]);
        // Only the winning (project) skill contributes paths.
        assert!(catalog.read_paths.contains(&project_md));
        assert!(catalog.read_paths.contains(&project.join("dup")));
        assert!(!catalog.read_paths.contains(&global_md));
        assert!(!catalog.read_paths.contains(&global.join("dup")));
    }

    #[test]
    fn collision_first_wins_project_beats_global() {
        let root = tempdir().unwrap();
        let project = root.path().join("proj/.harness/skills");
        let global_harness = root.path().join("global/.harness/skills");
        fs::create_dir_all(project.join("dup")).unwrap();
        fs::create_dir_all(global_harness.join("dup")).unwrap();
        fs::write(
            project.join("dup/SKILL.md"),
            "---\nname: dup\ndescription: project dup\n---\nproject body\n",
        )
        .unwrap();
        fs::write(
            global_harness.join("dup/SKILL.md"),
            "---\nname: dup\ndescription: global dup\n---\nglobal body\n",
        )
        .unwrap();
        let catalog = discover(&[
            (project.clone(), SkillMode::Harness),
            (global_harness.clone(), SkillMode::Harness),
        ]);
        assert_eq!(catalog.skills.len(), 1);
        assert_eq!(catalog.skills[0].description, "project dup");
        assert!(
            catalog
                .diagnostics
                .iter()
                .any(|d| d.severity == SkillSeverity::Collision),
            "expected a collision diagnostic"
        );
    }

    #[test]
    fn prompt_is_xml_and_includes_location_and_skips_disabled() {
        let root = tempdir().unwrap();
        let harness = root.path().join(".harness/skills");
        fs::create_dir_all(harness.join("a")).unwrap();
        fs::write(
            harness.join("a/SKILL.md"),
            "---\nname: alpha\ndescription: Alpha desc\n---\nbody\n",
        )
        .unwrap();
        fs::create_dir_all(harness.join("secret")).unwrap();
        fs::write(
            harness.join("secret/SKILL.md"),
            "---\nname: secret\ndescription: Manual only\ndisable-model-invocation: true\n---\nbody\n",
        )
        .unwrap();
        let catalog = discover(&[(harness, SkillMode::Harness)]);
        let prompt = format_skills_prompt(&catalog);
        assert!(prompt.contains("<available_skills>"));
        assert!(prompt.contains("<name>alpha</name>"));
        assert!(prompt.contains("<description>Alpha desc</description>"));
        assert!(prompt.contains("<location>"));
        assert!(
            !prompt.contains("secret"),
            "disabled skills must be excluded"
        );
    }

    #[test]
    fn expand_tilde_resolves_home() {
        let home = home_dir().expect("test requires HOME or USERPROFILE");
        assert_eq!(
            expand_tilde(Path::new("~/.harness/skills")),
            home.join(".harness/skills")
        );
        assert_eq!(expand_tilde(Path::new("~/")), home);
        assert_eq!(expand_tilde(Path::new("plain")), PathBuf::from("plain"));
    }
}
