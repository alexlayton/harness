use super::{Tool, ToolOutput, ToolPrompt, ToolSpec, resolve_workspace_path};
use async_trait::async_trait;
use fff_search::{
    Constraint, FFFMode, FilePicker, FilePickerOptions, FuzzySearchOptions, MixedItemRef,
    MixedSearchConfig, PaginationArgs, QueryParser, SharedFilePicker, SharedFrecency,
};
use llm::ToolDefinition;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use tokio::fs;
use tokio::sync::Semaphore;
use tokio_util::sync::CancellationToken;

pub const DEFAULT_FIND_LIMIT: usize = 100;
pub const MAX_FIND_LIMIT: usize = 500;
const MAX_SEARCH_CANDIDATES: usize = 20_000;
const MAX_OUTPUT_LINES: usize = 2_000;
const MAX_OUTPUT_BYTES: usize = 50 * 1024;
const MAX_TRUNCATION_NOTICE_BYTES: usize = 512;

/// Cadence of the scan-wait loop in [`search_sync`].  fff's `wait_for_scan`
/// polls internally every 10 ms, so scan-completion latency is unaffected by
/// this outer interval; it only governs how often the loop can observe
/// cancellation and enforce the overall timeout.  A 200 ms cadence keeps a
/// 10 s cancelled scan to ~50 wake-ups instead of ~200 at 50 ms.
const SCAN_WAIT_POLL: Duration = Duration::from_millis(200);

/// Runtime knobs for the long-lived FFF index and bounded searches.
#[derive(Clone, Debug)]
pub struct FindConfig {
    pub scan_timeout: Duration,
    pub max_concurrent_searches: usize,
    pub default_limit: usize,
    pub max_limit: usize,
}

impl Default for FindConfig {
    fn default() -> Self {
        Self {
            scan_timeout: Duration::from_secs(10),
            max_concurrent_searches: 2,
            default_limit: DEFAULT_FIND_LIMIT,
            max_limit: MAX_FIND_LIMIT,
        }
    }
}

/// A per-workspace, watched FFF index.  `FilePicker` is intentionally created
/// once and shared by all `FindTool` invocations.
pub struct FileSearchIndex {
    root: PathBuf,
    picker: SharedFilePicker,
    // Keeping this handle alive documents that the picker was initialized with
    // a no-op frecency backend and leaves room for a persistent backend later.
    _frecency: SharedFrecency,
    config: FindConfig,
    search_slots: Arc<Semaphore>,
    shutdown: AtomicBool,
}

impl FileSearchIndex {
    pub fn new(root: impl AsRef<Path>) -> Result<Self, String> {
        Self::new_with_config(root, FindConfig::default())
    }

    pub fn new_with_config(root: impl AsRef<Path>, config: FindConfig) -> Result<Self, String> {
        let root = std::fs::canonicalize(root.as_ref())
            .map_err(|error| format!("cannot canonicalize workspace: {error}"))?;
        if !root.is_dir() {
            return Err(format!("workspace is not a directory: {}", root.display()));
        }
        let mut config = config;
        config.max_limit = config.max_limit.clamp(1, MAX_FIND_LIMIT);
        config.default_limit = config.default_limit.clamp(1, config.max_limit);

        let picker = SharedFilePicker::default();
        let frecency = SharedFrecency::noop();
        FilePicker::new_with_shared_state(
            picker.clone(),
            frecency.clone(),
            FilePickerOptions {
                base_path: root.to_string_lossy().into_owned(),
                mode: FFFMode::Ai,
                watch: true,
                follow_symlinks: false,
                enable_mmap_cache: false,
                enable_content_indexing: false,
                ..FilePickerOptions::default()
            },
        )
        .map_err(|error| error.to_string())?;

        Ok(Self {
            root,
            picker,
            _frecency: frecency,
            search_slots: Arc::new(Semaphore::new(config.max_concurrent_searches.max(1))),
            config,
            shutdown: AtomicBool::new(false),
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn config(&self) -> &FindConfig {
        &self.config
    }

    /// Explicitly stop the watcher and release the picker.  `Drop` calls this
    /// as a safety net, while the application can use it at a known lifecycle
    /// boundary before awaiting its agent task.
    pub fn shutdown(&self) {
        if self.shutdown.swap(true, Ordering::AcqRel) {
            return;
        }
        self.picker.cancel();
        self.picker.shutdown_watches_and_wait();
        if let Ok(mut guard) = self.picker.write() {
            if let Some(picker) = guard.as_mut() {
                picker.cancel();
                picker.stop_background_monitor();
            }
            guard.take();
        }
    }

    async fn search(
        &self,
        query: String,
        scope: Option<String>,
        limit: usize,
        cancel: CancellationToken,
    ) -> Result<SearchOutput, String> {
        if self.shutdown.load(Ordering::Acquire) {
            return Err("find index is shut down".into());
        }
        if query.trim().is_empty() {
            return Err("query must not be empty".into());
        }

        let scope = match scope {
            Some(scope) if !scope.trim().is_empty() && scope != "." => {
                let scope = self.validate_scope(&scope).await?;
                (!scope.is_empty()).then_some(scope)
            }
            _ => None,
        };
        if cancel.is_cancelled() {
            return Err("cancelled".into());
        }
        let permit = tokio::select! {
            permit = Arc::clone(&self.search_slots).acquire_owned() => {
                permit.map_err(|_| "find index is shut down".to_owned())?
            }
            _ = cancel.cancelled() => return Err("cancelled".into()),
        };

        let picker = self.picker.clone();
        let scan_timeout = self.config.scan_timeout;
        let query_for_job = query.clone();
        let scope_for_job = scope.clone();
        let cancel_for_job = cancel.clone();
        let join = tokio::task::spawn_blocking(move || {
            // If the caller is cancelled while FFF is doing synchronous work,
            // the task still owns the semaphore permit.  This keeps repeated
            // cancellations from creating unbounded detached searches.
            let _permit = permit;
            search_sync(
                &picker,
                &query_for_job,
                scope_for_job.as_deref(),
                limit,
                scan_timeout,
                &cancel_for_job,
            )
        });

        tokio::select! {
            result = join => result.map_err(|error| format!("find search task failed: {error}"))?,
            _ = cancel.cancelled() => Err("cancelled".into()),
        }
    }

    async fn validate_scope(&self, scope: &str) -> Result<String, String> {
        let candidate = resolve_workspace_path(scope, Some(&self.root), false).await?;
        let canonical = fs::canonicalize(&candidate)
            .await
            .map_err(|error| format!("cannot use search path {scope}: {error}"))?;
        if !canonical.starts_with(&self.root) {
            return Err(format!(
                "search path {scope} is outside workspace root {}",
                self.root.display()
            ));
        }
        let metadata = fs::metadata(&canonical)
            .await
            .map_err(|error| format!("cannot inspect search path {scope}: {error}"))?;
        if !metadata.is_dir() {
            return Err(format!("search path {scope} is not a directory"));
        }
        let relative = canonical
            .strip_prefix(&self.root)
            .map_err(|_| format!("search path {scope} is outside the workspace"))?
            .to_string_lossy()
            .replace('\\', "/")
            .trim_matches('/')
            .to_owned();
        Ok(relative)
    }
}

impl Drop for FileSearchIndex {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[derive(Debug)]
struct SearchOutput {
    paths: Vec<String>,
    total_matched: usize,
}

fn search_sync(
    shared_picker: &SharedFilePicker,
    query: &str,
    scope: Option<&str>,
    requested_limit: usize,
    scan_timeout: Duration,
    cancel: &CancellationToken,
) -> Result<SearchOutput, String> {
    let scan_start = Instant::now();
    loop {
        if cancel.is_cancelled() {
            return Err("cancelled".into());
        }
        let elapsed = scan_start.elapsed();
        if elapsed >= scan_timeout {
            return Err(format!(
                "initial file scan did not finish within {} seconds",
                scan_timeout.as_secs()
            ));
        }
        let remaining = scan_timeout.saturating_sub(elapsed);
        if shared_picker.wait_for_scan(remaining.min(SCAN_WAIT_POLL)) {
            break;
        }
    }
    if cancel.is_cancelled() {
        return Err("cancelled".into());
    }

    let guard = shared_picker
        .read()
        .map_err(|error| format!("cannot access find index: {error}"))?;
    let picker = guard
        .as_ref()
        .ok_or_else(|| "find index is unavailable".to_owned())?;

    // MixedSearchConfig preserves a trailing slash as the directory-only
    // signal and keeps ordinary fuzzy path text intact.  The generated glob
    // adds exact directory scoping without a second FilePicker.
    let parser = QueryParser::new(MixedSearchConfig);
    let mut parsed = parser.parse(query);
    let scoped_glob = scope.map(|scope| format!("{}/**", escape_glob(scope)));
    if let Some(glob) = scoped_glob.as_deref() {
        parsed.constraints.push(Constraint::Glob(glob));
    }

    let pagination_limit = if scope.is_some() {
        MAX_SEARCH_CANDIDATES
    } else {
        requested_limit
    };
    let result = picker.fuzzy_search_mixed(
        &parsed,
        None,
        FuzzySearchOptions {
            pagination: PaginationArgs {
                offset: 0,
                limit: pagination_limit,
            },
            ..FuzzySearchOptions::default()
        },
    );
    if cancel.is_cancelled() {
        return Err("cancelled".into());
    }

    let scope_prefix = scope.map(|scope| format!("{scope}/"));
    let mut paths = Vec::with_capacity(result.items.len().min(requested_limit));
    for item in result.items {
        if cancel.is_cancelled() {
            return Err("cancelled".into());
        }
        let (path, is_directory) = match item {
            MixedItemRef::File(file) => (file.relative_path(picker), false),
            MixedItemRef::Dir(directory) => (directory.relative_path(picker), true),
        };
        let path = path.replace('\\', "/");
        if let Some(prefix) = scope_prefix.as_deref()
            && path != scope.unwrap_or_default()
            && !path.starts_with(prefix)
        {
            continue;
        }
        let path = if is_directory && !path.ends_with('/') {
            format!("{path}/")
        } else {
            path
        };
        paths.push(path);
        if paths.len() >= requested_limit {
            break;
        }
    }

    Ok(SearchOutput {
        paths,
        total_matched: result.total_matched,
    })
}

fn escape_glob(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for character in value.chars() {
        match character {
            '\\' | '*' | '?' | '[' | ']' | '{' | '}' => {
                escaped.push('\\');
                escaped.push(character);
            }
            character => escaped.push(character),
        }
    }
    escaped
}

pub struct FindTool {
    index: Arc<FileSearchIndex>,
}

impl FindTool {
    pub fn new(index: Arc<FileSearchIndex>) -> Self {
        Self { index }
    }

    pub fn from_workspace(root: impl AsRef<Path>) -> Result<Self, String> {
        Ok(Self::new(Arc::new(FileSearchIndex::new(root)?)))
    }

    pub fn index(&self) -> &Arc<FileSearchIndex> {
        &self.index
    }
}

#[async_trait]
impl Tool for FindTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            definition: ToolDefinition {
                name: "find".into(),
                description: "Search indexed project files and directories using a fuzzy FFF query. Queries may include filename/glob constraints such as *.rs or **/*.json. Results respect repository ignore rules and are returned as paths relative to the workspace.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "minLength": 1,
                            "description": "Fuzzy filename or path query, optionally including FFF glob constraints"
                        },
                        "path": {
                            "type": "string",
                            "description": "Optional workspace-relative directory scope"
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": MAX_FIND_LIMIT,
                            "default": DEFAULT_FIND_LIMIT,
                            "description": "Maximum number of results"
                        }
                    },
                    "required": ["query"],
                    "additionalProperties": false
                }),
            },
            prompt: ToolPrompt::new(
                "Find files and directories by fuzzy query",
                ["Use find for repository path discovery instead of bash find, ls, or shell globbing.".to_owned()],
            ),
        }
    }

    async fn execute(&self, args: Value, cancel: CancellationToken) -> ToolOutput {
        let query = match args.get("query").and_then(Value::as_str) {
            Some(query) if !query.trim().is_empty() => query.trim().to_owned(),
            _ => return error("find", "query must be a non-empty string"),
        };
        let path = match args.get("path") {
            None => None,
            Some(Value::String(path)) if !path.trim().is_empty() => Some(path.clone()),
            Some(_) => return error("find", "path must be a non-empty string when provided"),
        };
        let limit = match args.get("limit") {
            None => self
                .index
                .config
                .default_limit
                .min(self.index.config.max_limit),
            Some(value) => match value.as_u64() {
                Some(value)
                    if value > 0
                        && value <= self.index.config.max_limit as u64
                        && value <= usize::MAX as u64 =>
                {
                    value as usize
                }
                _ => {
                    return error(
                        "find",
                        &format!(
                            "limit must be a positive integer no greater than {}",
                            self.index.config.max_limit
                        ),
                    );
                }
            },
        };
        if cancel.is_cancelled() {
            return error(&format!("find {query}"), "cancelled");
        }

        let summary = match path.as_deref() {
            Some(path) => format!("find {query} in {path}"),
            None => format!("find {query}"),
        };
        let result = self.index.search(query.clone(), path, limit, cancel).await;
        match result {
            Ok(result) => {
                let content = format_results(&query, result, limit, self.index.config.max_limit);
                ToolOutput {
                    content,
                    is_error: false,
                    summary,
                }
            }
            Err(message) => error(&summary, &message),
        }
    }
}

fn format_results(
    query: &str,
    result: SearchOutput,
    requested_limit: usize,
    max_limit: usize,
) -> String {
    if result.paths.is_empty() {
        return format!("No files found for query `{query}`.");
    }

    let total = result.total_matched.max(result.paths.len());
    let mut output = String::new();
    let mut shown = 0usize;
    let mut output_truncated = false;
    let output_budget = MAX_OUTPUT_BYTES.saturating_sub(MAX_TRUNCATION_NOTICE_BYTES);
    for path in &result.paths {
        if shown >= MAX_OUTPUT_LINES || output.len() + path.len() + 1 > output_budget {
            output_truncated = true;
            break;
        }
        output.push_str(path);
        output.push('\n');
        shown += 1;
    }

    let limited = total > shown || result.paths.len() >= requested_limit && total > requested_limit;
    if limited || output_truncated {
        let omitted = total.saturating_sub(shown);
        let notice = if requested_limit < max_limit && total > requested_limit {
            format!(
                "[truncated: showing {shown} of at least {total}; increase limit (maximum {max_limit}) or narrow the query]"
            )
        } else if omitted > 0 {
            format!(
                "[truncated: showing {shown} of at least {total}; narrow the query to see fewer results]"
            )
        } else {
            format!("[truncated: showing {shown} results; output size limit reached]")
        };
        if output.len() + notice.len() >= MAX_OUTPUT_BYTES {
            let max_output = MAX_OUTPUT_BYTES.saturating_sub(notice.len() + 1);
            output = truncate_utf8_owned(&output, max_output);
            output.push('\n');
        }
        output.push_str(&notice);
    }
    output.trim_end_matches('\n').to_owned()
}

fn truncate_utf8_owned(value: &str, max_bytes: usize) -> String {
    if value.len() <= max_bytes {
        return value.to_owned();
    }
    let mut end = max_bytes;
    while end > 0 && !value.is_char_boundary(end) {
        end -= 1;
    }
    value[..end].to_owned()
}

fn error(summary: &str, content: &str) -> ToolOutput {
    ToolOutput {
        content: content.to_owned(),
        is_error: true,
        summary: summary.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    async fn wait_for_results(index: &FileSearchIndex, query: &str) -> SearchOutput {
        index
            .search(query.to_owned(), None, 100, CancellationToken::new())
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn searches_files_and_directories_without_a_shell() {
        let directory = tempdir().unwrap();
        std::fs::create_dir_all(directory.path().join("src/nested")).unwrap();
        std::fs::write(directory.path().join("src/ViewModel.rs"), "fn main() {}\n").unwrap();
        std::fs::write(directory.path().join("src/nested/data.json"), "{}\n").unwrap();

        let index = FileSearchIndex::new_with_config(
            directory.path(),
            FindConfig {
                scan_timeout: Duration::from_secs(5),
                ..FindConfig::default()
            },
        )
        .unwrap();
        let result = wait_for_results(&index, "ViewModel").await;
        assert!(result.paths.iter().any(|path| path == "src/ViewModel.rs"));
        let rust = wait_for_results(&index, "*.rs").await;
        assert!(rust.paths.iter().any(|path| path == "src/ViewModel.rs"));
        let json = wait_for_results(&index, "**/*.json").await;
        assert!(json.paths.iter().any(|path| path == "src/nested/data.json"));
        let dirs = wait_for_results(&index, "nested").await;
        assert!(dirs.paths.iter().any(|path| path == "src/nested/"));
    }

    #[tokio::test]
    async fn validates_scope_and_empty_queries() {
        let directory = tempdir().unwrap();
        std::fs::create_dir_all(directory.path().join("src/nested")).unwrap();
        std::fs::write(directory.path().join("src/main.rs"), "fn main() {}\n").unwrap();
        std::fs::write(directory.path().join("other.rs"), "fn other() {}\n").unwrap();
        let index = FileSearchIndex::new(directory.path()).unwrap();
        let scoped = index
            .search(
                "*.rs".into(),
                Some("src".into()),
                100,
                CancellationToken::new(),
            )
            .await
            .unwrap();
        assert_eq!(scoped.paths, vec!["src/main.rs"]);
        let error = index
            .search(
                "x".into(),
                Some("../outside".into()),
                10,
                CancellationToken::new(),
            )
            .await
            .unwrap_err();
        assert!(error.contains("outside") || error.contains("search path"));
        let error = index
            .search(" ".into(), None, 10, CancellationToken::new())
            .await
            .unwrap_err();
        assert!(error.contains("empty"));
    }

    #[tokio::test]
    async fn respects_ignore_rules_and_excludes_symlinked_files() {
        let directory = tempdir().unwrap();
        std::fs::create_dir_all(directory.path().join("src/ignored")).unwrap();
        std::fs::create_dir(directory.path().join(".git")).unwrap();
        std::fs::write(directory.path().join(".gitignore"), "src/ignored/\n").unwrap();
        std::fs::write(directory.path().join("src/visible.rs"), "visible\n").unwrap();
        std::fs::write(directory.path().join("src/ignored/hidden.rs"), "hidden\n").unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink(
            directory.path().join("src/visible.rs"),
            directory.path().join("src/link.rs"),
        )
        .unwrap();

        let index = FileSearchIndex::new(directory.path()).unwrap();
        let result = wait_for_results(&index, "*.rs").await;
        assert!(result.paths.iter().any(|path| path == "src/visible.rs"));
        assert!(!result.paths.iter().any(|path| path.contains("ignored")));
        #[cfg(unix)]
        assert!(!result.paths.iter().any(|path| path == "src/link.rs"));
    }

    #[test]
    fn output_is_bounded_and_actionable() {
        let result = SearchOutput {
            paths: (0..600).map(|i| format!("file-{i}.rs")).collect(),
            total_matched: 600,
        };
        let output = format_results("*.rs", result, 100, 500);
        assert!(output.contains("truncated"));
        assert!(output.contains("increase limit"));
        assert!(output.len() <= MAX_OUTPUT_BYTES);
    }
}
