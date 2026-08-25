use super::{Concurrency, Tool, ToolOutput, ToolPrompt, ToolSpec, resolve_workspace_path};
use async_trait::async_trait;
use fff_search::{
    Constraint, FFFMode, FilePicker, FilePickerOptions, FuzzySearchOptions, GrepMode, GrepResult,
    GrepSearchOptions, MixedItemRef, MixedSearchConfig, PaginationArgs, QueryParser,
    SharedFilePicker, SharedFrecency, parse_grep_query,
};
use llm::ToolDefinition;
use llm::util::truncate_utf8_prefix;
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use tokio::fs;
use tokio::sync::{OnceCell, Semaphore};
use tokio_util::sync::CancellationToken;

pub const DEFAULT_FIND_LIMIT: usize = 100;
pub const MAX_FIND_LIMIT: usize = 500;
pub const DEFAULT_GREP_LIMIT: usize = 50;
pub const MAX_GREP_LIMIT: usize = 500;
/// A pathological regex can spend minutes scanning a large tree; cap each
/// grep call at a generous budget and return partial results with a notice.
const GREP_TIME_BUDGET_MS: u64 = 10_000;
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

/// A per-workspace, watched FFF index. The [`FilePicker`] is created lazily
/// on first use — never during tool-registry construction — so process startup
/// does not pay for a full filesystem scan before the first frame paints. The
/// scan itself then runs on fff's own background thread while the UI is live;
/// the first `find`/`grep` call awaits scan completion exactly as before.
pub struct FileSearchIndex {
    root: PathBuf,
    picker: OnceCell<SharedFilePicker>,
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

        Ok(Self {
            root,
            picker: OnceCell::new(),
            _frecency: SharedFrecency::noop(),
            search_slots: Arc::new(Semaphore::new(config.max_concurrent_searches.max(1))),
            config,
            shutdown: AtomicBool::new(false),
        })
    }

    /// Create the underlying picker on first use. Construction itself is
    /// cheap (thread spawns only); the filesystem walk runs on fff's own
    /// background thread afterwards, and the first search waits for scan
    /// completion exactly as it always has. Failures are cached by the
    /// [`OnceCell`] so every call observes the same error instead of
    /// re-spawning watchers.
    async fn ensure_picker(&self) -> Result<&SharedFilePicker, String> {
        if self.shutdown.load(Ordering::Acquire) {
            return Err("find index is shut down".into());
        }
        let picker = self
            .picker
            .get_or_try_init(|| async {
                if self.shutdown.load(Ordering::Acquire) {
                    return Err("find index is shut down".to_owned());
                }
                let picker = SharedFilePicker::default();
                FilePicker::new_with_shared_state(
                    picker.clone(),
                    SharedFrecency::noop(),
                    FilePickerOptions {
                        base_path: self.root.to_string_lossy().into_owned(),
                        mode: FFFMode::Ai,
                        // Unit tests create many short-lived temporary indexes in
                        // parallel; watchers add no coverage there and can starve
                        // FFF's scan pool. Production indexes remain watched.
                        watch: !cfg!(test),
                        follow_symlinks: false,
                        enable_mmap_cache: false,
                        enable_content_indexing: false,
                        ..FilePickerOptions::default()
                    },
                )
                .map_err(|error| error.to_string())?;
                Ok(picker)
            })
            .await?;
        if self.shutdown.load(Ordering::Acquire) {
            stop_picker(picker);
            return Err("find index is shut down".into());
        }
        Ok(picker)
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn config(&self) -> &FindConfig {
        &self.config
    }

    /// Explicitly stop the watcher and release the picker.  `Drop` calls this
    /// as a safety net, while the application can use it at a known lifecycle
    /// boundary before awaiting its agent task.  A lazily-created index that
    /// was never used has nothing to shut down.
    pub fn shutdown(&self) {
        if self.shutdown.swap(true, Ordering::AcqRel) {
            return;
        }
        let Some(picker) = self.picker.get() else {
            return;
        };
        stop_picker(picker);
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
        let picker = self.ensure_picker().await?.clone();

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

    /// Search file contents for a pattern across the same watched index used
    /// by [`Self::search`].  Shares the concurrency semaphore and scope
    /// validation so grep and find stay bounded and workspace-rooted.
    pub(crate) async fn grep(
        &self,
        pattern: String,
        scope: Option<String>,
        limit: usize,
        context: usize,
        mode: GrepMode,
        cancel: CancellationToken,
    ) -> Result<GrepRawOutput, String> {
        if self.shutdown.load(Ordering::Acquire) {
            return Err("grep index is shut down".into());
        }
        if pattern.trim().is_empty() {
            return Err("pattern must not be empty".into());
        }
        let picker = self.ensure_picker().await?.clone();

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
                permit.map_err(|_| "grep index is shut down".to_owned())?
            }
            _ = cancel.cancelled() => return Err("cancelled".into()),
        };

        let scan_timeout = self.config.scan_timeout;
        let pattern_for_job = pattern.clone();
        let scope_for_job = scope.clone();
        let cancel_for_job = cancel.clone();
        let options = GrepSearchOptions {
            // Ask FFF for one sentinel match beyond the public hard cap. Its
            // page limit is soft (it finishes the current file), so Harness
            // still applies the aggregate limit while collecting below.
            page_limit: limit.saturating_add(1),
            max_matches_per_file: limit.saturating_add(1),
            mode,
            before_context: context,
            after_context: context,
            trim_whitespace: true,
            time_budget_ms: GREP_TIME_BUDGET_MS,
            ..GrepSearchOptions::default()
        };
        let join = tokio::task::spawn_blocking(move || {
            let _permit = permit;
            grep_sync(
                &picker,
                &pattern_for_job,
                scope_for_job.as_deref(),
                &options,
                limit,
                scan_timeout,
                &cancel_for_job,
            )
        });

        tokio::select! {
            result = join => result.map_err(|error| format!("grep search task failed: {error}"))?,
            _ = cancel.cancelled() => Err("cancelled".into()),
        }
    }

    /// Search for several literal alternatives in one native FFF traversal.
    pub(crate) async fn multi_grep(
        &self,
        patterns: Vec<String>,
        scope: Option<String>,
        limit: usize,
        context: usize,
        cancel: CancellationToken,
    ) -> Result<GrepRawOutput, String> {
        if self.shutdown.load(Ordering::Acquire) {
            return Err("grep index is shut down".into());
        }
        let picker = self.ensure_picker().await?.clone();
        let scope = match scope {
            Some(scope) if !scope.trim().is_empty() && scope != "." => {
                let scope = self.validate_scope(&scope).await?;
                (!scope.is_empty()).then_some(scope)
            }
            _ => None,
        };
        let permit = tokio::select! {
            permit = Arc::clone(&self.search_slots).acquire_owned() => {
                permit.map_err(|_| "grep index is shut down".to_owned())?
            }
            _ = cancel.cancelled() => return Err("cancelled".into()),
        };
        let options = GrepSearchOptions {
            page_limit: limit.saturating_add(1),
            max_matches_per_file: limit.saturating_add(1),
            mode: GrepMode::PlainText,
            before_context: context,
            after_context: context,
            trim_whitespace: true,
            time_budget_ms: GREP_TIME_BUDGET_MS,
            ..GrepSearchOptions::default()
        };
        let scope_for_job = scope.clone();
        let cancel_for_job = cancel.clone();
        let scan_timeout = self.config.scan_timeout;
        let join = tokio::task::spawn_blocking(move || {
            let _permit = permit;
            multi_grep_sync(
                &picker,
                &patterns,
                scope_for_job.as_deref(),
                &options,
                limit,
                scan_timeout,
                &cancel_for_job,
            )
        });
        tokio::select! {
            result = join => result.map_err(|error| format!("multigrep search task failed: {error}"))?,
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

fn stop_picker(picker: &SharedFilePicker) {
    picker.cancel();
    picker.shutdown_watches_and_wait();
    if let Ok(mut guard) = picker.write() {
        if let Some(picker) = guard.as_mut() {
            picker.cancel();
            picker.stop_background_monitor();
        }
        guard.take();
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

/// Fully formatted, owned result of a grep search.  Everything the formatter
/// needs is copied out while the picker read guard is alive.
pub(crate) struct GrepRawOutput {
    lines: Vec<String>,
    match_count: usize,
    file_count: usize,
    rendered_line_count: usize,
    has_more: bool,
    pub(crate) regex_fallback_error: Option<String>,
    literal_fallback: bool,
}

fn grep_sync(
    shared_picker: &SharedFilePicker,
    pattern: &str,
    scope: Option<&str>,
    options: &GrepSearchOptions,
    hard_limit: usize,
    scan_timeout: Duration,
    cancel: &CancellationToken,
) -> Result<GrepRawOutput, String> {
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

    let mut parsed = parse_grep_query(pattern);
    let scoped_glob = scope.map(|scope| format!("{}/**", escape_glob(scope)));
    if let Some(glob) = scoped_glob.as_deref() {
        parsed.constraints.push(Constraint::Glob(glob));
    }
    if cancel.is_cancelled() {
        return Err("cancelled".into());
    }
    let result = picker.grep(&parsed, options);
    collect_grep_result(result, picker, scope, hard_limit, cancel)
}

fn multi_grep_sync(
    shared_picker: &SharedFilePicker,
    patterns: &[String],
    scope: Option<&str>,
    options: &GrepSearchOptions,
    hard_limit: usize,
    scan_timeout: Duration,
    cancel: &CancellationToken,
) -> Result<GrepRawOutput, String> {
    let scan_start = Instant::now();
    while !shared_picker.wait_for_scan(
        scan_timeout
            .saturating_sub(scan_start.elapsed())
            .min(SCAN_WAIT_POLL),
    ) {
        if cancel.is_cancelled() {
            return Err("cancelled".into());
        }
        if scan_start.elapsed() >= scan_timeout {
            return Err(format!(
                "initial file scan did not finish within {} seconds",
                scan_timeout.as_secs()
            ));
        }
    }
    let guard = shared_picker
        .read()
        .map_err(|error| format!("cannot access find index: {error}"))?;
    let picker = guard
        .as_ref()
        .ok_or_else(|| "find index is unavailable".to_owned())?;
    let scoped_glob = scope.map(|scope| format!("{}/**", escape_glob(scope)));
    let constraints = scoped_glob
        .as_deref()
        .map(|glob| vec![Constraint::Glob(glob)])
        .unwrap_or_default();
    let pattern_refs = patterns.iter().map(String::as_str).collect::<Vec<_>>();
    let result = picker.multi_grep(&pattern_refs, &constraints, options);
    collect_grep_result(result, picker, scope, hard_limit, cancel)
}

fn collect_grep_result(
    result: GrepResult<'_>,
    picker: &FilePicker,
    scope: Option<&str>,
    hard_limit: usize,
    cancel: &CancellationToken,
) -> Result<GrepRawOutput, String> {
    let scope_prefix = scope.map(|scope| format!("{scope}/"));
    let mut source_lines: BTreeMap<(String, u64), (String, bool)> = BTreeMap::new();
    let mut matched_files = BTreeSet::new();
    let mut match_count = 0usize;
    let mut hard_truncated = false;
    for m in &result.matches {
        if cancel.is_cancelled() {
            return Err("cancelled".into());
        }
        let path = result.files[m.file_index]
            .relative_path(picker)
            .replace('\\', "/");
        if let Some(prefix) = scope_prefix.as_deref()
            && path != scope.unwrap_or_default()
            && !path.starts_with(prefix)
        {
            continue;
        }
        if match_count >= hard_limit {
            hard_truncated = true;
            break;
        }
        match_count += 1;
        matched_files.insert(path.clone());
        let before_count = m.context_before.len();
        for (index, line) in m.context_before.iter().enumerate() {
            let line_number = m.line_number.saturating_sub((before_count - index) as u64);
            source_lines
                .entry((path.clone(), line_number))
                .or_insert_with(|| (line.clone(), false));
        }
        source_lines
            .entry((path.clone(), m.line_number))
            .and_modify(|entry| {
                entry.0.clone_from(&m.line_content);
                entry.1 = true;
            })
            .or_insert_with(|| (m.line_content.clone(), true));
        for (index, line) in m.context_after.iter().enumerate() {
            let line_number = m.line_number + index as u64 + 1;
            source_lines
                .entry((path.clone(), line_number))
                .or_insert_with(|| (line.clone(), false));
        }
    }

    let lines = source_lines
        .into_iter()
        .map(|((path, line_number), (content, is_match))| {
            let separator = if is_match { ':' } else { '-' };
            format!("{path}{separator}{line_number}{separator}{content}")
        })
        .collect::<Vec<_>>();
    let rendered_line_count = lines.len();
    Ok(GrepRawOutput {
        lines,
        match_count,
        file_count: matched_files.len(),
        rendered_line_count,
        has_more: hard_truncated || result.next_file_offset > 0,
        regex_fallback_error: result.regex_fallback_error.clone(),
        literal_fallback: result.literal_fallback,
    })
}

pub struct FindTool {
    index: Arc<FileSearchIndex>,
}

impl FindTool {
    pub fn new(index: Arc<FileSearchIndex>) -> Self {
        Self { index }
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
                            "minLength": 1,
                            "description": "Optional workspace-relative directory scope. Omit this field to search the entire workspace; when provided it must not be empty."
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

    fn concurrency(&self, _args: &Value) -> Concurrency {
        Concurrency::ReadOnly
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
            output = truncate_utf8_prefix(&output, max_output).to_owned();
            output.push('\n');
        }
        output.push_str(&notice);
    }
    output.trim_end_matches('\n').to_owned()
}

/// Render grep matches ripgrep-style: `path:line:content` for matches and
/// `path-line-content` for context lines, with a trailing notice when results
/// were truncated or the query silently degraded.
pub(crate) fn format_grep_output(
    raw: GrepRawOutput,
    pattern: &str,
    _requested_limit: usize,
    max_limit: usize,
) -> String {
    if raw.lines.is_empty() {
        let fallback = if raw.literal_fallback {
            " (query constraints were ignored; searched the full pattern as literal text)"
        } else {
            ""
        };
        return format!("No matches for pattern `{pattern}`{fallback}.");
    }

    let mut output = String::new();
    let mut byte_truncated = false;
    let output_budget = MAX_OUTPUT_BYTES.saturating_sub(MAX_TRUNCATION_NOTICE_BYTES);
    let mut rendered = 0usize;
    for line in &raw.lines {
        if rendered >= MAX_OUTPUT_LINES || output.len() + line.len() + 1 > output_budget {
            byte_truncated = true;
            break;
        }
        output.push_str(line);
        output.push('\n');
        rendered += 1;
    }

    let mut notices: Vec<String> = Vec::new();
    if raw.literal_fallback {
        notices.push(
            "query constraints did not match any files; searched the full pattern as literal text"
                .to_owned(),
        );
    }
    if raw.has_more {
        notices.push(format!(
            "hard match limit reached: {} matches in {} file(s), {rendered} source lines rendered; increase limit (maximum {max_limit}) or narrow the pattern",
            raw.match_count, raw.file_count,
        ));
    } else if byte_truncated {
        notices
            .push("output size limit reached; narrow the pattern to see fewer results".to_owned());
    }
    if !raw.has_more && (raw.match_count > 1 || raw.rendered_line_count != raw.match_count) {
        notices.push(format!(
            "{} matches in {} file(s), {} source lines rendered",
            raw.match_count, raw.file_count, rendered
        ));
    }

    if !notices.is_empty() {
        let joined = notices.join("; ");
        let notice_overhead = joined.len() + 3; // \n [ ]
        if output.len() + notice_overhead > MAX_OUTPUT_BYTES {
            let max_output = MAX_OUTPUT_BYTES.saturating_sub(notice_overhead);
            output = truncate_utf8_prefix(&output, max_output).to_owned();
        }
        output.push('\n');
        output.push('[');
        output.push_str(&joined);
        output.push(']');
    }
    output.trim_end_matches('\n').to_owned()
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

    #[test]
    fn scope_schema_requires_a_non_empty_string_when_present() {
        let directory = tempdir().unwrap();
        let tool = FindTool::new(Arc::new(FileSearchIndex::new(directory.path()).unwrap()));
        let parameters = tool.spec().definition.parameters;

        assert_eq!(parameters["properties"]["path"]["minLength"], 1);
        assert!(
            parameters["properties"]["path"]["description"]
                .as_str()
                .unwrap()
                .contains("Omit this field")
        );
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

    #[test]
    fn grep_output_is_bounded_and_actionable() {
        let raw = GrepRawOutput {
            lines: (0..600)
                .map(|i| format!("file-{i}.rs:{i}:line {i}"))
                .collect(),
            match_count: 600,
            file_count: 600,
            rendered_line_count: 600,
            has_more: true,
            regex_fallback_error: None,
            literal_fallback: false,
        };
        let output = format_grep_output(raw, "line", 100, 500);
        assert!(output.contains("increase limit"));
        assert!(output.len() <= MAX_OUTPUT_BYTES);

        let no_matches = format_grep_output(
            GrepRawOutput {
                lines: Vec::new(),
                match_count: 0,
                file_count: 0,
                rendered_line_count: 0,
                has_more: false,
                regex_fallback_error: None,
                literal_fallback: false,
            },
            "zzz",
            50,
            500,
        );
        assert!(no_matches.contains("No matches"));

        let fallback = format_grep_output(
            GrepRawOutput {
                lines: vec!["a.rs:1:x".into()],
                match_count: 1,
                file_count: 1,
                rendered_line_count: 1,
                has_more: false,
                regex_fallback_error: Some("bad regex".into()),
                literal_fallback: false,
            },
            "(unclosed",
            50,
            500,
        );
        assert_eq!(fallback, "a.rs:1:x");
    }
}
