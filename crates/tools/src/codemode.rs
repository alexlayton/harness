//! Bounded, fresh-state Monty orchestration. Only the async host driver can
//! admit tool calls; a detached interpreter step has no host authority.

use super::{McpOrigin, McpServerInfo, ToolOutput, ToolPrompt, ToolSpec};
use async_trait::async_trait;
use llm::ToolDefinition;
use monty::{MontyRun, RunProgress};
use monty_types::{
    CompileOptions, ExcType, ExtFunctionResult, MontyException, MontyObject, NameLookupResult,
    ObjectRef, PrintWriter, ResourceLimits, ResourceTracker, SleepMode,
};
use serde_json::{Map, Value, json};
use std::fmt::Write;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio_util::sync::CancellationToken;

/// Name of the codemode tool. Shared so the registry, dispatch, and
/// summaries cannot drift apart.
pub const CODEMODE_TOOL_NAME: &str = "codemode";

const MAX_CODE_BYTES: usize = 32 * 1024;
const MAX_VALUE_BYTES: usize = 50 * 1024;
const MAX_TOOL_CALLS: usize = 50;
const MAX_SUSPENSIONS: usize = 1000;
const COMPUTE_LIMIT: Duration = Duration::from_secs(5);
const WALL_LIMIT: Duration = Duration::from_secs(120);
const MAX_CATALOG_BYTES: usize = 512 * 1024;
/// Budget for successful nested-call outputs echoed back when a snippet
/// fails. Without them the model has to replay every earlier call (and pay a
/// round trip) just to recover results that were already computed.
const MAX_SALVAGED_BYTES: usize = 24 * 1024;

#[cfg(test)]
mod tests;

/// An internal tool lifecycle, separate from model-issued tool calls.
#[derive(Clone, Debug)]
pub enum CodemodeEvent {
    /// Must be acknowledged before the tool is allowed to start.
    Started {
        call_id: String,
        name: String,
        arguments: Value,
    },
    /// A real result; absence after a start means execution status is unknown.
    Finished {
        call_id: String,
        name: String,
        output: ToolOutput,
    },
}

/// Host policy for nested calls. Implementations must check tool access and
/// apply per-tool secret handling before returning either result form.
#[async_trait]
pub trait CodemodeExecutor: Send + Sync {
    /// Execute one selected tool, returning display text and optional JSON.
    async fn call(
        &self,
        name: &str,
        args: Value,
        cancel: CancellationToken,
    ) -> (ToolOutput, Option<Value>);

    /// Record a lifecycle event. A failed acknowledgement stops the snippet;
    /// it is not a catchable Python exception that could permit more calls.
    async fn record(&self, _event: CodemodeEvent) -> Result<(), String> {
        Ok(())
    }
}

/// One immutable catalogue, generated from the same registered tool specs that
/// previously supplied native definitions. The dictionary interface preserves
/// arbitrary MCP names and keyword names without lossy Python aliases.
///
/// Only a compact index is sent to the model. Full descriptions and JSON
/// schemas stay host-side and are reached from Python through `describe()`,
/// so the provider-visible prompt does not grow with every connected server.
pub(crate) struct Codemode {
    pub(crate) names: Vec<String>,
    entries: Vec<Entry>,
    servers: Vec<McpServerInfo>,
    spec: ToolSpec,
}

/// Full catalogue entry, served lazily by `describe(name)`.
struct Entry {
    name: String,
    summary: String,
    full: Value,
    origin: Option<McpOrigin>,
}

/// Bytes of the model-facing index spent on `name(args) - summary` lines.
/// Tools past this budget are listed by name only.
const INDEX_BUDGET: usize = 4 * 1024;
/// Bytes for the names-only tail. Anything beyond is reachable via `TOOLS`.
const INDEX_TAIL_BUDGET: usize = 2 * 1024;
const MAX_SUMMARY_CHARS: usize = 100;
const MAX_SIGNATURE_PARAMS: usize = 8;

/// Globals resolved lazily by the host when a snippet first uses the name.
const GLOBAL_TOOLS: &str = "TOOLS";
const GLOBAL_MODULES: &str = "MODULES";
const GLOBAL_SERVERS: &str = "SERVERS";
const GLOBAL_DESCRIBE: &str = "describe";

/// First sentence of a tool description, without the generated MCP prefix.
fn first_sentence(description: &str) -> String {
    let mut text = description.trim();
    if text.starts_with("MCP tool `")
        && let Some(end) = text.find("`.")
    {
        let rest = &text[end + 2..];
        // Skip the "from server `x`." clause when present.
        text = rest.trim_start();
    }
    let line = text.lines().next().unwrap_or("");
    let end = line
        .match_indices(". ")
        .next()
        .map_or(line.len(), |(index, _)| index + 1);
    let sentence = line[..end].trim();
    if sentence.chars().count() <= MAX_SUMMARY_CHARS {
        sentence.to_owned()
    } else {
        let cut: String = sentence.chars().take(MAX_SUMMARY_CHARS - 1).collect();
        format!("{}…", cut.trim_end())
    }
}

/// Compact signature such as `read(path, offset?, limit?)`; `?` is optional.
fn signature(name: &str, parameters: &Value) -> String {
    let required: Vec<&str> = parameters
        .get("required")
        .and_then(Value::as_array)
        .map(|names| names.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    let mut parts = Vec::new();
    if let Some(properties) = parameters.get("properties").and_then(Value::as_object) {
        for (index, key) in properties.keys().enumerate() {
            if index == MAX_SIGNATURE_PARAMS {
                parts.push("…".to_owned());
                break;
            }
            parts.push(if required.contains(&key.as_str()) {
                key.clone()
            } else {
                format!("{key}?")
            });
        }
    }
    format!("{name}({})", parts.join(", "))
}

/// Bytes for the whole server roster line; every server always appears.
const ROSTER_BUDGET: usize = 1024;
/// Most tool names (and bytes) listed for one server in the roster.
const ROSTER_MAX_NAMES: usize = 8;
const ROSTER_SERVER_BUDGET: usize = 300;
/// Full index lines every MCP server may show regardless of its fair share.
const MIN_LINES_PER_SERVER: usize = 3;

/// Servers in first-seen order with their entry indexes.
fn server_groups(entries: &[Entry]) -> Vec<(&str, Vec<usize>)> {
    let mut groups: Vec<(&str, Vec<usize>)> = Vec::new();
    for (index, entry) in entries.iter().enumerate() {
        let Some(origin) = &entry.origin else {
            continue;
        };
        match groups.iter_mut().find(|(name, _)| *name == origin.server) {
            Some((_, members)) => members.push(index),
            None => groups.push((&origin.server, vec![index])),
        }
    }
    groups
}

/// First line of untrusted server text, stripped of control characters and
/// capped, for the one-line roster.
fn roster_reason(text: &str) -> String {
    let line: String = text
        .lines()
        .find(|line| !line.trim().is_empty())
        .unwrap_or("")
        .chars()
        .filter(|c| !c.is_control())
        .collect();
    let line = line.trim();
    if line.chars().count() <= 120 {
        line.to_owned()
    } else {
        let cut: String = line.chars().take(119).collect();
        format!("{}…", cut.trim_end())
    }
}

/// One roster line naming every MCP server, independent of the index
/// budgets so a large server cannot hide a small one. Servers that failed to
/// start are listed with the reason. Empty without any MCP server.
fn render_roster(entries: &[Entry], servers: &[McpServerInfo]) -> String {
    let groups = server_groups(entries);
    let short = |name: &str, count: usize| {
        format!("{name} ({count} tool{})", if count == 1 { "" } else { "s" })
    };
    // (name, reason) for servers that failed to start.
    let failed: Vec<(&str, String)> = servers
        .iter()
        .filter_map(|server| {
            let reason = server.failure.as_deref()?;
            Some((server.name.as_str(), roster_reason(reason)))
        })
        .collect();
    let failed_short = |name: &str| format!("{name} (failed to start)");
    if groups.is_empty() && failed.is_empty() {
        return String::new();
    }
    // Reserve room for the short form of servers not yet rendered.
    let mut reserve: usize = groups
        .iter()
        .map(|(name, members)| short(name, members.len()).len() + 2)
        .sum::<usize>()
        + failed
            .iter()
            .map(|(name, _)| failed_short(name).len() + 2)
            .sum::<usize>();
    let mut parts = Vec::new();
    let mut used = "MCP servers: \n".len();
    for (name, members) in &groups {
        let short_form = short(name, members.len());
        reserve -= short_form.len() + 2;
        let names: Vec<&str> = members
            .iter()
            .filter_map(|&index| entries[index].origin.as_ref().map(|o| o.tool.as_str()))
            .collect();
        let detailed = format!("{name} ({}: {})", members.len(), names.join(", "));
        let part = if members.len() <= ROSTER_MAX_NAMES
            && detailed.len() <= ROSTER_SERVER_BUDGET
            && used + detailed.len() + 2 + reserve <= ROSTER_BUDGET
        {
            detailed
        } else {
            short_form
        };
        used += part.len() + 2;
        parts.push(part);
    }
    for (name, reason) in &failed {
        let short_form = failed_short(name);
        reserve -= short_form.len() + 2;
        let detailed = format!("{name} (failed to start: {reason})");
        let part = if used + detailed.len() + 2 + reserve <= ROSTER_BUDGET {
            detailed
        } else {
            short_form
        };
        used += part.len() + 2;
        parts.push(part);
    }
    format!("MCP servers: {}\n", parts.join(", "))
}

/// Render the model-facing index within its byte budgets. Built-ins come
/// first; each MCP server then gets a fair share of the budget before any one
/// server may use the remainder, and the names-only tail interleaves servers.
fn render_index(entries: &[Entry], parameters: &[String]) -> String {
    let lines: Vec<String> = entries
        .iter()
        .zip(parameters)
        .map(|(entry, signature)| {
            if entry.summary.is_empty() {
                format!("- {signature}\n")
            } else {
                format!("- {signature} - {}\n", entry.summary)
            }
        })
        .collect();
    // Bucket 0 holds built-ins; the rest are MCP servers in first-seen order.
    let mut buckets: Vec<Vec<usize>> = vec![
        (0..entries.len())
            .filter(|&index| entries[index].origin.is_none())
            .collect(),
    ];
    buckets.extend(
        server_groups(entries)
            .into_iter()
            .map(|(_, members)| members),
    );
    let mut shown = vec![false; entries.len()];
    // Next unconsidered position in each bucket.
    let mut next = vec![0usize; buckets.len()];
    let mut total = 0usize;
    // Pass 1: guaranteed shares. Built-ins may use half the budget so they
    // cannot starve servers; each server gets an equal slice of the rest.
    let servers = buckets.len() - 1;
    for (bucket, members) in buckets.iter().enumerate() {
        let share = if bucket == 0 {
            INDEX_BUDGET / 2
        } else {
            INDEX_BUDGET.saturating_sub(total) / (servers - bucket + 1)
        };
        let (mut used, mut count) = (0usize, 0usize);
        while let Some(&index) = members.get(next[bucket]) {
            let len = lines[index].len();
            let within = used + len <= share || (bucket > 0 && count < MIN_LINES_PER_SERVER);
            if !within || total + len > INDEX_BUDGET {
                break;
            }
            shown[index] = true;
            used += len;
            total += len;
            count += 1;
            next[bucket] += 1;
        }
    }
    // Pass 2: round-robin one line at a time until nothing else fits.
    let mut done = vec![false; buckets.len()];
    while done.iter().any(|done| !done) {
        for (bucket, members) in buckets.iter().enumerate() {
            if done[bucket] {
                continue;
            }
            match members.get(next[bucket]) {
                Some(&index) if total + lines[index].len() <= INDEX_BUDGET => {
                    shown[index] = true;
                    total += lines[index].len();
                    next[bucket] += 1;
                }
                _ => done[bucket] = true,
            }
        }
    }
    let mut out = String::new();
    for members in &buckets {
        for &index in members {
            if shown[index] {
                out.push_str(&lines[index]);
            }
        }
    }
    // Tail: remaining names, interleaved per bucket.
    let rest: Vec<Vec<&str>> = buckets
        .iter()
        .map(|members| {
            members
                .iter()
                .filter(|&&index| !shown[index])
                .map(|&index| entries[index].name.as_str())
                .collect()
        })
        .collect();
    let tail_len: usize = rest.iter().map(Vec::len).sum();
    if tail_len > 0 {
        let mut names = String::new();
        let mut count = 0;
        'fill: for round in 0.. {
            let mut any = false;
            for bucket in &rest {
                let Some(name) = bucket.get(round) else {
                    continue;
                };
                any = true;
                if names.len() + name.len() + 2 > INDEX_TAIL_BUDGET {
                    break 'fill;
                }
                if count > 0 {
                    names.push_str(", ");
                }
                names.push_str(name);
                count += 1;
            }
            if !any {
                break;
            }
        }
        out.push_str(&format!("More (names only; use describe): {names}"));
        if count < tail_len {
            out.push_str(&format!(" and {} more in TOOLS", tail_len - count));
        }
        out.push('\n');
    }
    out
}

impl Codemode {
    pub(crate) fn new<'a>(
        tools: impl IntoIterator<Item = (&'a ToolSpec, Option<Value>, Option<McpOrigin>)>,
        servers: &[McpServerInfo],
    ) -> Result<Self, String> {
        let mut names = Vec::new();
        let mut entries = Vec::new();
        let mut signatures = Vec::new();
        let mut total = 0usize;
        for (spec, output_schema, origin) in tools {
            let definition = &spec.definition;
            if definition.name == GLOBAL_DESCRIBE {
                return Err(format!(
                    "tool name is reserved by codemode: {GLOBAL_DESCRIBE}"
                ));
            }
            let full = json!({
                "name": definition.name,
                "description": definition.description,
                "parameters": definition.parameters,
                "output_schema": output_schema,
            });
            // Bound the catalogue as it grows; the full text is kept host-side.
            total = total.saturating_add(full.to_string().len());
            if total > MAX_CATALOG_BYTES {
                return Err("codemode catalogue exceeds 512 KiB; select fewer tools".into());
            }
            signatures.push(signature(&definition.name, &definition.parameters));
            entries.push(Entry {
                name: definition.name.clone(),
                summary: first_sentence(&definition.description),
                full,
                origin,
            });
            names.push(definition.name.clone());
        }
        if names.is_empty() {
            return Err("no eligible tools selected".into());
        }
        let index = format!(
            "{}{}",
            render_roster(&entries, servers),
            render_index(&entries, &signatures)
        );
        let mut guidelines = vec![
            "Use codemode instead of bash for Python computation; batch dependent tool calls into one snippet and return only what the answer needs. Do not blindly replay a failed snippet: earlier calls may have changed state.".to_owned(),
        ];
        if entries.iter().any(|entry| entry.origin.is_some()) {
            guidelines.push(
                "When the user names a tool, service or MCP server, or a task fits one in the MCP server roster, call it via codemode (`tools[...]`) instead of a shell CLI of the same name. If unsure, inspect `TOOLS` before using bash.".to_owned(),
            );
        }
        let spec = ToolSpec {
            definition: ToolDefinition {
                name: CODEMODE_TOOL_NAME.into(),
                description: format!(
                    concat!(
                        "Run Python (Monty) that calls selected tools as functions; prefer it over bash for Python computation. ",
                        "Put dependent calls, loops and filtering in ONE snippet. Call tools[\"name\"](**kwargs): keyword arguments only, synchronous, no await. ",
                        "Structured tools return dict/list/scalars, others return str; a failed tool raises RuntimeError. ",
                        "The last expression is the result; print() output also enters context, so return only the facts needed. ",
                        "Globals: TOOLS (name -> summary), SERVERS (MCP server -> instructions, tools, status), describe(name) (full description, parameters and output schema; use it when arguments are unclear), MODULES (importable and unavailable modules). ",
                        "No file, env, network or OS access, no third-party packages, no state between calls. ",
                        "Limits: 32 KiB code, 50 tool calls, 50 KiB per value, 5 s compute, 120 s total. ",
                        "Completed tool calls are not rolled back on failure; inspect before replaying. ",
                        "Tools (`?` = optional argument):\n{}"
                    ),
                    index
                ),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "Python source for the whole task. Return a concise final expression."},
                        "description": {"type": "string", "minLength": 1, "maxLength": 80, "description": "Short, single-line action label describing the purpose."}
                    },
                    "required": ["code", "description"],
                    "additionalProperties": false
                }),
            },
            prompt: ToolPrompt::new(
                "Run Python that calls selected tools as functions (see its description)",
                guidelines,
            ),
        };
        Ok(Self {
            names,
            entries,
            servers: servers.to_vec(),
            spec,
        })
    }

    /// `SERVERS`: server name -> `{instructions, tools, status}`. Instructions
    /// are server-supplied text, already capped and sanitized by the host.
    fn servers_global(&self) -> MontyObject {
        let mut names: Vec<&str> = server_groups(&self.entries)
            .into_iter()
            .map(|(name, _)| name)
            .collect();
        for server in &self.servers {
            if !names.contains(&server.name.as_str()) {
                names.push(&server.name);
            }
        }
        MontyObject::dict(names.into_iter().map(|name| {
            let info = self.servers.iter().find(|server| server.name == name);
            let tools = self
                .entries
                .iter()
                .filter(|entry| entry.origin.as_ref().is_some_and(|o| o.server == name))
                .map(|entry| MontyObject::string(&entry.name));
            let status = match info.and_then(|info| info.failure.as_deref()) {
                Some(reason) => format!("failed to start: {}", roster_reason(reason)),
                None => "ready".to_owned(),
            };
            (
                MontyObject::string(name),
                MontyObject::dict([
                    (
                        MontyObject::string("instructions"),
                        MontyObject::string(
                            info.and_then(|info| info.instructions.as_deref())
                                .unwrap_or(""),
                        ),
                    ),
                    (MontyObject::string("tools"), MontyObject::list(tools)),
                    (MontyObject::string("status"), MontyObject::string(&status)),
                ]),
            )
        }))
    }

    /// Value for a lazily resolved sandbox global, or `None` if undefined.
    fn global(&self, name: &str) -> Option<MontyObject> {
        match name {
            GLOBAL_TOOLS => Some(MontyObject::dict(self.entries.iter().map(|entry| {
                (
                    MontyObject::string(&entry.name),
                    MontyObject::string(&entry.summary),
                )
            }))),
            GLOBAL_SERVERS => Some(self.servers_global()),
            GLOBAL_MODULES => {
                let list =
                    |text: &str| MontyObject::list(text.split(", ").map(MontyObject::string));
                Some(MontyObject::dict([
                    (
                        MontyObject::string("available"),
                        list(super::python::AVAILABLE_MODULES),
                    ),
                    (
                        MontyObject::string("unavailable"),
                        list(super::python::UNAVAILABLE_MODULES),
                    ),
                ]))
            }
            GLOBAL_DESCRIBE => Some(MontyObject::function(GLOBAL_DESCRIBE, None)),
            _ => None,
        }
    }

    /// Add a recovery hint to an unknown `tools[...]` key. Models see their
    /// top-level tools with a provider-specific `_` prefix (and tools such as
    /// `bash` that are not callable from a snippet at all), so a bare
    /// `KeyError` makes them guess again.
    fn explain(&self, error: String) -> String {
        let Some(key) = error
            .lines()
            .rev()
            .find_map(|line| line.trim().strip_prefix("KeyError: "))
            .map(|key| key.trim().trim_matches(['\'', '"']).to_owned())
        else {
            return error;
        };
        let bare = key.trim_start_matches('_');
        let hint = if self.contains(bare) {
            format!("use tools[\"{bare}\"] (codemode tool names have no `_` prefix)")
        } else {
            format!(
                "`{bare}` is not callable from codemode; call it as a separate top-level tool. Available: {}",
                self.names.join(", ")
            )
        };
        format!("{error}\nhint: {hint}")
    }

    /// Host-side metadata lookup. It cannot run a tool, so it needs no audit
    /// record and does not count toward the nested-call limit.
    fn describe(&self, args: &monty_types::CallArgs) -> Result<MontyObject, String> {
        let mut positional = args.args();
        let mut keywords = args.kwargs();
        let name = match (positional.len(), keywords.len()) {
            (1, 0) => positional.next().and_then(|value| value.as_str()),
            (0, 1) => keywords
                .next()
                .filter(|(key, _)| key.as_str() == Some("name"))
                .and_then(|(_, value)| value.as_str()),
            _ => None,
        }
        .ok_or("describe(name) takes one tool name string")?;
        let entry = self
            .entries
            .iter()
            .find(|entry| entry.name == name)
            .ok_or_else(|| format!("unknown tool: {name}; see TOOLS"))?;
        Ok(metadata_to_monty(&entry.full, 0))
    }

    pub(crate) fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    pub(crate) fn contains(&self, name: &str) -> bool {
        self.names.iter().any(|allowed| allowed == name)
    }

    pub(crate) async fn execute(
        &self,
        args: Value,
        cancel: CancellationToken,
        executor: &dyn CodemodeExecutor,
    ) -> ToolOutput {
        let summary = summary(&args);
        if super::python::description(&args).is_none() {
            return failure(
                "description must be a non-empty, single-line label of at most 80 characters",
                summary,
            );
        }
        let Some(code) = args.get("code").and_then(Value::as_str) else {
            return failure("missing required argument: code", summary);
        };
        if code.len() > MAX_CODE_BYTES {
            return failure("code exceeds 32 KiB", summary);
        }
        // A dropped/expired workflow can leave a pure interpreter step running
        // until its cooperative compute limit. It cannot dispatch host tools.
        let workflow_cancel = cancel.child_token();
        let _cancel_on_drop = workflow_cancel.clone().drop_guard();
        let printed: Printed = Arc::new(Mutex::new(String::new()));
        let mut audit = Vec::new();
        let mut salvaged = Vec::new();
        let deadline = tokio::time::Instant::now() + WALL_LIMIT;
        let result = tokio::select! {
            biased;
            _ = workflow_cancel.cancelled() => Err("cancelled; an unfinished nested call may have applied changes".into()),
            result = tokio::time::timeout_at(deadline, self.drive(code.to_owned(), &printed, &mut audit, &mut salvaged, &workflow_cancel, executor)) => {
                result.unwrap_or_else(|_| Err("codemode elapsed-time limit exceeded; an unfinished nested call may have applied changes".into()))
            }
        };
        let value = match result {
            Ok(value) => value,
            Err(error) => {
                let error = self.explain(error);
                // The dropped drive future may have left an interpreter step
                // running; recover what it had printed so far.
                let printed = recover_printed(&printed).await;
                return failure(&format_error(&error, &audit, &salvaged, &printed), summary);
            }
        };
        // Formatting is also computation: shared values can have enormous
        // representations. Keep it off Tokio and stop on the first full write.
        let printed = recover_printed(&printed).await;
        let printed_for_error = printed.clone();
        let render = tokio::task::spawn_blocking(move || {
            let mut output = super::python::BoundedOutput::new(String::new());
            if value != MontyObject::none() {
                let _ = writeln!(output, "Result: {value}");
            }
            let _ = output.write_str(&printed);
            output.finish()
        });
        let content = tokio::select! {
            biased;
            _ = workflow_cancel.cancelled() => Err("cancelled while formatting the result".to_owned()),
            result = tokio::time::timeout_at(deadline, render) => match result {
                Ok(Ok(content)) => Ok(content),
                Ok(Err(error)) => Err(format!("result formatter failed: {error}")),
                Err(_) => Err("codemode elapsed-time limit exceeded while formatting the result".into()),
            }
        };
        match content {
            Ok(content) => ToolOutput {
                content,
                is_error: false,
                summary,
            },
            Err(error) => failure(
                &format_error(&error, &audit, &salvaged, &printed_for_error),
                summary,
            ),
        }
    }

    async fn drive(
        &self,
        code: String,
        printed: &Printed,
        audit: &mut Vec<String>,
        salvaged: &mut Vec<String>,
        cancel: &CancellationToken,
        executor: &dyn CodemodeExecutor,
    ) -> Result<MontyObject, String> {
        let inputs = MontyObject::dict(
            self.names
                .iter()
                .map(|name| (MontyObject::string(name), MontyObject::function(name, None))),
        );
        let mut progress = step(printed, move |print| {
            MontyRun::new(
                code,
                "<codemode>",
                vec!["tools".into()],
                CompileOptions::default(),
            )?
            .with_os_policy(monty_types::OsPolicy {
                datetime: monty_types::DateTimeSource::CallHost,
                random_start: monty_types::RandomStart::CallHost,
                sleep: SleepMode::Zero,
                ..monty_types::OsPolicy::default()
            })
            .start(
                vec![inputs],
                ResourceTracker::new(ResourceLimits {
                    max_feed_duration: Some(COMPUTE_LIMIT),
                    max_turn_duration: Some(COMPUTE_LIMIT),
                    max_recursion_depth: 200,
                    max_suspensions: MAX_SUSPENSIONS,
                    ..ResourceLimits::default()
                }),
                print,
            )
        })
        .await?;
        let mut suspensions = 0;
        loop {
            if !matches!(progress, RunProgress::Complete(_)) {
                suspensions += 1;
                if suspensions > MAX_SUSPENSIONS {
                    return Err("codemode suspension limit exceeded (1000); inspect partial results before retrying".into());
                }
            }
            if cancel.is_cancelled() {
                return Err("cancelled; execution status of unfinished tools is unknown".into());
            }
            progress = match progress {
                RunProgress::Complete(value) => return Ok(value),
                RunProgress::FunctionCall(call) => {
                    if call.function_name == GLOBAL_DESCRIBE && call.object_id.is_none() {
                        let result = match self.describe(&call.args) {
                            Ok(value) => ExtFunctionResult::Return(value),
                            Err(error) => exception(&error),
                        };
                        step(printed, move |print| call.resume(result, print)).await?
                    } else if !self.contains(&call.function_name) || call.object_id.is_some() {
                        step(printed, move |print| {
                            call.resume(exception("tool is not available in codemode"), print)
                        })
                        .await?
                    } else if audit.len() >= MAX_TOOL_CALLS {
                        step(printed, move |print| call.resume(exception("nested tool-call limit exceeded (50); inspect partial results before retrying"), print)).await?
                    } else {
                        let arguments = match arguments(&call.args) {
                            Ok(arguments) => arguments,
                            Err(error) => {
                                progress = step(printed, move |print| {
                                    call.resume(exception(&error), print)
                                })
                                .await?;
                                continue;
                            }
                        };
                        let name = call.function_name.clone();
                        let call_id = call.call_id.to_string();
                        // The host may persist the start and then lose the
                        // acknowledgement (cancel/admission race), so the
                        // audit entry must exist before the record attempt to
                        // stay a superset of the durable log. Its text claims
                        // nothing beyond "unknown" until a result is recorded.
                        audit.push(format!(
                            "{}: {name} — execution status unknown",
                            audit.len() + 1
                        ));
                        executor
                            .record(CodemodeEvent::Started {
                                call_id: call_id.clone(),
                                name: name.clone(),
                                arguments: arguments.clone(),
                            })
                            .await?;
                        // Check again after the acknowledgement: cancellation
                        // between durable admission and execution must fail closed.
                        if cancel.is_cancelled() {
                            return Err("cancelled before nested tool execution".into());
                        }
                        let (mut output, structured) =
                            executor.call(&name, arguments, cancel.child_token()).await;
                        if output.content.len() > MAX_VALUE_BYTES {
                            output
                                .content
                                .truncate(output.content.floor_char_boundary(MAX_VALUE_BYTES));
                            output.content.push_str("\n[output truncated]");
                        }
                        if !output.is_error {
                            let used: usize = salvaged.iter().map(String::len).sum();
                            let room = MAX_SALVAGED_BYTES.saturating_sub(used);
                            if room > 0 {
                                let mut text = output.content.clone();
                                if text.len() > room {
                                    text.truncate(text.floor_char_boundary(room));
                                    text.push_str("\n[truncated]");
                                }
                                salvaged.push(format!("{}: {name}\n{text}", audit.len()));
                            }
                        }
                        let result = if output.is_error {
                            exception(&output.content)
                        } else {
                            match structured {
                                Some(value) => match json_to_monty(&value, 0) {
                                    Ok(value) => ExtFunctionResult::Return(value),
                                    // The call succeeded (and may have had
                                    // side effects). Raising would invite the
                                    // code to retry it, so return the display
                                    // text with a note instead.
                                    Err(error) => {
                                        ExtFunctionResult::Return(MontyObject::string(format!(
                                            "{}\n[structured result not returned: {error}; this is the display text]",
                                            output.content
                                        )))
                                    }
                                },
                                None => {
                                    ExtFunctionResult::Return(MontyObject::string(&output.content))
                                }
                            }
                        };
                        let status = if output.is_error {
                            "returned error"
                        } else {
                            "returned"
                        };
                        let finished = format!("{}: {name} — {status}", audit.len());
                        executor
                            .record(CodemodeEvent::Finished {
                                call_id,
                                name,
                                output,
                            })
                            .await?;
                        // Only claim a result once it is durable; if the
                        // record above fails the log has no result for this
                        // call, so the entry stays "execution status unknown".
                        *audit.last_mut().expect("admitted call") = finished;
                        step(printed, move |print| call.resume(result, print)).await?
                    }
                }
                RunProgress::NameLookup(lookup) => {
                    // Resolve only the documented globals, and only for plain
                    // names, not attribute lookups on host objects.
                    let value = lookup
                        .object_id()
                        .is_none()
                        .then(|| self.global(&lookup.name))
                        .flatten();
                    step(printed, move |print| {
                        lookup.resume(
                            value.map_or(NameLookupResult::Undefined, NameLookupResult::Value),
                            print,
                        )
                    })
                    .await?
                }
                RunProgress::OsCall(call) => {
                    step(printed, move |print| {
                        call.resume(
                            exception("direct OS access is disabled; use an available tool"),
                            print,
                        )
                    })
                    .await?
                }
                RunProgress::ResolveFutures(_) => {
                    return Err(
                        "async tool calls are not supported; call tools synchronously".into(),
                    );
                }
            };
        }
    }
}

/// Purpose label shared by live execution and session replay.
pub(crate) fn summary(args: &Value) -> String {
    super::python::description(args).map_or_else(
        || CODEMODE_TOOL_NAME.into(),
        |label| format!("{CODEMODE_TOOL_NAME} {label}"),
    )
}

fn format_error(error: &str, audit: &[String], salvaged: &[String], printed: &str) -> String {
    let mut output = super::python::BoundedOutput::new(String::new());
    // Keep recovery information ahead of supplementary output. A full print
    // buffer must never hide completed mutations or the failure reason.
    let _ = writeln!(
        output,
        "{} nested tool call(s) started. Changes are not rolled back; inspect before retrying.",
        audit.len()
    );
    for entry in audit {
        let _ = writeln!(output, "{entry}");
    }
    let _ = writeln!(output, "{error}");
    if !salvaged.is_empty() {
        let _ = writeln!(
            output,
            "Results of calls that succeeded before the failure (do not replay them):"
        );
        for entry in salvaged {
            let _ = writeln!(output, "{entry}");
        }
    }
    let _ = output.write_str(printed);
    output.finish()
}

fn failure(content: &str, summary: String) -> ToolOutput {
    ToolOutput {
        content: content.into(),
        is_error: true,
        summary,
    }
}

fn exception(message: &str) -> ExtFunctionResult {
    ExtFunctionResult::Error(MontyException::new(
        ExcType::RuntimeError,
        Some(message.into()),
    ))
}

// Only this synchronous phase enters the interpreter. No executor or registry
// is moved into spawn_blocking, so dropping its caller cannot grant late tools.
/// Interpreter output shared with the blocking step so that a cancelled or
/// timed-out workflow can still recover what was printed before it stopped.
type Printed = Arc<Mutex<String>>;

/// Appends each print fragment to the shared buffer immediately (locking only
/// per write), so a step that is still computing when the workflow is
/// cancelled has already made its earlier output visible. Applies the same
/// byte cap as `PrintWriter::CollectString`.
struct SharedPrint(Printed);

impl SharedPrint {
    fn append(&mut self, text: &str) -> Result<(), MontyException> {
        let mut buffer = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        monty_types::check_print_collect_limit(buffer.len(), text.len(), Some(MAX_VALUE_BYTES))?;
        buffer.push_str(text);
        Ok(())
    }
}

impl monty_types::PrintWriterCallback for SharedPrint {
    fn stdout_write(&mut self, output: std::borrow::Cow<'_, str>) -> Result<(), MontyException> {
        self.append(&output)
    }

    fn stdout_push(&mut self, end: char) -> Result<(), MontyException> {
        self.append(end.encode_utf8(&mut [0; 4]))
    }
}

/// Take the printed text. The lock is only held per write, so this normally
/// succeeds at once; a bounded retry and a note guard against a poisoned or
/// contended buffer.
async fn recover_printed(printed: &Printed) -> String {
    for _ in 0..20 {
        // Scope the lock result so no guard is held across the sleep.
        let taken = match printed.try_lock() {
            Ok(mut guard) => Some(std::mem::take(&mut *guard)),
            Err(std::sync::TryLockError::Poisoned(poisoned)) => {
                Some(std::mem::take(&mut *poisoned.into_inner()))
            }
            Err(std::sync::TryLockError::WouldBlock) => None,
        };
        if let Some(text) = taken {
            return text;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    "[output of the interrupted interpreter step is unavailable]\n".into()
}

async fn step(
    printed: &Printed,
    run: impl FnOnce(PrintWriter<'_>) -> Result<RunProgress, MontyException> + Send + 'static,
) -> Result<RunProgress, String> {
    let mut sink = SharedPrint(Arc::clone(printed));
    let result = tokio::task::spawn_blocking(move || run(PrintWriter::Callback(&mut sink)))
        .await
        .map_err(|error| format!("interpreter task failed: {error}"))?;
    result.map_err(|error| error.to_string())
}

fn arguments(args: &monty_types::CallArgs) -> Result<Value, String> {
    if args.args().len() != 0 {
        return Err("tool functions accept keyword arguments only".into());
    }
    let mut values = Map::new();
    let mut budget = MAX_VALUE_BYTES;
    for (key, value) in args.kwargs() {
        let key = key.as_str().ok_or("tool keyword names must be strings")?;
        spend(&mut budget, key.len())?;
        values.insert(key.into(), monty_to_json(value, 0, &mut budget)?);
    }
    let value = Value::Object(values);
    if value.to_string().len() > MAX_VALUE_BYTES {
        return Err("tool arguments exceed 50 KiB".into());
    }
    Ok(value)
}

fn spend(budget: &mut usize, size: usize) -> Result<(), String> {
    *budget = budget
        .checked_sub(size.max(1))
        .ok_or("tool arguments exceed 50 KiB")?;
    Ok(())
}

fn monty_to_json(value: ObjectRef<'_>, depth: usize, budget: &mut usize) -> Result<Value, String> {
    if depth > 64 {
        return Err("tool arguments are cyclic or exceed 64 levels".into());
    }
    spend(budget, 1)?;
    match value.type_name() {
        "NoneType" => Ok(Value::Null),
        "bool" => Ok(Value::Bool(value.as_bool().ok_or("invalid bool")?)),
        "int" => value
            .as_int()
            .map(|value| json!(value))
            .ok_or("tool integers must fit in signed 64 bits".into()),
        "float" => serde_json::Number::from_f64(value.as_float().ok_or("invalid float")?)
            .map(Value::Number)
            .ok_or("tool numbers must be finite".into()),
        "str" => {
            let text = value.as_str().ok_or("invalid string")?;
            spend(budget, text.len())?;
            Ok(Value::String(text.into()))
        }
        "list" | "tuple" => {
            let items = value.items().ok_or("invalid sequence")?;
            if items.len() > *budget {
                return Err("tool arguments exceed 50 KiB".into());
            }
            items
                .into_iter()
                .map(|item| monty_to_json(item, depth + 1, budget))
                .collect::<Result<Vec<_>, _>>()
                .map(Value::Array)
        }
        "dict" => {
            let pairs = value.pairs().ok_or("invalid dictionary")?;
            if pairs.len() > *budget {
                return Err("tool arguments exceed 50 KiB".into());
            }
            let mut values = Map::new();
            for (key, child) in pairs {
                let key = key.as_str().ok_or("tool dictionary keys must be strings")?;
                spend(budget, key.len())?;
                values.insert(key.into(), monty_to_json(child, depth + 1, budget)?);
            }
            Ok(Value::Object(values))
        }
        kind => Err(format!(
            "{kind} cannot be passed to a tool; use JSON-compatible values"
        )),
    }
}

/// Convert catalogue metadata for `describe()`. Unlike tool arguments and
/// results, the catalogue is already bounded at registration (512 KiB), and
/// a valid schema must stay readable: never fail on size, depth, or integer
/// range. Values that Python cannot represent faithfully degrade to their JSON
/// text instead of hiding the whole definition.
fn metadata_to_monty(value: &Value, depth: usize) -> MontyObject {
    if depth > 64 {
        return MontyObject::string(value.to_string());
    }
    match value {
        Value::Null => MontyObject::none(),
        Value::Bool(value) => MontyObject::bool(*value),
        Value::Number(number) => {
            if let Some(value) = number.as_i64() {
                MontyObject::int(value)
            } else if let Some(value) = number.as_u64() {
                MontyObject::bigint(num_bigint::BigInt::from(value))
            } else if let Some(value) = number.as_f64() {
                MontyObject::float(value)
            } else {
                MontyObject::string(number.to_string())
            }
        }
        Value::String(value) => MontyObject::string(value),
        Value::Array(values) => MontyObject::list(
            values
                .iter()
                .map(|value| metadata_to_monty(value, depth + 1))
                .collect::<Vec<_>>(),
        ),
        Value::Object(values) => MontyObject::dict(
            values
                .iter()
                .map(|(key, value)| {
                    (
                        MontyObject::string(key),
                        metadata_to_monty(value, depth + 1),
                    )
                })
                .collect::<Vec<_>>(),
        ),
    }
}

fn json_to_monty(value: &Value, depth: usize) -> Result<MontyObject, String> {
    if depth > 64 {
        return Err("structured tool result exceeds 64 levels".into());
    }
    // The wire size is bounded before recursive construction, including JSON
    // escaping. Integer conversion must not silently lose precision.
    if depth == 0 && value.to_string().len() > MAX_VALUE_BYTES {
        return Err("structured tool result exceeds 50 KiB".into());
    }
    Ok(match value {
        Value::Null => MontyObject::none(),
        Value::Bool(value) => MontyObject::bool(*value),
        Value::Number(value) => {
            if let Some(value) = value.as_i64() {
                MontyObject::int(value)
            } else if value.is_u64() {
                return Err("structured tool integer exceeds signed 64 bits".into());
            } else {
                MontyObject::float(value.as_f64().ok_or("invalid result number")?)
            }
        }
        Value::String(value) => MontyObject::string(value),
        Value::Array(values) => MontyObject::list(
            values
                .iter()
                .map(|value| json_to_monty(value, depth + 1))
                .collect::<Result<Vec<_>, _>>()?,
        ),
        Value::Object(values) => MontyObject::dict(
            values
                .iter()
                .map(|(key, value)| {
                    Ok((MontyObject::string(key), json_to_monty(value, depth + 1)?))
                })
                .collect::<Result<Vec<_>, String>>()?,
        ),
    })
}
