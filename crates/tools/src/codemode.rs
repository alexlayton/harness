//! Bounded, fresh-state Monty orchestration. Only the async host driver can
//! admit tool calls; a detached interpreter step has no host authority.

use super::{ToolOutput, ToolPrompt, ToolSpec};
use async_trait::async_trait;
use llm::ToolDefinition;
use monty::{MontyRun, RunProgress};
use monty_types::{
    CompileOptions, ExcType, ExtFunctionResult, MontyException, MontyObject, NameLookupResult,
    ObjectRef, PrintWriter, ResourceLimits, ResourceTracker, SleepMode,
};
use serde_json::{Map, Value, json};
use std::fmt::Write;
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
    spec: ToolSpec,
}

/// Full catalogue entry, served lazily by `describe(name)`.
struct Entry {
    name: String,
    summary: String,
    full: Value,
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

/// Render the model-facing index within its byte budgets.
fn render_index(entries: &[Entry], parameters: &[String]) -> String {
    let mut out = String::new();
    let mut tail: Vec<&str> = Vec::new();
    for (entry, signature) in entries.iter().zip(parameters) {
        let line = if entry.summary.is_empty() {
            format!("- {signature}\n")
        } else {
            format!("- {signature} - {}\n", entry.summary)
        };
        if tail.is_empty() && out.len() + line.len() <= INDEX_BUDGET {
            out.push_str(&line);
        } else {
            tail.push(&entry.name);
        }
    }
    if !tail.is_empty() {
        let mut names = String::new();
        let mut shown = 0;
        for name in &tail {
            if names.len() + name.len() + 2 > INDEX_TAIL_BUDGET {
                break;
            }
            if shown > 0 {
                names.push_str(", ");
            }
            names.push_str(name);
            shown += 1;
        }
        out.push_str(&format!("More (names only; use describe): {names}"));
        if shown < tail.len() {
            out.push_str(&format!(" and {} more in TOOLS", tail.len() - shown));
        }
        out.push('\n');
    }
    out
}

impl Codemode {
    pub(crate) fn new<'a>(
        tools: impl IntoIterator<Item = (&'a ToolSpec, Option<Value>)>,
    ) -> Result<Self, String> {
        let mut names = Vec::new();
        let mut entries = Vec::new();
        let mut signatures = Vec::new();
        let mut total = 0usize;
        for (spec, output_schema) in tools {
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
            });
            names.push(definition.name.clone());
        }
        if names.is_empty() {
            return Err("no eligible tools selected".into());
        }
        let index = render_index(&entries, &signatures);
        let spec = ToolSpec {
            definition: ToolDefinition {
                name: CODEMODE_TOOL_NAME.into(),
                description: format!(
                    concat!(
                        "Run Python (Monty) that calls selected tools as functions; prefer it over bash for Python computation. ",
                        "Put dependent calls, loops and filtering in ONE snippet. Call tools[\"name\"](**kwargs): keyword arguments only, synchronous, no await. ",
                        "Structured tools return dict/list/scalars, others return str; a failed tool raises RuntimeError. ",
                        "The last expression is the result; print() output also enters context, so return only the facts needed. ",
                        "Globals: TOOLS (name -> summary), describe(name) (full description, parameters and output schema; use it when arguments are unclear), MODULES (importable and unavailable modules). ",
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
                [
                    "Use codemode instead of bash for Python computation; batch dependent tool calls into one snippet and return only what the answer needs. Do not blindly replay a failed snippet: earlier calls may have changed state.",
                ],
            ),
        };
        Ok(Self {
            names,
            entries,
            spec,
        })
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
        let mut printed = String::new();
        let mut audit = Vec::new();
        let mut salvaged = Vec::new();
        let deadline = tokio::time::Instant::now() + WALL_LIMIT;
        let result = tokio::select! {
            biased;
            _ = workflow_cancel.cancelled() => Err("cancelled; an unfinished nested call may have applied changes".into()),
            result = tokio::time::timeout_at(deadline, self.drive(code.to_owned(), &mut printed, &mut audit, &mut salvaged, &workflow_cancel, executor)) => {
                result.unwrap_or_else(|_| Err("codemode elapsed-time limit exceeded; an unfinished nested call may have applied changes".into()))
            }
        };
        let value = match result {
            Ok(value) => value,
            Err(error) => {
                let error = self.explain(error);
                return failure(&format_error(&error, &audit, &salvaged, &printed), summary);
            }
        };
        // Formatting is also computation: shared values can have enormous
        // representations. Keep it off Tokio and stop on the first full write.
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
            Err(error) => failure(&format_error(&error, &audit, &salvaged, ""), summary),
        }
    }

    async fn drive(
        &self,
        code: String,
        printed: &mut String,
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
                        audit.push(format!(
                            "{}: {name} — execution status unknown",
                            audit.len() + 1
                        ));
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
                        *audit.last_mut().expect("admitted call") =
                            format!("{}: {name} — {status}", audit.len());
                        executor
                            .record(CodemodeEvent::Finished {
                                call_id,
                                name,
                                output,
                            })
                            .await?;
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
async fn step(
    printed: &mut String,
    run: impl FnOnce(PrintWriter<'_>) -> Result<RunProgress, MontyException> + Send + 'static,
) -> Result<RunProgress, String> {
    let mut output = std::mem::take(printed);
    let (result, output) = tokio::task::spawn_blocking(move || {
        let result = run(PrintWriter::CollectString(
            &mut output,
            Some(MAX_VALUE_BYTES),
        ));
        (result, output)
    })
    .await
    .map_err(|error| format!("interpreter task failed: {error}"))?;
    *printed = output;
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
