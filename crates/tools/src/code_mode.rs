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
use serde::Serialize;
use serde::ser::{SerializeSeq, Serializer};
use serde_json::{Map, Value, json};
use std::fmt::Write;
use std::io;
use std::time::Duration;
use tokio_util::sync::CancellationToken;

const MAX_CODE_BYTES: usize = 32 * 1024;
const MAX_VALUE_BYTES: usize = 50 * 1024;
const MAX_TOOL_CALLS: usize = 50;
const MAX_SUSPENSIONS: usize = 1000;
const COMPUTE_LIMIT: Duration = Duration::from_secs(5);
const WALL_LIMIT: Duration = Duration::from_secs(120);
const MAX_CATALOG_BYTES: usize = 512 * 1024;

#[cfg(test)]
mod tests;

/// An internal tool lifecycle, separate from model-issued tool calls.
#[derive(Clone, Debug)]
pub enum CodeModeEvent {
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
pub trait CodeModeExecutor: Send + Sync {
    /// Execute one selected tool, returning display text and optional JSON.
    async fn call(
        &self,
        name: &str,
        args: Value,
        cancel: CancellationToken,
    ) -> (ToolOutput, Option<Value>);

    /// Record a lifecycle event. A failed acknowledgement stops the snippet;
    /// it is not a catchable Python exception that could permit more calls.
    async fn record(&self, _event: CodeModeEvent) -> Result<(), String> {
        Ok(())
    }
}

/// One immutable catalogue, generated from the same registered tool specs that
/// previously supplied native definitions. The dictionary interface preserves
/// arbitrary MCP names and keyword names without lossy Python aliases.
pub(crate) struct CodeMode {
    pub(crate) names: Vec<String>,
    spec: ToolSpec,
}

#[derive(Serialize)]
struct CatalogEntry<'a> {
    name: &'a str,
    description: &'a str,
    parameters: &'a Value,
    output_schema: Option<Value>,
}

// Stop serialization on overflow, not after constructing an arbitrarily large
// JSON string. Tool specs are borrowed and output schemas are obtained lazily,
// so rejection also avoids copying the rest of the selected catalogue.
struct CatalogWriter(Vec<u8>);

impl io::Write for CatalogWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() > MAX_CATALOG_BYTES.saturating_sub(self.0.len()) {
            return Err(io::Error::new(
                io::ErrorKind::FileTooLarge,
                "code-mode catalogue exceeds 512 KiB; select fewer tools",
            ));
        }
        self.0.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl CodeMode {
    pub(crate) fn new<'a>(
        tools: impl IntoIterator<Item = (&'a ToolSpec, Option<Value>)>,
    ) -> Result<Self, String> {
        let mut writer = CatalogWriter(Vec::new());
        let mut serializer = serde_json::Serializer::pretty(&mut writer);
        let mut catalog = serializer
            .serialize_seq(None)
            .map_err(|err| err.to_string())?;
        let mut names = Vec::new();
        for (spec, output_schema) in tools {
            catalog
                .serialize_element(&CatalogEntry {
                    name: &spec.definition.name,
                    description: &spec.definition.description,
                    parameters: &spec.definition.parameters,
                    output_schema,
                })
                .map_err(|err| err.to_string())?;
            names.push(spec.definition.name.clone());
        }
        if names.is_empty() {
            return Err("no eligible tools selected".into());
        }
        catalog.end().map_err(|err| err.to_string())?;
        let catalog = String::from_utf8(writer.0).map_err(|err| err.to_string())?;
        let spec = ToolSpec {
            definition: ToolDefinition {
                name: "run_code".into(),
                description: format!(
                    concat!(
                        "Execute Python calculations and tool orchestration in Monty with selected Harness tools as functions. ",
                        "Tool calls are optional: calculations and data transformations can use ordinary Python without accessing tools. Prefer run_code over bash for Python computation. ",
                        "Prefer ONE run_code call for a complete, self-contained task when it fits the limits. ",
                        "Plan the workflow before calling: put required preflight checks, repeated lookups, dependent calls, and result filtering in the SAME snippet. ",
                        "The catalogue entries are Python functions, not native tools. Call them only inside run_code; do not issue them as separate tool calls. ",
                        "Use loops for multiple inputs and variables/conditionals to use earlier results in later calls; do not return to the model between steps that Python can handle. ",
                        "Do not add unnecessary service checks or split a task merely to inspect a successful check. ",
                        "Split only when results need model reasoning or a user decision, an unexpected failure needs inspection, or the workflow would exceed limits. Never guess arguments or continue unsafe work just to keep one call. ",
                        "Call tools[\"name\"](**keyword_arguments), using names and arguments from the catalogue below. Calls are synchronous and sequential; do not await them or use asyncio.gather. ",
                        "Return a concise combined result with only the facts needed for the answer. Extract relevant fields or text inside Python instead of printing complete reports, debug logs, or successful preflight output. ",
                        "The final expression is returned; print() is supplementary output and also enters model context. Once the result is sufficient, answer the user directly. ",
                        "Tools with structured JSON return that JSON as Python dict/list/scalars; other tools (and structured results too large to convert) return display text as str. Do not assume text results are JSON. Failed tools raise RuntimeError. ",
                        "Only the listed functions are allowed. No direct filesystem, environment, network, clock, or OS access. No built-in shell, subagents, recursive code execution, third-party packages, or state between calls. ",
                        "Limits: 32 KiB code, 50 nested calls, 50 KiB arguments/results/output, 5 seconds total interpreter execution, 120 seconds elapsed time. ",
                        "On failure inspect the started-call summary before retrying: completed changes are not rolled back. Execution is in-process, not crash-isolated or a memory security boundary. ",
                        "Available functions and their original JSON schemas:\n{}"
                    ),
                    catalog
                ),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "Python source for the complete task: combine required checks, loops, dependent tool calls, and filtering in one snippet when safe and within limits. Access tool functions through the tools dictionary. Return a concise combined final expression, not complete raw reports."},
                        "description": {"type": "string", "minLength": 1, "maxLength": 80, "description": "Short, single-line action label describing the purpose."}
                    },
                    "required": ["code", "description"],
                    "additionalProperties": false
                }),
            },
            prompt: ToolPrompt::new(
                "Run Python calculations and orchestrate selected tools with sequential function calls",
                [
                    "Use run_code for standalone Python calculations and data transformations instead of launching Python with bash. No tool calls are required for pure computation.",
                    "Prefer one run_code call per complete, self-contained task within the execution limits. Include required preflight checks, loops over multiple inputs, dependent calls, and filtering in the same snippet; do not add unnecessary checks or return to the model between steps Python can handle. Split when results need model reasoning, user input, failure inspection, or more execution budget, not merely because tools depend on each other.",
                    "Use tools[\"name\"](**kwargs); nested calls run sequentially and return JSON when available, otherwise text. Return only a concise combined result needed for the answer. Printed text also enters model context: do not print complete reports or successful preflight output unless needed. Once the result is sufficient, answer directly.",
                    "Catalogue entries are Python functions, not native tools. Call them only inside run_code, never as separate tool calls.",
                    "Code-mode state is fresh for each call. On errors or interruption, do not blindly replay a snippet: earlier tool calls can already have changed files or external state. Never continue unsafe work just to keep a single call.",
                ],
            ),
        };
        Ok(Self { names, spec })
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
        executor: &dyn CodeModeExecutor,
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
        let deadline = tokio::time::Instant::now() + WALL_LIMIT;
        let result = tokio::select! {
            biased;
            _ = workflow_cancel.cancelled() => Err("cancelled; an unfinished nested call may have applied changes".into()),
            result = tokio::time::timeout_at(deadline, self.drive(code.to_owned(), &mut printed, &mut audit, &workflow_cancel, executor)) => {
                result.unwrap_or_else(|_| Err("code-mode elapsed-time limit exceeded; an unfinished nested call may have applied changes".into()))
            }
        };
        let value = match result {
            Ok(value) => value,
            Err(error) => return failure(&format_error(&error, &audit, &printed), summary),
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
                Err(_) => Err("code-mode elapsed-time limit exceeded while formatting the result".into()),
            }
        };
        match content {
            Ok(content) => ToolOutput {
                content,
                is_error: false,
                summary,
            },
            Err(error) => failure(&format_error(&error, &audit, ""), summary),
        }
    }

    async fn drive(
        &self,
        code: String,
        printed: &mut String,
        audit: &mut Vec<String>,
        cancel: &CancellationToken,
        executor: &dyn CodeModeExecutor,
    ) -> Result<MontyObject, String> {
        let inputs = MontyObject::dict(
            self.names
                .iter()
                .map(|name| (MontyObject::string(name), MontyObject::function(name, None))),
        );
        let mut progress = step(printed, move |print| {
            MontyRun::new(
                code,
                "<run_code>",
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
                    return Err("code-mode suspension limit exceeded (1000); inspect partial results before retrying".into());
                }
            }
            if cancel.is_cancelled() {
                return Err("cancelled; execution status of unfinished tools is unknown".into());
            }
            progress = match progress {
                RunProgress::Complete(value) => return Ok(value),
                RunProgress::FunctionCall(call) => {
                    if !self.contains(&call.function_name) || call.object_id.is_some() {
                        step(printed, move |print| {
                            call.resume(exception("tool is not available in code mode"), print)
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
                            .record(CodeModeEvent::Started {
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
                                        ExtFunctionResult::Return(MontyObject::string(&format!(
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
                            .record(CodeModeEvent::Finished {
                                call_id,
                                name,
                                output,
                            })
                            .await?;
                        step(printed, move |print| call.resume(result, print)).await?
                    }
                }
                RunProgress::NameLookup(lookup) => {
                    step(printed, move |print| {
                        lookup.resume(NameLookupResult::Undefined, print)
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
    super::python::description(args)
        .map_or_else(|| "run_code".into(), |label| format!("run_code {label}"))
}

fn format_error(error: &str, audit: &[String], printed: &str) -> String {
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
