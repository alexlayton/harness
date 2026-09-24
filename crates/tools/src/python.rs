//! Small, filesystem-free Python calculations through Monty.
//!
//! This uses the in-process interpreter to avoid requiring a separately
//! installed worker binary. It is not crash isolation: a Monty process abort
//! also aborts Harness. Never describe this tool as a security boundary.

use super::{Concurrency, Tool, ToolOutput, ToolPrompt, ToolSpec};
use async_trait::async_trait;
use llm::ToolDefinition;
use monty::MontyRun;
use monty_types::{CompileOptions, PrintWriter, ResourceLimits, ResourceTracker, SleepMode};
use serde_json::{Value, json};
use std::fmt::{self, Write};
use std::time::Duration;
use tokio_util::sync::CancellationToken;

const MAX_CODE_BYTES: usize = 32 * 1024;
const MAX_OUTPUT_BYTES: usize = 50 * 1024;
const RUN_LIMIT: Duration = Duration::from_secs(5);
// Keep these lists in the tool definition so the model does not need web access
// to discover Monty's fixed import surface (Monty 1.0.0-beta.2).
const AVAILABLE_MODULES: &str = "asyncio, base64, binascii, collections, copy, dataclasses, datetime, functools, itertools, json, math, os, pathlib, random, re, sys, time, typing, unicodedata";
const UNAVAILABLE_MODULES: &str = "abc, argparse, array, bisect, contextlib, csv, ctypes, decimal, enum, fractions, hashlib, heapq, hmac, http, inspect, io, logging, multiprocessing, operator, pickle, queue, socket, string, struct, subprocess, tempfile, threading, traceback, types, typing_extensions, unittest, urllib, uuid, warnings, weakref, zipfile, zlib, _collections_abc, _typeshed";

/// Execute standalone Python snippets in the Monty interpreter.
pub struct PythonTool;

#[async_trait]
impl Tool for PythonTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            definition: ToolDefinition {
                name: "python".into(),
                description: format!(
                    "Run a standalone Python snippet in Monty for calculations and data transformations. Prefer this over bash for Python computation; use dedicated tools for files and bash for commands, tests, builds and git. Each call starts fresh. Provide a short description of the calculation for the visible tool status. Returns print output and the final expression. Available modules (some have limited APIs): {AVAILABLE_MODULES}. Unavailable modules (not exhaustive): {UNAVAILABLE_MODULES}. No third-party packages, filesystem access, or host functions. This in-process interpreter is NOT crash-isolated; do not use it to execute untrusted third-party scripts."
                ),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "code": { "type": "string", "description": "Python source (at most 32 KiB). print() output and the final expression are returned. No state persists between calls." },
                        "description": { "type": "string", "minLength": 1, "maxLength": 80, "description": "Short, single-line action label shown in the tool status, e.g. 'Compute the 100th Fibonacci number'. Describe the purpose, not the Python source." }
                    },
                    "required": ["code", "description"],
                    "additionalProperties": false
                }),
            },
            prompt: ToolPrompt::new(
                "Run short Python calculations (Monty, no filesystem)",
                [
                    "Prefer python over bash for standalone Python calculations; use dedicated file tools for workspace files, and bash for external commands, builds, tests and git. Supply a short description of what this call computes so the tool status is meaningful.",
                    "Python runs in Monty, not CPython: no pip/third-party packages or file access (including via os/pathlib). The tool definition lists available and unavailable modules. Each invocation has fresh globals and a 5-second execution budget.",
                ],
            ),
        }
    }

    fn concurrency(&self, _args: &Value) -> Concurrency {
        // In-process execution is not provably side-effect-free, and the
        // interpreter's resource limits are cooperative rather than a process
        // boundary. Never batch it with workspace reads or other calls.
        Concurrency::Exclusive
    }

    async fn execute(&self, args: Value, cancel: CancellationToken) -> ToolOutput {
        let Some(description) = description(&args) else {
            return error(
                "description must be a non-empty, single-line label of at most 80 characters",
            );
        };
        let summary = format!("python {description}");
        let Some(code) = args.get("code").and_then(Value::as_str) else {
            return error_with_summary("missing required argument: code", summary);
        };
        if code.len() > MAX_CODE_BYTES {
            return error_with_summary("code exceeds 32 KiB", summary);
        }
        if cancel.is_cancelled() {
            return error_with_summary("cancelled", summary);
        }
        let code = code.to_owned();
        // Parsing and execution are synchronous. Monty's duration limit is
        // checked during execution, so do not hold up Tokio's async threads.
        // Dropping this future cannot stop an already running blocking task;
        // its own execution limit is the backstop.
        let task = tokio::task::spawn_blocking(move || run_code(code));
        let mut output = tokio::select! {
            biased;
            _ = cancel.cancelled() => error("cancelled (running code may continue until its execution limit)"),
            result = task => match result {
                Ok(output) => output,
                Err(join_error) => error(&format!("interpreter task failed: {join_error}")),
            },
        };
        output.summary = summary;
        output
    }
}

fn description(args: &Value) -> Option<&str> {
    let description = args.get("description")?.as_str()?.trim();
    (!description.is_empty()
        && description.chars().count() <= 80
        && !description.chars().any(char::is_control))
    .then_some(description)
}

/// The same label is used before dispatch, on completion and in session replay.
pub(crate) fn summary(args: &Value) -> String {
    description(args).map_or_else(|| "python".into(), |label| format!("python {label}"))
}

fn run_code(code: String) -> ToolOutput {
    let runner = match MontyRun::new(code, "<python>", vec![], CompileOptions::default()) {
        Ok(runner) => runner,
        Err(err) => return error(&err.to_string()),
    };
    // No inputs, host callbacks or filesystem mounts: OS/file calls that
    // need a host handler fail instead of inheriting Harness's authority.
    // max_memory is NOT enforced without monty-alloc installed in the binary;
    // do not claim a memory sandbox here.
    let limits = ResourceLimits {
        max_feed_duration: Some(RUN_LIMIT),
        max_turn_duration: Some(RUN_LIMIT),
        max_recursion_depth: 200,
        ..ResourceLimits::default()
    };
    let mut output = String::new();
    let result = runner
        .with_os_policy(monty_types::OsPolicy {
            sleep: SleepMode::Zero,
            ..monty_types::OsPolicy::default()
        })
        .run(
            vec![],
            ResourceTracker::new(limits),
            PrintWriter::CollectString(&mut output, Some(MAX_OUTPUT_BYTES)),
        );
    match result {
        Ok(value) => {
            let mut bounded = BoundedOutput::new(output);
            if value != monty_types::MontyObject::none() {
                if !bounded.text.is_empty() && !bounded.text.ends_with('\n') {
                    let _ = bounded.write_str("\n");
                }
                let _ = bounded.write_str("Result: ");
                let _ = write!(bounded, "{value}");
            }
            ToolOutput {
                content: bounded.finish(),
                is_error: false,
                summary: "python".into(),
            }
        }
        Err(err) => {
            let mut bounded = BoundedOutput::new(output);
            if !bounded.text.is_empty() && !bounded.text.ends_with('\n') {
                let _ = bounded.write_str("\n");
            }
            let _ = write!(bounded, "{err}");
            ToolOutput {
                content: bounded.finish(),
                is_error: true,
                summary: "python".into(),
            }
        }
    }
}

struct BoundedOutput {
    text: String,
    truncated: bool,
}

impl BoundedOutput {
    fn new(text: String) -> Self {
        Self {
            text,
            truncated: false,
        }
    }

    fn finish(mut self) -> String {
        if self.truncated {
            self.text.push_str("\n[output truncated]");
        }
        self.text
    }
}

impl Write for BoundedOutput {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        let remaining = MAX_OUTPUT_BYTES.saturating_sub(self.text.len());
        if s.len() > remaining {
            let boundary = s.floor_char_boundary(remaining);
            self.text.push_str(&s[..boundary]);
            self.truncated = true;
        } else {
            self.text.push_str(s);
        }
        Ok(())
    }
}

fn error(message: &str) -> ToolOutput {
    error_with_summary(message, "python".into())
}

fn error_with_summary(message: &str, summary: String) -> ToolOutput {
    ToolOutput {
        content: message.into(),
        is_error: true,
        summary,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn returns_print_and_final_expression_without_host_files() {
        let tool = PythonTool;
        let output = tool
            .execute(
                json!({"code": "print('hello')\n6 * 7", "description": "Calculate six times seven"}),
                CancellationToken::new(),
            )
            .await;
        assert!(!output.is_error, "{}", output.content);
        assert_eq!(output.content, "hello\nResult: 42");
        assert_eq!(output.summary, "python Calculate six times seven");

        let denied = tool
            .execute(
                json!({"code": "open('/etc/passwd').read()", "description": "Read system file"}),
                CancellationToken::new(),
            )
            .await;
        assert!(denied.is_error);
        assert_eq!(denied.summary, "python Read system file");
    }
}
