use super::*;
use crate::{Tool, ToolRegistry};
use std::sync::{Arc, Mutex};
use tokio::sync::Notify;

struct Echo {
    name: &'static str,
    calls: Arc<Mutex<Vec<Value>>>,
    started: Option<Arc<Notify>>,
}

#[async_trait]
impl Tool for Echo {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            definition: ToolDefinition {
                name: self.name.into(),
                description: "Echo structured arguments".into(),
                parameters: json!({"type": "object", "properties": {"value": {"type": "integer"}}}),
            },
            prompt: ToolPrompt::new("Echo", ["Use keyword arguments"]),
        }
    }

    async fn execute(&self, args: Value, _cancel: CancellationToken) -> ToolOutput {
        self.calls.lock().unwrap().push(args.clone());
        if let Some(started) = &self.started {
            started.notify_one();
            std::future::pending::<()>().await;
        }
        ToolOutput {
            content: args.to_string(),
            summary: "echo".into(),
            is_error: false,
        }
    }

    async fn execute_structured(
        &self,
        args: Value,
        cancel: CancellationToken,
    ) -> (ToolOutput, Option<Value>) {
        let output = self.execute(args.clone(), cancel).await;
        (output, Some(args))
    }
}

fn registry(name: &'static str) -> (ToolRegistry, Arc<Mutex<Vec<Value>>>) {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let mut registry = ToolRegistry::new(vec![Box::new(Echo {
        name,
        calls: calls.clone(),
        started: None,
    })]);
    registry.enable_codemode(&[]).unwrap();
    (registry, calls)
}

fn snippet(code: &str) -> Value {
    json!({"code": code, "description": "Check orchestration"})
}

#[tokio::test]
async fn structured_dependent_calls_preserve_names_and_hold_gate_once() {
    let (mut registry, calls) = registry("mcp__server__tool-with-dashes");
    let gate = Arc::new(tokio::sync::Mutex::new(()));
    registry.set_execution_gate(gate.clone());
    let snapshot = registry.snapshot();
    assert_eq!(snapshot.definitions.len(), 1);
    assert_eq!(snapshot.definitions[0].name, "codemode");
    assert_eq!(snapshot.prompt_context.snippets[0].name, "codemode");
    assert!(
        snapshot.definitions[0]
            .description
            .contains("mcp__server__tool-with-dashes")
    );
    let output = tokio::time::timeout(Duration::from_secs(10), registry.execute("codemode", snippet(
        "first = tools['mcp__server__tool-with-dashes'](**{'class': 4})\nsecond = tools['mcp__server__tool-with-dashes'](value=first['class'] + 1)\nsecond['value']"
    ), CancellationToken::new())).await.expect("workflow must not reacquire its gate");
    assert!(!output.is_error, "{}", output.content);
    assert!(output.content.contains("Result: 5"));
    assert_eq!(
        *calls.lock().unwrap(),
        vec![json!({"class": 4}), json!({"value": 5})]
    );
    assert!(gate.try_lock().is_ok());
    assert!(
        registry
            .execute(
                "mcp__server__tool-with-dashes",
                json!({}),
                CancellationToken::new()
            )
            .await
            .is_error
    );
}

#[tokio::test]
async fn python_is_replaced_only_in_the_registry_with_codemode_enabled() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let mut registry = ToolRegistry::new(vec![
        Box::new(Echo {
            name: "echo",
            calls: calls.clone(),
            started: None,
        }),
        Box::new(Echo {
            name: "other",
            calls,
            started: None,
        }),
        Box::new(crate::PythonTool),
        Box::new(crate::BashTool::with_workspace_root(".")),
    ]);
    let python_prompt = crate::PythonTool.spec().prompt;
    let native = registry.snapshot();
    assert!(native.definitions.iter().any(|tool| tool.name == "python"));
    assert!(
        native
            .prompt_context
            .snippets
            .iter()
            .any(|tool| tool.name == "python")
    );
    for guideline in &python_prompt.guidelines {
        assert!(native.prompt_context.guidelines.contains(guideline));
    }

    // Standalone Python is replaced even with only some tools folded in.
    registry.enable_codemode(&["echo".into()]).unwrap();
    let codemode = registry.snapshot();
    let names: Vec<_> = codemode
        .definitions
        .iter()
        .map(|tool| tool.name.as_str())
        .collect();
    assert_eq!(names, ["other", "bash", "codemode"]);
    assert!(
        !codemode
            .prompt_context
            .snippets
            .iter()
            .any(|tool| tool.name == "python")
    );
    for guideline in &python_prompt.guidelines {
        assert!(!codemode.prompt_context.guidelines.contains(guideline));
    }
    assert!(
        registry
            .execute("python", snippet("6 * 7"), CancellationToken::new())
            .await
            .is_error
    );
    // Hidden from the native catalogue must not mean admitted as a callback.
    assert!(
        registry
            .execute_codemode_tool("python", snippet("6 * 7"), CancellationToken::new())
            .await
            .0
            .is_error
    );
    assert!(
        registry
            .execute(
                "codemode",
                snippet("tools['python'](code='6 * 7', description='Calculate')"),
                CancellationToken::new()
            )
            .await
            .is_error
    );
    let output = registry
        .execute("codemode", snippet("6 * 7"), CancellationToken::new())
        .await;
    assert!(!output.is_error, "{}", output.content);
    assert!(output.content.contains("Result: 42"));

    // A separate registry, such as a workspace child, retains its own policy.
    let child = ToolRegistry::new(vec![Box::new(crate::PythonTool)]);
    assert_eq!(child.definitions()[0].name, "python");
    let output = child
        .execute("python", snippet("6 * 7"), CancellationToken::new())
        .await;
    assert!(!output.is_error, "{}", output.content);
    assert!(output.content.contains("Result: 42"));
}

#[tokio::test]
async fn tool_budget_and_host_access_fail_closed_without_replaying_calls() {
    let (registry, calls) = registry("echo");
    let output = registry
        .execute(
            "codemode",
            snippet("for i in range(51):\n    tools['echo'](value=i)"),
            CancellationToken::new(),
        )
        .await;
    assert!(output.is_error);
    assert!(output.content.contains("50 nested tool call(s) started"));
    assert_eq!(calls.lock().unwrap().len(), 50);
    for code in [
        "tools['bash'](command='touch escaped')",
        "tools['codemode'](code='pass')",
        "open('/tmp/not-allowed', 'w')",
        "import os\nos.getenv('HOME')",
        "import time\ntime.time()",
        "import random\nrandom.random()",
        "tools['echo'](object())",
    ] {
        let output = registry
            .execute("codemode", snippet(code), CancellationToken::new())
            .await;
        assert!(output.is_error, "{code}: {}", output.content);
    }
    assert_eq!(calls.lock().unwrap().len(), 50);
    // Each call has fresh globals, not a REPL carried over from the previous snippet.
    registry
        .execute(
            "codemode",
            snippet("previous = 42"),
            CancellationToken::new(),
        )
        .await;
    assert!(
        registry
            .execute("codemode", snippet("previous"), CancellationToken::new())
            .await
            .is_error
    );
}

#[tokio::test]
async fn cancellation_and_future_drop_cannot_admit_later_tools() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let started = Arc::new(Notify::new());
    let mut registry = ToolRegistry::new(vec![Box::new(Echo {
        name: "echo",
        calls: calls.clone(),
        started: Some(started.clone()),
    })]);
    registry.enable_codemode(&[]).unwrap();
    let gate = Arc::new(tokio::sync::Mutex::new(()));
    registry.set_execution_gate(gate.clone());
    let code = snippet("tools['echo'](value=1)\ntools['echo'](value=2)");
    let cancel = CancellationToken::new();
    let run = registry.execute("codemode", code.clone(), cancel.clone());
    tokio::pin!(run);
    tokio::select! {
        _ = started.notified() => {},
        output = &mut run => panic!("unexpected completion: {output:?}"),
        _ = tokio::time::sleep(Duration::from_secs(10)) => panic!("tool did not start"),
    }
    assert!(gate.try_lock().is_err());
    cancel.cancel();
    let output = run.await;
    assert!(output.is_error);
    assert!(output.content.contains("execution status unknown"));
    assert!(gate.try_lock().is_ok());
    {
        let run = registry.execute("codemode", code, CancellationToken::new());
        tokio::pin!(run);
        tokio::select! {
            _ = started.notified() => {},
            _ = &mut run => panic!("unexpected completion"),
        }
        // Drop while the first tool is pending. No detached driver owns the
        // registry or can admit the second callback.
    }
    assert!(gate.try_lock().is_ok());
    assert_eq!(
        *calls.lock().unwrap(),
        vec![json!({"value": 1}), json!({"value": 1})]
    );
}

#[tokio::test]
async fn representations_and_printed_text_cannot_hide_errors_or_exceed_limits() {
    let (registry, _) = registry("echo");
    let output = tokio::time::timeout(
        Duration::from_secs(10),
        registry.execute(
            "codemode",
            snippet("value = [0]\nfor _ in range(60):\n    value = [value, value]\nvalue"),
            CancellationToken::new(),
        ),
    )
    .await
    .expect("formatting must stop at the byte limit");
    assert!(!output.is_error, "{}", output.content);
    assert!(output.content.ends_with("[output truncated]"));
    assert!(output.content.len() <= MAX_VALUE_BYTES + 32);
    let output = registry
        .execute(
            "codemode",
            snippet("tools['echo'](value=1)\nprint('x' * 51200, end='')\n1 / 0"),
            CancellationToken::new(),
        )
        .await;
    assert!(output.is_error);
    assert!(output.content.starts_with("1 nested tool call(s) started"));
    assert!(output.content.contains("ZeroDivisionError"));
    assert!(output.content.contains("1: echo — returned"));
    let output = registry.execute("codemode", snippet(
        "for _ in range(1001):\n    try:\n        missing\n    except NameError:\n        pass"
    ), CancellationToken::new()).await;
    assert!(output.is_error);
    assert!(output.content.contains("suspension limit exceeded"));
}

#[test]
fn oversized_catalogue_stops_before_requesting_more_schemas() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct SchemaTool {
        name: &'static str,
        requests: Arc<AtomicUsize>,
        oversized: bool,
    }

    #[async_trait]
    impl Tool for SchemaTool {
        fn spec(&self) -> ToolSpec {
            ToolSpec {
                definition: ToolDefinition {
                    name: self.name.into(),
                    description: "Return structured data".into(),
                    parameters: json!({"type": "object"}),
                },
                prompt: ToolPrompt::default(),
            }
        }

        fn output_schema(&self) -> Option<Value> {
            self.requests.fetch_add(1, Ordering::SeqCst);
            // Escaping alone exceeds the budget; raw string length does not.
            Some(json!({
                "type": "object",
                "description": if self.oversized { "\"".repeat(MAX_CATALOG_BYTES / 2) } else { String::new() },
            }))
        }

        async fn execute(&self, _: Value, _: CancellationToken) -> ToolOutput {
            unreachable!("catalogue construction must not execute tools")
        }
    }

    let requests = Arc::new(AtomicUsize::new(0));
    let mut registry = ToolRegistry::new(vec![
        Box::new(SchemaTool {
            name: "oversized",
            requests: requests.clone(),
            oversized: true,
        }),
        Box::new(SchemaTool {
            name: "later",
            requests: requests.clone(),
            oversized: false,
        }),
    ]);
    let native = registry.snapshot();
    assert!(registry.enable_codemode(&[]).is_err());
    // Reject while serializing the first tool, not after materializing all
    // output schemas or exposing a partial codemode catalogue.
    assert_eq!(requests.load(Ordering::SeqCst), 1);
    assert!(!registry.codemode_enabled());
    assert_eq!(registry.snapshot(), native);
}

#[test]
fn selections_and_json_conversion_reject_unsupported_values() {
    let (mut registry, _) = registry("echo");
    assert!(registry.enable_codemode(&[]).is_err());
    for name in ["bash", "python", "subagent", "codemode", "missing"] {
        let mut empty = ToolRegistry::empty();
        assert!(empty.enable_codemode(&[name.into()]).is_err());
    }
    let mut budget = MAX_VALUE_BYTES;
    assert!(monty_to_json(MontyObject::float(f64::NAN).as_ref(), 0, &mut budget).is_err());
    assert!(json_to_monty(&json!(u64::MAX), 0).is_err());
    let mut args = monty_types::CallArgs::new();
    args.push_kwarg(
        "value",
        MontyObject::string("x".repeat(MAX_VALUE_BYTES + 1)),
    );
    assert!(arguments(&args).is_err());
}

#[tokio::test]
async fn discovery_globals_are_lazy_host_metadata_and_keep_the_prompt_small() {
    let (registry, calls) = registry("echo");
    let description = registry.snapshot().definitions[0].description.clone();
    // The index names the tool but never embeds its JSON schema.
    assert!(description.contains("echo(value?)"));
    assert!(!description.contains("\"type\": \"integer\""));

    let output = registry
        .execute(
            "codemode",
            snippet(
                "info = describe('echo')\n[list(TOOLS), info['parameters']['properties']['value']['type'], 'json' in MODULES['available']]",
            ),
            CancellationToken::new(),
        )
        .await;
    assert!(!output.is_error, "{}", output.content);
    assert!(output.content.contains("'echo'") && output.content.contains("'integer'"));
    assert!(output.content.contains("True"));
    // Discovery never invokes a tool.
    assert!(calls.lock().unwrap().is_empty());

    let missing = registry
        .execute(
            "codemode",
            snippet("describe('nope')"),
            CancellationToken::new(),
        )
        .await;
    assert!(missing.is_error && missing.content.contains("unknown tool: nope"));
}

struct BigSchema;

#[async_trait]
impl Tool for BigSchema {
    fn spec(&self) -> ToolSpec {
        // Larger than the 50 KiB tool-result cap, with an integer above i64.
        let properties: serde_json::Map<String, Value> = (0..1500)
            .map(|index| {
                (
                    format!("field_{index}"),
                    json!({"type": "integer", "maximum": 18446744073709551615u64,
                           "description": "padding to exceed the value cap"}),
                )
            })
            .collect();
        ToolSpec {
            definition: ToolDefinition {
                name: "big".into(),
                description: "Tool with a large schema".into(),
                parameters: json!({"type": "object", "properties": properties}),
            },
            prompt: ToolPrompt::default(),
        }
    }

    async fn execute(&self, _args: Value, _cancel: CancellationToken) -> ToolOutput {
        unreachable!("describe must not execute tools")
    }
}

#[tokio::test]
async fn describe_returns_schemas_beyond_tool_result_limits() {
    let mut registry = ToolRegistry::new(vec![Box::new(BigSchema)]);
    registry.enable_codemode(&[]).unwrap();
    let output = registry
        .execute(
            "codemode",
            snippet(
                "p = describe('big')['parameters']['properties']\n(len(p), p['field_7']['maximum'])",
            ),
            CancellationToken::new(),
        )
        .await;
    assert!(!output.is_error, "{}", output.content);
    assert!(
        output.content.contains("(1500, 18446744073709551615)"),
        "{}",
        output.content
    );
}
