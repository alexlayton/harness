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
    registry.enable_code_mode(&[]).unwrap();
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
    assert_eq!(snapshot.definitions[0].name, "run_code");
    assert_eq!(snapshot.prompt_context.snippets[0].name, "run_code");
    assert!(
        snapshot.definitions[0]
            .description
            .contains("mcp__server__tool-with-dashes")
    );
    let output = tokio::time::timeout(Duration::from_secs(10), registry.execute("run_code", snippet(
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
async fn tool_budget_and_host_access_fail_closed_without_replaying_calls() {
    let (registry, calls) = registry("echo");
    let output = registry
        .execute(
            "run_code",
            snippet("for i in range(51):\n    tools['echo'](value=i)"),
            CancellationToken::new(),
        )
        .await;
    assert!(output.is_error);
    assert!(output.content.contains("50 nested tool call(s) started"));
    assert_eq!(calls.lock().unwrap().len(), 50);
    for code in [
        "tools['bash'](command='touch escaped')",
        "tools['run_code'](code='pass')",
        "open('/tmp/not-allowed', 'w')",
        "import os\nos.getenv('HOME')",
        "import time\ntime.time()",
        "import random\nrandom.random()",
        "tools['echo'](object())",
    ] {
        let output = registry
            .execute("run_code", snippet(code), CancellationToken::new())
            .await;
        assert!(output.is_error, "{code}: {}", output.content);
    }
    assert_eq!(calls.lock().unwrap().len(), 50);
    // Each call has fresh globals, not a REPL carried over from the previous snippet.
    registry
        .execute(
            "run_code",
            snippet("previous = 42"),
            CancellationToken::new(),
        )
        .await;
    assert!(
        registry
            .execute("run_code", snippet("previous"), CancellationToken::new())
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
    registry.enable_code_mode(&[]).unwrap();
    let gate = Arc::new(tokio::sync::Mutex::new(()));
    registry.set_execution_gate(gate.clone());
    let code = snippet("tools['echo'](value=1)\ntools['echo'](value=2)");
    let cancel = CancellationToken::new();
    let run = registry.execute("run_code", code.clone(), cancel.clone());
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
        let run = registry.execute("run_code", code, CancellationToken::new());
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
            "run_code",
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
            "run_code",
            snippet("tools['echo'](value=1)\nprint('x' * 51200, end='')\n1 / 0"),
            CancellationToken::new(),
        )
        .await;
    assert!(output.is_error);
    assert!(output.content.starts_with("1 nested tool call(s) started"));
    assert!(output.content.contains("ZeroDivisionError"));
    assert!(output.content.contains("1: echo — returned"));
    let output = registry.execute("run_code", snippet(
        "for _ in range(1001):\n    try:\n        missing\n    except NameError:\n        pass"
    ), CancellationToken::new()).await;
    assert!(output.is_error);
    assert!(output.content.contains("suspension limit exceeded"));
}

#[test]
fn selections_and_json_conversion_reject_unsupported_values() {
    let (mut registry, _) = registry("echo");
    assert!(registry.enable_code_mode(&[]).is_err());
    for name in ["bash", "python", "subagent", "run_code", "missing"] {
        let mut empty = ToolRegistry::empty();
        assert!(empty.enable_code_mode(&[name.into()]).is_err());
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
