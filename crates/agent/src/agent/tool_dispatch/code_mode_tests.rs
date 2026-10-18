use super::*;
use async_trait::async_trait;
use serde_json::{Value, json};
use session::{SessionCreateOptions, SessionStore, StoredToolCall};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;
use tempfile::tempdir;
use tools::{Tool, ToolSpec};

struct CountingTool(Arc<AtomicUsize>);

#[async_trait]
impl Tool for CountingTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            definition: llm::ToolDefinition {
                name: "counting".into(),
                description: "Record a callback side effect and return its value".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {"value": {"type": "integer"}},
                    "required": ["value"]
                }),
            },
            prompt: tools::ToolPrompt::default(),
        }
    }

    async fn execute(&self, args: Value, _cancel: CancellationToken) -> ToolOutput {
        self.0.fetch_add(1, Ordering::SeqCst);
        ToolOutput {
            content: args["value"].to_string(),
            is_error: false,
            summary: "counted callback".into(),
        }
    }
}

fn registry() -> (ToolRegistry, Arc<AtomicUsize>) {
    let executions = Arc::new(AtomicUsize::new(0));
    let mut registry = ToolRegistry::new(vec![Box::new(CountingTool(executions.clone()))]);
    registry.enable_code_mode(&[]).unwrap();
    (registry, executions)
}

fn batch() -> ToolBatch {
    ToolBatch {
        calls: vec![ToolCall {
            id: "parent-code-call".into(),
            name: "run_code".into(),
            arguments: json!({
                "code": "tools['counting'](value=7)",
                "description": "Run one callback"
            }),
        }],
        class: Concurrency::Exclusive,
    }
}

enum StartAction {
    Reject,
    Cancel(CancellationToken),
}

struct StartHooks {
    action: StartAction,
    starts: usize,
}

impl ToolDispatchHooks for StartHooks {
    fn started(&mut self, _call: &ToolCall, _started: Instant, _synthetic: bool) {}

    fn finished(
        &mut self,
        _call: &ToolCall,
        _output: &ToolOutput,
        _started: Instant,
        _state: CallState,
    ) {
    }

    fn nested(&mut self, parent: &str, event: CodeModeEvent) -> Result<(), String> {
        assert_eq!(parent, "parent-code-call");
        if let CodeModeEvent::Started {
            name, arguments, ..
        } = event
        {
            assert_eq!(name, "counting");
            assert_eq!(arguments, json!({"value": 7}));
            self.starts += 1;
            match &self.action {
                StartAction::Reject => return Err("nested start could not be persisted".into()),
                // The dispatcher sends the successful ack after this returns.
                // Control is then ready before the workflow consumes that ack.
                StartAction::Cancel(token) => token.cancel(),
            }
        }
        Ok(())
    }
}

#[tokio::test]
async fn failed_nested_start_ack_prevents_callback_side_effects() {
    let (registry, executions) = registry();
    let batch = batch();
    let cancel = CancellationToken::new();
    let mut control = CancellationControl::new(&cancel, DispatchCancellation::Explicit);
    let mut hooks = StartHooks {
        action: StartAction::Reject,
        starts: 0,
    };

    let outcome = tokio::time::timeout(
        Duration::from_secs(5),
        execute_tool_batch_with_secrets(
            &registry,
            &batch,
            1,
            &mut control,
            &mut hooks,
            None,
            false,
        ),
    )
    .await
    .expect("failed acknowledgement must stop the workflow");

    assert_eq!(hooks.starts, 1);
    assert_eq!(executions.load(Ordering::SeqCst), 0);
    assert_eq!(
        outcome.nested_error.as_deref(),
        Some("nested start could not be persisted")
    );
    assert_eq!(outcome.cancellation, Some(DispatchCancellation::Explicit));
    assert_eq!(outcome.outcomes.len(), 1);
    assert!(outcome.outcomes[0].output.is_error);
}

#[tokio::test]
async fn cancellation_after_nested_start_ack_cannot_launch_during_ready_drain() {
    let (registry, executions) = registry();
    let batch = batch();
    let cancel = CancellationToken::new();
    let mut control = CancellationControl::new(&cancel, DispatchCancellation::Explicit);
    let mut hooks = StartHooks {
        action: StartAction::Cancel(cancel.clone()),
        starts: 0,
    };

    let outcome = tokio::time::timeout(
        Duration::from_secs(5),
        execute_tool_batch_with_secrets(
            &registry,
            &batch,
            1,
            &mut control,
            &mut hooks,
            None,
            false,
        ),
    )
    .await
    .expect("cancellation after the start ack must not hang");

    assert_eq!(hooks.starts, 1);
    assert!(cancel.is_cancelled());
    assert_eq!(outcome.cancellation, Some(DispatchCancellation::Explicit));
    assert_eq!(outcome.nested_error, None);
    // The ready-result drain polls the workflow again. That poll must not
    // turn its now-ready start acknowledgement into a tool execution.
    assert_eq!(executions.load(Ordering::SeqCst), 0);
    assert_eq!(outcome.outcomes.len(), 1);
    assert!(outcome.outcomes[0].output.is_error);
}

#[tokio::test]
async fn session_hooks_persist_nested_audit_records_before_outer_result_only() {
    let root = tempdir().unwrap();
    let workspace = tempdir().unwrap();
    let store = SessionStore::new(root.path(), workspace.path()).unwrap();
    let session = store.create(SessionCreateOptions::default()).unwrap();
    let mut state = AgentSessionState {
        store,
        session,
        active_lease: None,
    };
    let (registry, executions) = registry();
    let batch = batch();
    let parent = &batch.calls[0];
    state
        .store
        .append_event(
            &mut state.session,
            SessionEvent::ToolCall {
                call: StoredToolCall::from(parent),
            },
        )
        .unwrap();
    let (events, _receiver) = mpsc::unbounded_channel();
    let secrets = SecretMasker::default();
    let cancel = CancellationToken::new();
    let mut control = CancellationControl::new(&cancel, DispatchCancellation::Explicit);
    let mut hooks = AgentToolDispatchHooks::new(&events, Some(&mut state), &secrets);

    let outcome = tokio::time::timeout(
        Duration::from_secs(5),
        execute_tool_batch_with_secrets(
            &registry,
            &batch,
            1,
            &mut control,
            &mut hooks,
            Some(&secrets),
            true,
        ),
    )
    .await
    .expect("session hooks must acknowledge nested events");
    drop(hooks);

    assert_eq!(outcome.nested_error, None);
    assert_eq!(outcome.cancellation, None);
    assert_eq!(executions.load(Ordering::SeqCst), 1);
    assert_eq!(outcome.outcomes.len(), 1);
    let output = &outcome.outcomes[0].output;
    assert!(!output.is_error, "{}", output.content);
    assert_eq!(state.session.events.len(), 3);

    // The owning dispatch loop appends the outer result after the batch and
    // its session hooks return. Do the same without constructing a provider.
    state
        .store
        .append_event(
            &mut state.session,
            SessionEvent::ToolResult {
                tool_call_id: parent.id.clone(),
                content: output.content.clone(),
                is_error: output.is_error,
                tool_name: Some(parent.name.clone()),
            },
        )
        .unwrap();

    let loaded = state.store.open(&state.session.id()).unwrap();
    assert_eq!(loaded.events.len(), 4);
    assert!(matches!(
        loaded.events[0].event,
        SessionEvent::ToolCall { .. }
    ));
    let SessionEvent::Unknown {
        kind: start_kind,
        data: start,
    } = &loaded.events[1].event
    else {
        panic!("nested start must remain an Unknown audit record");
    };
    let SessionEvent::Unknown {
        kind: finish_kind,
        data: finish,
    } = &loaded.events[2].event
    else {
        panic!("nested result must remain an Unknown audit record");
    };
    assert_eq!(start_kind, "code_mode_call");
    assert_eq!(finish_kind, "code_mode_result");
    assert_eq!(start["parent_call_id"], parent.id);
    assert_eq!(finish["parent_call_id"], parent.id);
    assert_eq!(start["call_id"], finish["call_id"]);
    assert!(
        start["call_id"]
            .as_str()
            .unwrap()
            .starts_with(&format!("{}:", parent.id))
    );
    assert_eq!(start["name"], "counting");
    assert_eq!(finish["name"], "counting");
    assert_eq!(start["arguments"], json!({"value": 7}));
    assert_eq!(finish["content"], "7");
    assert_eq!(finish["is_error"], false);
    assert!(matches!(
        &loaded.events[3].event,
        SessionEvent::ToolResult { tool_call_id, .. } if tool_call_id == &parent.id
    ));

    // Audit records stay in the durable log, but never add nested tool
    // calls or results to provider context.
    assert_eq!(
        loaded.context_messages(),
        vec![
            Message::assistant(vec![Content::ToolCall(parent.clone())]),
            Message {
                role: Role::Tool,
                content: vec![Content::ToolResult {
                    tool_call_id: parent.id.clone(),
                    content: output.content.clone(),
                    is_error: false,
                }],
            },
        ]
    );
}
