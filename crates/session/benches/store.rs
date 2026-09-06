use session::{SessionCreateOptions, SessionEvent, SessionStore, StoredMessage, StoredToolCall};
use std::hint::black_box;
use std::time::Instant;
use tempfile::tempdir;

fn main() {
    for turns in [1_000usize, 10_000] {
        let elapsed = benchmark_alternating_appends(turns);
        println!("alternating {turns} turns: {:.3}s", elapsed.as_secs_f64());
    }
}

fn benchmark_alternating_appends(turns: usize) -> std::time::Duration {
    let root = tempdir().expect("session root");
    let workspace = tempdir().expect("workspace root");
    let store = SessionStore::new(root.path(), workspace.path())
        .expect("create session store")
        .with_deferred_sync(true);
    let session = store
        .create(SessionCreateOptions::default())
        .expect("create session");
    let id = session.id();
    let mut first = store.open(&id).expect("open first session view");
    let mut second = store.open(&id).expect("open second session view");
    let tool_output = "x".repeat(4_096);
    let started = Instant::now();

    for turn in 0..turns {
        let target = if turn % 2 == 0 {
            &mut first
        } else {
            &mut second
        };
        store
            .append_event(
                target,
                SessionEvent::UserMessage {
                    message: StoredMessage::from_llm(&llm::Message::user(format!(
                        "inspect item {turn}"
                    ))),
                },
            )
            .expect("append user message");
        let call_id = format!("bench-{turn}");
        store
            .append_event(
                target,
                SessionEvent::ToolCall {
                    call: StoredToolCall {
                        id: call_id.clone(),
                        name: "read".into(),
                        arguments: serde_json::json!({"path": "large.txt"}),
                    },
                },
            )
            .expect("append tool call");
        store
            .append_event(
                target,
                SessionEvent::ToolResult {
                    tool_call_id: call_id,
                    content: tool_output.clone(),
                    is_error: false,
                    tool_name: Some("read".into()),
                },
            )
            .expect("append tool result");
        black_box(&target.events);
    }
    store.sync_session(&first).expect("sync benchmark session");
    started.elapsed()
}
