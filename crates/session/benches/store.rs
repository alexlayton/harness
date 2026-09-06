use session::{SessionCreateOptions, SessionEvent, SessionStore, StoredMessage, StoredToolCall};
use std::hint::black_box;
use std::time::Instant;
use tempfile::tempdir;

fn main() {
    // Near-linear scaling guard (PERF-5): 10× the turns must cost well
    // under 10× the time. Incremental append validation reconciles only
    // the new suffix, so growth is ~linear; a full-replay-per-append
    // regression would blow past this bar immediately.
    //
    // Recent runs (Linux, debug):
    //   alternating 1000 turns: ~0.11s, 10000 turns: ~1.1s (~10×)
    //   listing 1000 turns: ~0.003s, 10000 turns: ~0.035s (~11×)
    // Debug timing is noisy (machine + parallelism dependent), so the bar
    // is a generous 40× rather than a tight bound.
    const LINEARITY_BAR: f64 = 40.0;
    let mut append_times = Vec::new();
    let mut listing_times = Vec::new();
    for turns in [1_000usize, 10_000] {
        let elapsed = benchmark_alternating_appends(turns);
        println!("alternating {turns} turns: {:.3}s", elapsed.as_secs_f64());
        append_times.push(elapsed.as_secs_f64());
        let listing = benchmark_listing(turns);
        println!("listing {turns} turns: {:.3}s", listing.as_secs_f64());
        listing_times.push(listing.as_secs_f64());
    }
    let append_ratio = append_times[1] / append_times[0].max(f64::EPSILON);
    let listing_ratio = listing_times[1] / listing_times[0].max(f64::EPSILON);
    assert!(
        append_ratio < LINEARITY_BAR,
        "appends went super-linear: 10k/1k = {append_ratio:.1}× (bar {LINEARITY_BAR}×)"
    );
    assert!(
        listing_ratio < LINEARITY_BAR,
        "listing went super-linear: 10k/1k = {listing_ratio:.1}× (bar {LINEARITY_BAR}×)"
    );
    println!("near-linear scaling holds: append {append_ratio:.1}×, listing {listing_ratio:.1}×");
}

fn benchmark_listing(turns: usize) -> std::time::Duration {
    let root = tempdir().expect("session root");
    let workspace = tempdir().expect("workspace root");
    let store = SessionStore::new(root.path(), workspace.path())
        .expect("create session store")
        .with_deferred_sync(true);
    let mut session = store
        .create(SessionCreateOptions::default())
        .expect("create session");
    let tool_output = "x".repeat(4_096);
    for turn in 0..turns {
        let call_id = format!("listing-{turn}");
        store
            .append_event(
                &mut session,
                SessionEvent::UserMessage {
                    message: StoredMessage::from_llm(&llm::Message::user("inspect")),
                },
            )
            .expect("append user message");
        store
            .append_event(
                &mut session,
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
                &mut session,
                SessionEvent::ToolResult {
                    tool_call_id: call_id,
                    content: tool_output.clone(),
                    is_error: false,
                    tool_name: Some("read".into()),
                },
            )
            .expect("append tool result");
    }
    store.sync_session(&session).expect("sync listing session");
    let started = Instant::now();
    black_box(store.list().expect("list sessions"));
    started.elapsed()
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
