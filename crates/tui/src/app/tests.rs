use super::*;
use crossterm::event::KeyEvent;
use std::panic::AssertUnwindSafe;
use std::sync::OnceLock;

/// ANSI for one row as the frame writer produces it: sanitized, then styled.
fn ansi_of(line: &Row) -> String {
    render::fit_line_to_width(line, usize::MAX).to_ansi()
}

fn ui(width: u16, height: u16) -> CrossTerm {
    CrossTerm::base(
        "test-model",
        "test-provider",
        vec!["opencode-go".into(), "openrouter".into()],
        StartupEntries::default(),
        PathBuf::from("."),
        width,
        height,
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum TerminalOperation {
    EnableRaw,
    DisableRaw,
    EnableBracketed,
    DisableBracketed,
    PushKeyboard,
    PopKeyboard,
    Newline,
}

#[derive(Clone, Default)]
struct MockTerminal {
    calls: Arc<Mutex<Vec<TerminalOperation>>>,
    failures: Arc<Mutex<HashMap<TerminalOperation, usize>>>,
}

impl MockTerminal {
    fn fail_next(&self, operation: TerminalOperation) {
        self.failures.lock().unwrap().insert(operation, 1);
    }

    fn calls(&self) -> Vec<TerminalOperation> {
        self.calls.lock().unwrap().clone()
    }

    fn run(&self, operation: TerminalOperation) -> io::Result<()> {
        self.calls.lock().unwrap().push(operation);
        let mut failures = self.failures.lock().unwrap();
        let Some(remaining) = failures.get_mut(&operation) else {
            return Ok(());
        };
        if *remaining == 0 {
            return Ok(());
        }
        *remaining -= 1;
        Err(io::Error::other(format!("mock {operation:?}")))
    }
}

impl TerminalBackend for MockTerminal {
    fn size(&self) -> io::Result<(u16, u16)> {
        Ok((80, 24))
    }

    fn enable_raw_mode(&mut self) -> io::Result<()> {
        self.run(TerminalOperation::EnableRaw)
    }

    fn disable_raw_mode(&mut self) -> io::Result<()> {
        self.run(TerminalOperation::DisableRaw)
    }

    fn enable_bracketed_paste(&mut self, _out: &mut Stdout) -> io::Result<()> {
        self.run(TerminalOperation::EnableBracketed)
    }

    fn disable_bracketed_paste(&mut self, _out: &mut Stdout) -> io::Result<()> {
        self.run(TerminalOperation::DisableBracketed)
    }

    fn push_keyboard_flags(&mut self, _out: &mut Stdout) -> io::Result<()> {
        self.run(TerminalOperation::PushKeyboard)
    }

    fn pop_keyboard_flags(&mut self, _out: &mut Stdout) -> io::Result<()> {
        self.run(TerminalOperation::PopKeyboard)
    }

    fn write_newline(&mut self, _out: &mut Stdout) -> io::Result<()> {
        self.run(TerminalOperation::Newline)
    }
}

fn terminal_ui(backend: MockTerminal) -> CrossTerm {
    CrossTerm::with_backend(ui(80, 24), Box::new(backend))
}

fn successful_terminal_ui(backend: &MockTerminal) -> CrossTerm {
    let mut ui = terminal_ui(backend.clone());
    ui.setup_terminal().unwrap();
    ui
}

fn row_text(line: &Row) -> String {
    line.runs
        .iter()
        .map(|span| span.text.as_str())
        .collect::<String>()
}

fn row_width(line: &Row) -> usize {
    line.runs
        .iter()
        .map(|span| UnicodeWidthStr::width(span.text.as_str()))
        .sum()
}

#[test]
fn transcript_and_snapshot_keep_newest_entries_within_budget() {
    let big = "z".repeat(TRANSCRIPT_BUDGET_BYTES / 3);
    let mut transcript = vec![Entry::Banner {
        title_order: render::WelcomeTitleOrder::random(),
    }];
    let mut pending = (0..6)
        .map(|index| Entry::User {
            text: format!("{index}{big}"),
        })
        .collect::<Vec<_>>();
    commit_to_transcript(&mut transcript, &mut pending);
    assert!(transcript.iter().map(entry_bytes).sum::<usize>() <= TRANSCRIPT_BUDGET_BYTES);
    assert!(matches!(transcript[0], Entry::Banner { .. }));
    assert!(matches!(transcript.last(), Some(Entry::User { text }) if text.starts_with('5')));

    let entries = (0..6)
        .map(|index| SessionSnapshotEntry::User {
            text: format!("{index}{big}"),
        })
        .collect::<Vec<_>>();
    let window = newest_snapshot_entries(entries, TRANSCRIPT_BUDGET_BYTES);
    assert!(window.dropped > 0);
    assert_eq!(window.dropped + window.kept.len(), 6);
    assert!(matches!(
        window.kept.last(),
        Some(SessionSnapshotEntry::User { text }) if text.starts_with('5')
    ));
    // A single entry larger than the budget is still shown.
    let window = newest_snapshot_entries(
        vec![SessionSnapshotEntry::User {
            text: "w".repeat(TRANSCRIPT_BUDGET_BYTES + 1),
        }],
        TRANSCRIPT_BUDGET_BYTES,
    );
    assert_eq!((window.kept.len(), window.dropped), (1, 0));
}

#[test]
fn editing_moves_and_deletes_whole_grapheme_clusters() {
    // Family emoji (ZWJ sequence), a flag, and e + combining acute.
    for cluster in ["👨‍👩‍👧", "🇳🇿", "e\u{301}", "❤️"] {
        let mut input = format!("a{cluster}b");
        let mut cursor = 1 + cluster.len();
        move_left(&input, &mut cursor);
        assert_eq!(cursor, 1, "{cluster:?}");
        move_right(&input, &mut cursor);
        assert_eq!(cursor, 1 + cluster.len(), "{cluster:?}");
        delete_backward(&mut input, &mut cursor);
        assert_eq!(input, "ab", "{cluster:?}");
        let mut input = format!("a{cluster}b");
        let mut cursor = 1;
        delete_forward(&mut input, &mut cursor);
        assert_eq!(input, "ab", "{cluster:?}");
    }
}

#[test]
fn wrapped_rows_never_split_a_grapheme_cluster() {
    let text = "ab👨‍👩‍👧cd🇳🇿e\u{301}";
    for width in 1..8 {
        let rows = render::wrap_text(&[Row::plain(text)], width, ContentStyle::default());
        let joined = rows
            .iter()
            .flat_map(|row| row.runs.iter().map(|span| span.text.to_string()))
            .collect::<String>();
        // Clusters wider than the row are dropped whole, never split:
        // every emitted cluster is one of the original clusters.
        let originals = text
            .graphemes(true)
            .collect::<std::collections::HashSet<_>>();
        for row in &rows {
            let row_text = row
                .runs
                .iter()
                .map(|span| span.text.as_str())
                .collect::<String>();
            for cluster in row_text.graphemes(true) {
                assert!(originals.contains(cluster), "width {width}: {cluster:?}");
            }
        }
        assert!(!joined.is_empty());
    }
}

#[test]
fn pasted_cr_line_breaks_are_kept() {
    assert_eq!(normalize_paste_newlines("a\rb\r\nc\n"), "a\nb\nc\n");
    assert_eq!(
        render::sanitize_terminal_text(&normalize_paste_newlines("one\rtwo")),
        "one\ntwo"
    );
}

#[test]
fn event_drain_is_bounded_while_the_queue_stays_full() {
    // A producer that keeps the channel non-empty must not hold the loop
    // away from terminal input for longer than one bounded drain.
    let mut first = ui(80, 24);
    let (tx, mut rx) = mpsc::channel(EVENT_DRAIN_MAX_EVENTS * 4);
    for _ in 0..(EVENT_DRAIN_MAX_EVENTS * 4) {
        tx.try_send(UiEvent::TextDelta("x".into())).unwrap();
    }
    let applied = first.drain_queued_events(UiEvent::TextDelta("x".into()), &mut rx);
    assert!(applied <= EVENT_DRAIN_MAX_EVENTS);
    assert!(rx.try_recv().is_ok(), "the drain must leave work queued");
    // Merged deltas keep every byte, in order.
    assert_eq!(first.stream.as_ref().unwrap().markdown, "x".repeat(applied));
    let (tx, mut rx) = mpsc::channel(8);
    let mut ui = ui(80, 24);
    for event in [
        UiEvent::TextDelta("b".into()),
        UiEvent::ReasoningDelta("r".into()),
        UiEvent::TextDelta("c".into()),
    ] {
        tx.try_send(event).unwrap();
    }
    ui.drain_queued_events(UiEvent::TextDelta("a".into()), &mut rx);
    let stream = ui.stream.as_ref().unwrap();
    assert_eq!(
        (stream.markdown.as_str(), stream.reasoning.as_str()),
        ("abc", "r")
    );
}

fn begin_ask(ui: &mut CrossTerm) -> mpsc::UnboundedReceiver<InputMessage> {
    let (tx, mut rx) = mpsc::unbounded_channel();
    ui.submit_command("/ASK  keep  this question", &tx).unwrap();
    assert!(ui.busy);
    assert!(matches!(
        rx.try_recv(),
        Ok(InputMessage::Ask { question }) if question == "keep  this question"
    ));
    rx
}

fn assert_ask_not_retained(ui: &CrossTerm) {
    let retained = format!("{:?}{:?}", ui.transcript, ui.pending);
    assert!(!retained.contains("keep  this question"));
    assert!(!retained.contains("one-shot result"));
    assert!(ui.transcript.iter().all(|entry| !matches!(
        entry,
        Entry::User { text } if text.contains("keep  this question")
    )));
}

#[test]
fn accepting_a_candidate_cancels_the_inflight_path_scan() {
    let mut ui = ui(80, 24);
    let (tx, _rx) = mpsc::unbounded_channel();
    ui.input = "/ne".into();
    ui.cursor = ui.input.len();
    let token = CancellationToken::new();
    ui.path_completion_cancel = Some(token.clone());
    let generation = ui.path_completion_generation;
    let context = CompletionContext {
        target: CompletionTarget::Command,
        token_start: 0,
        token_end: 3,
        query: "ne".into(),
    };
    let candidate = Candidate {
        value: "new".into(),
        description: String::new(),
        kind: CandidateKind::Slash,
    };
    ui.accept_candidate(&candidate, &context, &tx).unwrap();
    assert!(token.is_cancelled());
    assert_ne!(ui.path_completion_generation, generation);
    assert!(ui.completion.is_none());
}

#[test]
fn editing_a_recalled_history_entry_keeps_the_edit() {
    let mut ui = ui(80, 24);
    ui.history = vec!["one".into(), "two".into()];
    ui.history_up();
    assert_eq!(ui.input, "two");
    ui.input.push_str("-edited");
    ui.cursor = ui.input.len();
    ui.history_down();
    assert_eq!(ui.input, "two-edited");
    ui.history_up();
    assert_eq!(ui.input, "two");
    ui.history_down();
    assert_eq!(ui.input, "two-edited", "edit is the restored draft");
}

#[test]
fn late_finish_events_do_not_rearm_busy_and_operation_finish_spares_turn() {
    let mut ui = ui(80, 24);
    ui.apply_event(UiEvent::TurnFinished);
    ui.apply_event(UiEvent::ToolCallFinished {
        call_id: "x".into(),
        name: "bash".into(),
        summary: String::new(),
        ok: true,
        duration_ms: 1,
        output: String::new(),
        error: None,
    });
    ui.apply_event(UiEvent::Retrying {
        attempt: 1,
        message: String::new(),
    });
    assert!(!ui.busy, "late events must not spin an idle UI");

    ui.apply_event(UiEvent::TextDelta("hi".into()));
    ui.apply_event(UiEvent::OperationFinished);
    assert!(ui.busy, "operation end must not clear a running turn");
    ui.apply_event(UiEvent::TurnFinished);
    assert!(!ui.busy);

    ui.apply_event(UiEvent::TextDelta("hi".into()));
    ui.apply_event(UiEvent::OperationStarted { label: "L".into() });
    assert_eq!(ui.activity, Activity::Preparing);
}

#[test]
fn retry_during_ask_does_not_make_it_a_turn() {
    let mut ui = ui(80, 24);
    let _rx = begin_ask(&mut ui);
    ui.apply_event(UiEvent::Retrying {
        attempt: 1,
        message: String::new(),
    });
    assert!(ui.busy);
    ui.apply_event(UiEvent::AskAnswer("answer".into()));
    ui.apply_event(UiEvent::OperationFinished);
    assert!(!ui.busy, "operation end must clear a retried ask");
}

#[test]
fn ask_success_completes_busy_and_keeps_answer_out_of_history() {
    let mut ui = ui(80, 24);
    let _rx = begin_ask(&mut ui);
    ui.apply_event(UiEvent::AskAnswer("one-shot result".into()));
    assert!(ui.busy, "answer delivery precedes operation completion");
    ui.apply_event(UiEvent::OperationFinished);
    assert!(!ui.busy);
    assert_ask_not_retained(&ui);
    assert_eq!(ui.ask_answer.as_deref(), Some("one-shot result"));
}

#[test]
fn queued_ask_start_restores_busy_so_esc_interrupts_the_request() {
    let mut ui = ui(80, 24);
    let (tx, mut rx) = mpsc::unbounded_channel();
    let cancel = CancellationToken::new();
    ui.apply_event(UiEvent::AskAnswer("previous answer".into()));
    ui.apply_event(UiEvent::OperationFinished);
    ui.apply_event(UiEvent::AskStarted);
    assert!(ui.busy);
    assert!(ui.ask_answer.is_none());
    ui.handle_input(
        &Event::Key(KeyEvent::new(KeyCode::Esc, KeyModifiers::empty())),
        &tx,
        &cancel,
    )
    .unwrap();
    assert_eq!(rx.try_recv().unwrap(), InputMessage::Interrupt);
    assert!(!cancel.is_cancelled());
}

#[test]
fn multiline_ask_submission_never_becomes_a_conversation_message() {
    let mut ui = ui(80, 24);
    let (tx, mut rx) = mpsc::unbounded_channel();
    ui.input = "/ASK Explain this passage:\n  The moon rose over the sea.".into();
    ui.cursor = ui.input.len();
    ui.submit(&tx).unwrap();
    assert_eq!(
        rx.try_recv().unwrap(),
        InputMessage::Ask {
            question: "Explain this passage:\n  The moon rose over the sea.".into(),
        }
    );
    assert!(ui.busy);
    assert!(ui.pending.is_empty());
    assert!(ui.transcript.is_empty());
}

#[test]
fn ask_answer_renders_inline_at_full_height_and_esc_dismisses_it() {
    let (tx, mut rx) = mpsc::unbounded_channel();
    let cancel = CancellationToken::new();
    let mut ui = ui(60, 24);
    let answer = (0..6)
        .map(|row| format!("\x1b[31mrow {row} of the answer\x1b[0m"))
        .collect::<Vec<_>>()
        .join("\n");
    ui.apply_event(UiEvent::AskAnswer(answer));
    let input = ui.input_layout();
    ui.commit_oversized_ask(input.rows.len(), 0);
    let build = ui.build_region(&input);
    let text = build.rows.iter().map(row_text).collect::<Vec<_>>();
    for row in 0..6 {
        assert!(
            text.iter()
                .any(|line| line.contains(&format!("row {row} of")))
        );
    }
    assert!(
        !text
            .iter()
            .any(|line| line.contains("PgUp") || line.contains("/ask ·")),
        "no pager chrome: {text:?}"
    );
    assert!(text.iter().all(|line| !line.contains('\x1b')));

    // Page keys do nothing; Esc closes the answer without touching history.
    for code in [KeyCode::PageDown, KeyCode::PageUp] {
        ui.handle_input(
            &Event::Key(KeyEvent::new(code, KeyModifiers::empty())),
            &tx,
            &cancel,
        )
        .unwrap();
        assert!(ui.ask_answer.is_some());
    }
    ui.handle_input(
        &Event::Key(KeyEvent::new(KeyCode::Esc, KeyModifiers::empty())),
        &tx,
        &cancel,
    )
    .unwrap();
    assert!(ui.ask_answer.is_none());
    assert!(ui.pending.is_empty());
    assert!(ui.transcript.is_empty());
    assert!(rx.try_recv().is_err());
    assert!(!cancel.is_cancelled());
}

#[test]
fn full_repaint_defers_a_newly_oversized_answer_instead_of_dropping_it() {
    let mut ui = ui(80, 24);
    let answer = (0..14)
        .map(|row| format!("line {row}"))
        .collect::<Vec<_>>()
        .join("\n");
    ui.apply_event(UiEvent::AskAnswer(answer));
    // The terminal shrinks so the visible answer no longer fits.
    ui.height = 8;
    let (build, oversized) = ui.stage_full_repaint();
    assert!(build.rows.len() <= 8);
    let text = oversized.expect("answer is handed back for incremental printing");
    assert!(text.contains("line 0") && text.contains("line 13"));
    // It must not have been moved into the transcript unprinted.
    assert!(!format!("{:?}", ui.transcript).contains("line 0"));
    assert!(ui.ask_answer.is_none());
}

#[test]
fn ask_answer_taller_than_the_screen_flows_into_scrollback() {
    let mut ui = ui(40, 8);
    let answer = (0..30)
        .map(|row| format!("line {row}"))
        .collect::<Vec<_>>()
        .join("\n");
    ui.apply_event(UiEvent::AskAnswer(answer));
    let input = ui.input_layout();
    ui.commit_oversized_ask(input.rows.len(), 0);
    assert!(ui.ask_answer.is_none());
    let printed = format!("{:?}", ui.pending);
    assert!(printed.contains("line 0") && printed.contains("line 29"));
    let build = ui.build_region(&ui.input_layout());
    assert!(build.rows.len() <= 8);
}

#[test]
fn ask_answer_is_cleared_when_session_changes_or_snapshot_loads() {
    let mut ui = ui(80, 24);
    ui.ask_answer = Some("answer from prior conversation".into());
    ui.apply_event(UiEvent::SessionChanged {
        id: "session-123".into(),
        title: None,
        loaded: true,
    });
    assert_eq!(ui.ask_answer, None);

    ui.ask_answer = Some("answer before snapshot".into());
    ui.apply_event(UiEvent::SessionSnapshot {
        entries: Vec::new(),
    });
    assert_eq!(ui.ask_answer, None);
}

#[test]
fn ask_provider_open_error_completes_busy_without_retaining_question() {
    let mut ui = ui(80, 24);
    ui.ask_answer = Some("previous answer".into());
    let _rx = begin_ask(&mut ui);
    assert!(ui.ask_answer.is_none());
    ui.apply_event(UiEvent::Error("provider open failed".into()));
    ui.apply_event(UiEvent::OperationFinished);
    assert!(!ui.busy);
    assert!(ui.ask_answer.is_none());
    assert_ask_not_retained(&ui);
}

#[test]
fn ask_mid_stream_error_completes_busy_without_retaining_question_or_partial_result() {
    let mut ui = ui(80, 24);
    let _rx = begin_ask(&mut ui);
    ui.apply_event(UiEvent::Error("provider stream failed".into()));
    ui.apply_event(UiEvent::OperationFinished);
    assert!(!ui.busy);
    assert_ask_not_retained(&ui);
}

fn record(status: ToolStatus) -> ToolRecord {
    ToolRecord {
        name: "bash".into(),
        summary: "bash: cargo test".into(),
        ok: !matches!(status, ToolStatus::Failure),
        duration_ms: 1_200,
        output: String::new(),
        omitted_lines: 0,
        error: matches!(status, ToolStatus::Failure).then(|| "boom\ntrace".into()),
        status,
    }
}

fn start(id: &str, summary: &str) -> UiEvent {
    UiEvent::ToolCallStarted {
        call_id: id.into(),
        name: "subagent".into(),
        summary: summary.into(),
    }
}

fn finish(id: &str, summary: &str) -> UiEvent {
    UiEvent::ToolCallFinished {
        call_id: id.into(),
        name: "subagent".into(),
        summary: summary.into(),
        ok: true,
        duration_ms: 10,
        output: "report".into(),
        error: None,
    }
}

fn active_summaries(ui: &CrossTerm) -> Vec<String> {
    ui.running_tools
        .iter()
        .map(|running| running.record.summary.clone())
        .collect()
}

#[test]
fn terminal_setup_restores_each_partial_prefix_without_popping_unset_modes() {
    let raw_failure = MockTerminal::default();
    raw_failure.fail_next(TerminalOperation::EnableRaw);
    let mut ui = terminal_ui(raw_failure.clone());
    let error = ui.setup_terminal().unwrap_err();
    assert!(format!("{error:#}").contains("mock EnableRaw"));
    assert_eq!(
        raw_failure.calls(),
        vec![TerminalOperation::EnableRaw, TerminalOperation::Newline]
    );

    let bracketed_failure = MockTerminal::default();
    bracketed_failure.fail_next(TerminalOperation::EnableBracketed);
    let mut ui = terminal_ui(bracketed_failure.clone());
    let error = ui.setup_terminal().unwrap_err();
    assert!(format!("{error:#}").contains("mock EnableBracketed"));
    assert_eq!(
        bracketed_failure.calls(),
        vec![
            TerminalOperation::EnableRaw,
            TerminalOperation::EnableBracketed,
            TerminalOperation::DisableRaw,
            TerminalOperation::Newline,
        ]
    );

    // Kitty keyboard enhancement is optional. A failed push is not marked
    // as enabled, so cleanup never pops a stack belonging to another UI.
    let keyboard_failure = MockTerminal::default();
    keyboard_failure.fail_next(TerminalOperation::PushKeyboard);
    let mut ui = terminal_ui(keyboard_failure.clone());
    ui.setup_terminal().unwrap();
    {
        let terminal = ui.terminal.lock().unwrap();
        assert!(terminal.modes.raw_mode);
        assert!(terminal.modes.bracketed_paste);
        assert!(!terminal.modes.keyboard_flags);
        assert!(!terminal.modes.newline_written);
    }
    ui.restore().unwrap();
    assert_eq!(
        keyboard_failure.calls(),
        vec![
            TerminalOperation::EnableRaw,
            TerminalOperation::EnableBracketed,
            TerminalOperation::PushKeyboard,
            TerminalOperation::DisableRaw,
            TerminalOperation::DisableBracketed,
            TerminalOperation::Newline,
        ]
    );
}

#[test]
fn terminal_cleanup_attempts_every_operation_and_retries_failures() {
    let cleanup_operations = [
        TerminalOperation::DisableRaw,
        TerminalOperation::DisableBracketed,
        TerminalOperation::PopKeyboard,
        TerminalOperation::Newline,
    ];

    for failed_operation in cleanup_operations {
        let backend = MockTerminal::default();
        let mut ui = successful_terminal_ui(&backend);
        backend.fail_next(failed_operation);

        let error = ui.restore().unwrap_err();
        assert!(
            format!("{error:#}").contains(&format!("mock {failed_operation:?}")),
            "first cleanup error was not reported: {error:#}"
        );
        let calls = backend.calls();
        for operation in cleanup_operations {
            assert_eq!(
                calls.iter().filter(|called| **called == operation).count(),
                1,
                "cleanup did not attempt {operation:?} after {failed_operation:?} failed"
            );
        }

        // The one-shot failure was consumed. Successful operations are no
        // longer retried, while the failed operation remains tracked.
        ui.restore().unwrap();
        let calls = backend.calls();
        assert_eq!(
            calls.last(),
            Some(&failed_operation),
            "cleanup did not retry {failed_operation:?}"
        );
    }
}

#[test]
fn terminal_cleanup_reports_the_first_error_after_all_attempts() {
    let backend = MockTerminal::default();
    let mut ui = successful_terminal_ui(&backend);
    backend.fail_next(TerminalOperation::DisableRaw);
    backend.fail_next(TerminalOperation::DisableBracketed);

    let error = ui.restore().unwrap_err();
    let rendered = format!("{error:#}");
    assert!(rendered.contains("mock DisableRaw"));
    assert!(!rendered.contains("mock DisableBracketed"));
    let calls = backend.calls();
    assert!(calls.contains(&TerminalOperation::DisableRaw));
    assert!(calls.contains(&TerminalOperation::DisableBracketed));
    assert!(calls.contains(&TerminalOperation::PopKeyboard));
    assert!(calls.contains(&TerminalOperation::Newline));

    // Both failed modes remain tracked and can be restored on a later
    // attempt; the already restored modes are not touched again.
    ui.restore().unwrap();
    let calls = backend.calls();
    assert_eq!(
        &calls[calls.len() - 2..],
        [
            TerminalOperation::DisableRaw,
            TerminalOperation::DisableBracketed
        ]
    );
}

#[test]
fn panic_cleanup_attempts_all_modes_and_drop_retries_unfinished_cleanup() {
    static PANIC_HOOK_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    let _hook_guard = PANIC_HOOK_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap();

    let backend = MockTerminal::default();
    let ui = successful_terminal_ui(&backend);
    for operation in [
        TerminalOperation::DisableRaw,
        TerminalOperation::DisableBracketed,
        TerminalOperation::PopKeyboard,
        TerminalOperation::Newline,
    ] {
        backend.fail_next(operation);
    }

    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    install_panic_hook(&ui.terminal);
    // A panic on another thread (for example a Tokio task whose panic is
    // caught) must leave the live UI's terminal modes alone.
    let worker = std::thread::spawn(|| panic!("simulated worker panic"));
    assert!(worker.join().is_err());
    assert!(
        backend.calls().iter().all(|called| !matches!(
            called,
            TerminalOperation::DisableRaw
                | TerminalOperation::DisableBracketed
                | TerminalOperation::PopKeyboard
                | TerminalOperation::Newline
        )),
        "a worker-thread panic restored the UI terminal"
    );
    let panic_result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        panic!("simulated terminal panic");
    }));
    std::panic::set_hook(previous);
    assert!(panic_result.is_err());

    let calls = backend.calls();
    for operation in [
        TerminalOperation::DisableRaw,
        TerminalOperation::DisableBracketed,
        TerminalOperation::PopKeyboard,
        TerminalOperation::Newline,
    ] {
        assert_eq!(
            calls.iter().filter(|called| **called == operation).count(),
            1
        );
    }

    // Drop is a second cleanup boundary. The failed operations were not
    // cleared by the panic hook, so this retry completes all restoration.
    drop(ui);
    let calls = backend.calls();
    for operation in [
        TerminalOperation::DisableRaw,
        TerminalOperation::DisableBracketed,
        TerminalOperation::PopKeyboard,
        TerminalOperation::Newline,
    ] {
        assert_eq!(
            calls.iter().filter(|called| **called == operation).count(),
            2
        );
    }
}

#[test]
fn ctrl_c_clears_a_draft_before_closing_harness() {
    let mut ui = ui(80, 24);
    ui.input = "unfinished prompt".into();
    ui.cursor = ui.input.len();
    ui.history_pos = Some(0);
    ui.draft = "older draft".into();
    ui.busy = true;
    let (tx, mut rx) = mpsc::unbounded_channel();
    let cancel = CancellationToken::new();
    let ctrl_c = Event::Key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL));

    assert!(!ui.handle_input(&ctrl_c, &tx, &cancel).unwrap());
    assert!(ui.input.is_empty());
    assert_eq!(ui.cursor, 0);
    assert_eq!(ui.history_pos, None);
    assert!(ui.draft.is_empty());
    assert!(!cancel.is_cancelled());
    assert!(
        rx.try_recv().is_err(),
        "clearing must not interrupt the turn"
    );

    assert!(ui.handle_input(&ctrl_c, &tx, &cancel).unwrap());
    assert!(cancel.is_cancelled());
    assert!(rx.try_recv().is_err());
}

#[test]
fn shift_enter_inserts_a_newline_without_submitting() {
    let mut ui = ui(80, 24);
    ui.input = "first line".into();
    ui.cursor = ui.input.len();
    let (tx, mut rx) = mpsc::unbounded_channel();
    let cancel = CancellationToken::new();
    let shift_enter = Event::Key(KeyEvent::new(KeyCode::Enter, KeyModifiers::SHIFT));

    assert!(!ui.handle_input(&shift_enter, &tx, &cancel).unwrap());
    assert_eq!(ui.input, "first line\n");
    assert_eq!(ui.cursor, ui.input.len());
    assert!(rx.try_recv().is_err());

    let enter = Event::Key(KeyEvent::new(KeyCode::Enter, KeyModifiers::empty()));
    assert!(!ui.handle_input(&enter, &tx, &cancel).unwrap());
    assert_eq!(
        rx.try_recv().unwrap(),
        InputMessage::Message("first line\n".into())
    );
}

#[test]
fn slash_background_results_do_not_finish_an_active_turn() {
    let mut ui = ui(80, 24);
    ui.apply_event(UiEvent::TextDelta("partial answer".into()));
    ui.apply_event(start("tool", "reading file"));
    ui.apply_event(UiEvent::TextDelta("more answer".into()));
    for id in [1, 2] {
        ui.apply_event(UiEvent::BackgroundOperationStarted {
            id,
            label: format!("Lookup {id}..."),
        });
    }
    ui.apply_event(UiEvent::BackgroundOperationFinished {
        id: 1,
        error: Some("usage request failed".into()),
    });
    assert!(ui.busy);
    assert_eq!(ui.stream.as_ref().unwrap().markdown, "more answer");
    assert_eq!(ui.running_tools.len(), 1);
    assert_eq!(ui.background_operations.len(), 1);
    assert_eq!(ui.background_operations[0].0, 2);
    assert!(
        ui.running_region()
            .lines
            .iter()
            .any(|line| row_text(line).contains("Lookup 2"))
    );
    ui.apply_event(UiEvent::TurnFinished);
    assert!(!ui.busy);
    assert_eq!(ui.background_operations.len(), 1);
    ui.apply_event(UiEvent::BackgroundOperationFinished { id: 2, error: None });
    assert!(ui.background_operations.is_empty());
}

#[test]
fn slash_compact_and_skill_show_activity_before_output() {
    for command in ["/compact", "/skill review"] {
        let mut ui = ui(80, 24);
        ui.skills.push(SkillEntry {
            name: "review".into(),
            description: "Review code".into(),
        });
        let (tx, _rx) = mpsc::unbounded_channel();
        ui.submit_command(command, &tx).unwrap();
        assert!(ui.busy);
        ui.apply_event(UiEvent::OperationFinished);
        assert!(!ui.busy);
        // A command queued during another turn gets fresh progress at dispatch.
        ui.apply_event(UiEvent::OperationStarted {
            label: "Compacting session...".into(),
        });
        let region = ui.build_region(&ui.input_layout());
        assert!(
            region
                .rows
                .iter()
                .any(|line| row_text(line).contains("Compacting session"))
        );
        ui.apply_event(UiEvent::OperationFinished);
        assert!(!ui.busy);
    }
}

#[test]
fn slash_session_completion_is_silent_but_sessions_command_reports_empty_list() {
    let mut ui = ui(80, 24);
    ui.apply_event(UiEvent::SessionList {
        sessions: Vec::new(),
    });
    assert!(ui.pending.is_empty());
    let (tx, _rx) = mpsc::unbounded_channel();
    ui.submit_command("/sessions", &tx).unwrap();
    ui.apply_event(UiEvent::SessionList {
        sessions: Vec::new(),
    });
    assert!(ui.pending.iter().any(
        |entry| matches!(entry, Entry::Notice { text } if text == "No sessions for this workspace")
    ));
}

#[test]
fn slash_empty_skills_still_show_discovery_errors() {
    let mut ui = ui(80, 24);
    ui.apply_event(UiEvent::SkillsLoaded {
        skills: Vec::new(),
        diagnostics: vec!["review/SKILL.md: invalid frontmatter".into()],
        empty: true,
    });
    assert!(
        matches!(&ui.pending[0], Entry::Notice { text } if text.contains("no skills discovered") && text.contains("invalid frontmatter"))
    );
}

#[test]
fn slash_skill_alias_is_echoed_once() {
    let mut ui = ui(80, 24);
    ui.skills.push(SkillEntry {
        name: "review".into(),
        description: "Review code".into(),
    });
    let (tx, _rx) = mpsc::unbounded_channel();
    ui.submit_command("/review", &tx).unwrap();
    assert_eq!(ui.pending.len(), 1);
    assert!(matches!(&ui.pending[0], Entry::Notice { text } if text == "/review · skill: review"));
}

#[test]
fn reasoning_command_reports_current_value_and_sends_changes() {
    let mut ui = ui(80, 24);
    let (tx, mut rx) = mpsc::unbounded_channel();

    ui.submit_command("/reasoning", &tx).unwrap();
    assert!(rx.try_recv().is_err());
    assert!(matches!(
        ui.pending.last(),
        Some(Entry::Notice { text }) if text.contains("current: auto")
    ));

    ui.submit_command("/reasoning high", &tx).unwrap();
    assert_eq!(
        rx.try_recv().unwrap(),
        InputMessage::SetReasoning {
            level: "high".into()
        }
    );
    ui.apply_event(UiEvent::ReasoningChanged {
        level: "high".into(),
    });
    ui.submit_command("/reasoning", &tx).unwrap();
    assert!(matches!(
        ui.pending.last(),
        Some(Entry::Notice { text }) if text.contains("current: high")
    ));
}

#[test]
fn concurrent_starts_accumulate_and_finishes_remove_by_call_id() {
    let mut ui = ui(80, 24);
    // Two same-name subagents with similar summaries — correlation must
    // be by call id, not by name or FIFO position.
    ui.apply_event(start("a", "subagent: audit one"));
    ui.apply_event(start("b", "subagent: audit two"));
    assert_eq!(
        active_summaries(&ui),
        vec!["subagent: audit one", "subagent: audit two"]
    );

    // Finish out of order: B first, A second.
    ui.apply_event(finish("b", "subagent: audit two"));
    assert_eq!(active_summaries(&ui), vec!["subagent: audit one"]);
    ui.apply_event(finish("a", "subagent: audit one"));
    assert!(ui.running_tools.is_empty());

    // Both completions were committed as tool entries.
    let committed = ui
        .pending
        .iter()
        .filter(|entry| matches!(entry, Entry::Tool { .. }))
        .count();
    assert_eq!(committed, 2);
}

#[test]
fn duplicate_start_updates_in_place_instead_of_duplicating() {
    let mut ui = ui(80, 24);
    ui.apply_event(start("a", "first"));
    ui.apply_event(start("a", "second"));
    assert_eq!(active_summaries(&ui), vec!["second"]);
}

#[test]
fn error_drains_all_active_tools_as_failed_and_turn_finished_clears() {
    let mut failing = ui(80, 24);
    failing.apply_event(start("a", "one"));
    failing.apply_event(start("b", "two"));
    failing.apply_event(UiEvent::Error("turn blew up".into()));
    assert!(failing.running_tools.is_empty());
    let failed = failing
        .pending
        .iter()
        .filter_map(|entry| match entry {
            Entry::Tool { record } => Some(record),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(failed.len(), 2);
    assert!(
        failed
            .iter()
            .all(|record| record.status == ToolStatus::Failure)
    );

    // TurnFinished clears any remaining actives without committing them.
    let mut fresh = ui(80, 24);
    fresh.apply_event(start("a", "one"));
    fresh.apply_event(UiEvent::TurnFinished);
    assert!(fresh.running_tools.is_empty());
}

#[test]
fn region_renders_every_active_tool_plus_overflow_line() {
    let mut ui = ui(80, 24);
    for index in 0..6 {
        ui.apply_event(start(
            &format!("call-{index}"),
            &format!("subagent: task {index}"),
        ));
    }
    let input = ui.input_layout();
    let build = ui.build_region(&input);
    let text = build
        .rows
        .iter()
        .map(row_text)
        .collect::<Vec<_>>()
        .join("\n");
    // Four visible in launch order, the rest folded into the overflow.
    assert!(text.contains("subagent: task 0"));
    assert!(text.contains("subagent: task 3"));
    assert!(text.contains("… 2 more running"));
    // The whole region still fits the terminal height.
    assert!(build.rows.len() <= 24);
}

#[test]
fn format_tokens_uses_k_and_m_units() {
    assert_eq!(format_tokens(0), "0");
    assert_eq!(format_tokens(999), "999");
    assert_eq!(format_tokens(1_000), "1k");
    assert_eq!(format_tokens(1_234), "1.2k");
    assert_eq!(format_tokens(12_345), "12.3k");
    assert_eq!(format_tokens(1_000_000), "1M");
    assert_eq!(format_tokens(1_500_000), "1.5M");
}

#[test]
fn placeholder_row_stays_bare_without_usage() {
    let layout = input_layout("", 0, 60, Theme::default(), "", "", "");
    assert_eq!(row_text(&layout.rows[0]), "› Type your message...");
}

#[test]
fn usage_updated_is_kept_for_the_placeholder_counter() {
    let mut ui = ui(80, 24);
    ui.apply_event(UiEvent::UsageUpdated {
        input_tokens: 1_234,
        output_tokens: 3_456,
        cached_tokens: 5,
        reasoning_tokens: 2,
        cost: "$0.01".into(),
    });
    ui.apply_event(UiEvent::ContextUsageUpdated {
        used_tokens: 24_000,
        max_tokens: 128_000,
    });
    let layout = ui.input_layout();
    assert_eq!(
        row_text(&layout.rows[0]),
        "› Type your message...   ↑ 1.2k ↓ 3.5k · 24k/128k ctx"
    );
}

#[test]
fn formats_subscription_usage_windows_and_resets() {
    let usage = SubscriptionUsage {
        plan: Some("Go".into()),
        windows: vec![
            SubscriptionUsageWindow {
                label: "rolling".into(),
                used_percent: 0,
                status: Some("ok".into()),
                resets_at: Some("2026-08-25T14:48:50.201Z".into()),
                resets_after_seconds: None,
            },
            SubscriptionUsageWindow {
                label: "monthly".into(),
                used_percent: 54,
                status: Some("limited".into()),
                resets_at: None,
                resets_after_seconds: Some(553_554),
            },
        ],
    };
    let now = parse_reset_timestamp("2026-08-25T10:00:00Z").unwrap();
    assert_eq!(
        format_subscription_usage_at("opencode-go", &usage, now),
        "opencode-go subscription usage · Go\nrolling    0% used · resets in 4h 48m\nmonthly   54% used · limited · resets in 6d 9h"
    );
}

#[test]
fn input_layout_wraps_and_places_the_cursor() {
    // "abcdefgh" at width 4 wraps into two full rows; a cursor at the
    // very end lands on a fresh third row (mirrors prompt_layout).
    let layout = input_layout("abcdefgh", 8, 4, Theme::default(), "", "", "");
    let values: Vec<String> = layout.rows.iter().map(row_text).collect();
    assert_eq!(values, vec!["› abcd", "  efgh", "  "]);
    assert_eq!(layout.cursor_row, 2);
    assert_eq!(layout.cursor_col, 0);

    // Cursor mid-word stays on the first row at the right column.
    let layout = input_layout("abcdef", 2, 4, Theme::default(), "", "", "");
    assert_eq!(layout.cursor_row, 0);
    assert_eq!(layout.cursor_col, 2);
}

#[test]
fn input_layout_handles_multiline_drafts() {
    // Two logical lines, cursor at the end of the second: continuation
    // rows are indented to line up under the prefix.
    let layout = input_layout("ab\ncd", 5, 10, Theme::default(), "", "", "");
    let values: Vec<String> = layout.rows.iter().map(row_text).collect();
    assert_eq!(values, vec!["› ab", "  cd"]);
    assert_eq!(layout.cursor_row, 1);
    assert_eq!(layout.cursor_col, 2);

    // Empty input shows the placeholder with the cursor after `› `.
    let layout = input_layout("", 0, 10, Theme::default(), "", "", "");
    assert_eq!(row_text(&layout.rows[0]), "› Type your ");
    assert_eq!(layout.cursor_row, 0);
    assert_eq!(layout.cursor_col, 0);
}

#[test]
fn input_layout_counts_wide_characters_by_display_width() {
    // Two CJK characters fill a width-4 row; the cursor after the first
    // sits at display column 2, not byte offset 3.
    let layout = input_layout("你好", 3, 4, Theme::default(), "", "", "");
    assert_eq!(layout.cursor_row, 0);
    assert_eq!(layout.cursor_col, 2);
}

#[test]
fn tool_lines_are_one_shell_style_line() {
    // Running: just the command with the `$` marker.
    let lines = tool_lines(&record(ToolStatus::Running), false, 60, Theme::default());
    assert_eq!(lines.len(), 1);
    assert_eq!(row_text(&lines[0]), "$ cargo test");

    // Success: duration suffix.
    let lines = tool_lines(&record(ToolStatus::Success), false, 60, Theme::default());
    assert_eq!(row_text(&lines[0]), "$ cargo test · 1.2s");

    // Failure: cross and the first error line only.
    let lines = tool_lines(&record(ToolStatus::Failure), false, 60, Theme::default());
    assert_eq!(row_text(&lines[0]), "$ cargo test  ✗ boom");

    // Non-bash tools keep their `tool param` summary shape, with the
    // verb accented like the `$` marker.
    let mut read = record(ToolStatus::Success);
    read.summary = "read src/main.rs".into();
    let lines = tool_lines(&read, false, 60, Theme::default());
    assert_eq!(row_text(&lines[0]), "read src/main.rs · 1.2s");
    assert_eq!(
        lines[0].runs[0].style.foreground_color,
        Some(Theme::default().accent)
    );
    assert_eq!(lines[0].runs[0].text, "read");
    assert_eq!(lines[0].runs[1].text, " src/main.rs");

    // A single-word summary (no parameter) is all accent, no crash.
    let mut bare = record(ToolStatus::Running);
    bare.summary = "compact".into();
    let lines = tool_lines(&bare, false, 60, Theme::default());
    assert_eq!(row_text(&lines[0]), "compact");
    assert_eq!(lines[0].runs.len(), 1);
}

#[test]
fn tool_lines_expanded_show_output_tail_and_error() {
    // A finished tool with output and an error: expanded shows the
    // bounded tail plus later diagnostic lines, indented without repeating
    // the first preview line.
    let mut failed = record(ToolStatus::Failure);
    failed.output = (0..10)
        .map(|i| format!("out line {i}"))
        .collect::<Vec<_>>()
        .join("\n");
    let lines = tool_lines(&failed, true, 60, Theme::default());
    // 1 summary + 4 tail lines (… N above + last 4) + 1 error line.
    assert_eq!(lines.len(), 1 + 5 + 1);
    assert_eq!(row_text(&lines[0]), "$ cargo test  ✗ boom");
    assert_eq!(row_text(&lines[1]), "  … 6 lines above");
    assert_eq!(row_text(&lines[5]), "  out line 9");
    assert_eq!(row_text(&lines[6]), "  trace");

    // Collapsed stays compact regardless of retained output.
    let collapsed = tool_lines(&failed, false, 60, Theme::default());
    assert_eq!(collapsed.len(), 1);

    // A running tool gains a trailing `running…` marker when expanded.
    let lines = tool_lines(&record(ToolStatus::Running), true, 60, Theme::default());
    assert_eq!(row_text(lines.last().unwrap()), "  running…");
}

#[test]
fn startup_update_notice_is_rendered_after_header_metadata() {
    let mut state = CrossTerm::base(
        "model",
        "provider",
        Vec::new(),
        StartupEntries {
            update_notice: Some("New version available: v1.2.3 (run `harness update`)".into()),
            ..StartupEntries::default()
        },
        PathBuf::from("."),
        80,
        24,
    );
    state.enqueue_welcome();
    assert!(
        matches!(state.pending.last(), Some(Entry::Notice { text }) if text.contains("v1.2.3"))
    );
}

#[test]
fn metadata_header_shows_context_mcp_and_skill_rows_capped() {
    let theme = Theme::default();
    let lines = metadata_lines(
        "~/proj",
        Some("feature"),
        "p",
        "m",
        &["AGENTS.md".into(), "~/.harness/AGENTS.md".into()],
        &["github".into(), "filesystem".into()],
        &["alpha".into(), "beta".into()],
        theme,
    );
    assert_eq!(lines.len(), 6);
    assert_eq!(row_text(&lines[0]), "│ cwd       ~/proj  (feature)");
    assert_eq!(row_text(&lines[1]), "│ provider  p");
    assert_eq!(row_text(&lines[2]), "│ model     m");
    assert_eq!(
        row_text(&lines[3]),
        "│ context   AGENTS.md, ~/.harness/AGENTS.md"
    );
    assert_eq!(row_text(&lines[4]), "│ skills    alpha, beta");
    assert_eq!(row_text(&lines[5]), "│ mcps      github, filesystem");
    for line in &lines {
        assert_eq!(line.runs[0].style.foreground_color, Some(theme.accent));
        assert!(line.runs[1].style.attributes.has(Attribute::Dim));
        assert_eq!(
            line.runs[2].style.foreground_color,
            Some(theme.primary_text)
        );
        assert!(!line.runs[2].style.attributes.has(Attribute::Dim));
    }

    // More than the cap folds into `+N more`.
    let many = (0..6).map(|index| format!("s{index}")).collect::<Vec<_>>();
    let lines = metadata_lines("~/proj", None, "p", "m", &[], &[], &many, theme);
    assert_eq!(lines.len(), 4);
    assert_eq!(row_text(&lines[3]), "│ skills    s0, s1, s2, s3 · +2 more");

    // Nothing loaded: no extra rows.
    let lines = metadata_lines("~/proj", None, "p", "m", &[], &[], &[], theme);
    assert_eq!(lines.len(), 3);
}

#[test]
fn long_metadata_rows_fit_narrow_widths() {
    // TUI-2: header metadata (cwd/branch/provider/model/context/skills)
    // goes through `entry_lines` → `fit_line_to_width`, so even
    // pathological inputs fit widths 1–3 as well as normal widths.
    let long_cwd = "/very/long/working/directory/that/keeps/going/and/going";
    let long_branch = "feature/extremely-long-branch-name-that-never-ends";
    let long_model = "some-provider-model-with-an-absurdly-long-identifier-v99";
    let many_context: Vec<String> = (0..10)
        .map(|index| format!("/deep/path/to/AGENTS-{index}.md"))
        .collect();
    let many_mcps: Vec<String> = (0..10).map(|index| format!("mcp-server-{index}")).collect();
    let many_skills: Vec<String> = (0..10)
        .map(|index| format!("skill-number-{index}"))
        .collect();
    for width in [1usize, 2, 3, 10, 40, 80] {
        let lines = metadata_lines(
            long_cwd,
            Some(long_branch),
            "provider-with-a-long-name",
            long_model,
            &many_context,
            &many_mcps,
            &many_skills,
            Theme::default(),
        );
        // Mirror the `entry_lines` Metadata path: fit every row.
        let fitted: Vec<_> = lines
            .iter()
            .map(|line| crate::render::fit_line_to_width(line, width))
            .collect();
        assert!(
            fitted.iter().all(|line| row_width(line) <= width),
            "width {width}: {:?}",
            fitted.iter().map(row_text).collect::<Vec<_>>()
        );
    }
}

#[test]
fn line_to_ansi_sanitizes_untrusted_content_but_keeps_generated_styles() {
    let line = Row::from(Run::new(
        "safe\t\u{1b}[2J\u{1b}]52;c;secret\u{07}done",
        ContentStyle::default().fg(Theme::default().accent),
    ));
    let ansi = ansi_of(&line);
    assert!(ansi.contains("safe    done"), "{ansi:?}");
    assert!(!ansi.contains("secret"), "OSC payload leaked: {ansi:?}");
    assert!(!ansi.contains("[2J"), "CSI payload leaked: {ansi:?}");
    assert!(
        ansi.contains("\u{1b}["),
        "generated style is missing: {ansi:?}"
    );
}

#[test]
fn line_to_ansi_emits_style_codes_only_for_styled_spans() {
    let styled = Row::from(Run::new(
        "hi",
        ContentStyle::default()
            .fg(Theme::default().accent)
            .add_modifier(Attribute::Bold),
    ));
    let ansi = ansi_of(&styled);
    assert!(ansi.contains("\u{1b}["));
    assert!(ansi.ends_with("\u{1b}[0m"));

    let plain = Row::plain("plain");
    assert_eq!(ansi_of(&plain), "plain");
}

#[test]
fn line_to_ansi_resets_bold_between_the_input_prefix_and_text() {
    let layout = input_layout("first\nsecond", 12, 40, Theme::default(), "", "", "");
    let first = ansi_of(&layout.rows[0]);
    let second = ansi_of(&layout.rows[1]);

    // The bold prefix is followed by a reset before normal input text;
    // both logical input rows therefore render at the same weight.
    assert!(first.contains("› \u{1b}[0m"));
    assert!(!second.contains(&format!(
        "{}",
        crossterm::style::SetAttribute(Attribute::Bold)
    )));
}

#[test]
fn editor_helpers_respect_char_boundaries() {
    let mut input = String::new();
    let mut cursor = 0usize;
    insert_text(&mut input, &mut cursor, "aé你");
    assert_eq!(cursor, 1 + 2 + 3);

    delete_backward(&mut input, &mut cursor);
    assert_eq!(input, "aé");
    assert_eq!(cursor, 3);

    move_left(&input, &mut cursor);
    assert_eq!(cursor, 1);
    move_right(&input, &mut cursor);
    assert_eq!(cursor, 3);
    delete_forward(&mut input, &mut cursor);
    assert_eq!(input, "aé");

    // CRLF paste payloads normalize to a single newline.
    let mut input = String::new();
    let mut cursor = 0usize;
    insert_text(&mut input, &mut cursor, "a\r\nb");
    assert_eq!(input, "a\nb");
}

#[test]
fn vertical_move_crosses_logical_lines_and_preserves_columns() {
    let input = "abc\ndefghi\nx";
    // Cursor 7 sits after "def" (column 3): moving up lands after "abc".
    assert_eq!(vertical_move(input, 7, -1), Some(3));
    assert_eq!(vertical_move(input, 1, 1), Some(5)); // after "a" → "d|efghi"
    assert_eq!(vertical_move(input, 1, -1), None); // already on the first line
    assert_eq!(vertical_move(input, input.len(), 1), None); // last line
}

#[test]
fn narrow_build_regions_keep_rows_and_cursor_bounded() {
    for width in 1..=3 {
        let mut ui = ui(width, 4);
        ui.input = "你好\ttext".into();
        ui.cursor = ui.input.len();
        ui.busy = true;
        let input = ui.input_layout();
        let build = ui.build_region(&input);
        assert!(build.rows.len() <= 4);
        assert!(
            build
                .rows
                .iter()
                .all(|line| row_width(line) <= render::content_width(width))
        );
        assert!(build.cursor_row < build.rows.len());
        assert!(build.cursor_col < width as usize);
    }
}

#[test]
fn activity_row_is_cleared_when_height_clipping_drops_it() {
    for height in [1, 2] {
        let mut ui = ui(20, height);
        ui.busy = true;
        let input = ui.input_layout();
        let build = ui.build_region(&input);
        assert_eq!(build.activity_row, None, "height {height}");
        ui.region = build.rows;
        ui.activity_region_row = build.activity_row;
        assert!(!ui.repaint_activity_only().unwrap());
    }
}

#[test]
fn plain_home_and_end_match_ctrl_a_and_ctrl_e() {
    let (input_tx, _input_rx) = mpsc::unbounded_channel();
    let cancel = CancellationToken::new();
    let mut ui = ui(40, 10);
    ui.input = "first\nsecond".into();
    ui.cursor = ui.input.len();

    ui.handle_input(
        &Event::Key(KeyEvent::new(KeyCode::Home, KeyModifiers::NONE)),
        &input_tx,
        &cancel,
    )
    .unwrap();
    assert_eq!(ui.cursor, "first\n".len());
    ui.handle_input(
        &Event::Key(KeyEvent::new(KeyCode::End, KeyModifiers::NONE)),
        &input_tx,
        &cancel,
    )
    .unwrap();
    assert_eq!(ui.cursor, ui.input.len());

    ui.cursor = ui.input.len();
    ui.handle_input(
        &Event::Key(KeyEvent::new(KeyCode::Char('a'), KeyModifiers::CONTROL)),
        &input_tx,
        &cancel,
    )
    .unwrap();
    assert_eq!(ui.cursor, "first\n".len());
    ui.handle_input(
        &Event::Key(KeyEvent::new(KeyCode::Char('e'), KeyModifiers::CONTROL)),
        &input_tx,
        &cancel,
    )
    .unwrap();
    assert_eq!(ui.cursor, ui.input.len());
}

#[test]
fn stream_caches_completed_markdown_blocks() {
    let mut ui = ui(80, 24);
    ui.apply_event(UiEvent::TextDelta("first\n\nsecond".into()));
    let first_cache = ui.stream_markdown_cache.as_ref().unwrap();
    assert_eq!(first_cache.source_offset, "first\n\n".len());
    assert_eq!(first_cache.rendered_blocks, 1);
    let first_lines = first_cache.lines.clone();
    ui.apply_event(UiEvent::TextDelta(" more".into()));
    let cache = ui.stream_markdown_cache.as_ref().unwrap();
    assert_eq!(cache.source_offset, "first\n\n".len());
    assert_eq!(cache.rendered_blocks, 1);
    assert_eq!(cache.lines, first_lines);
    let rendered = ui.stream_tail_lines(render::content_width(ui.width));
    assert!(row_text(rendered.last().unwrap()).contains("second"));
}

#[test]
fn history_window_visits_newest_entries_and_marks_hidden_rows() {
    let entries = vec![
        Entry::User { text: "old".into() },
        Entry::User {
            text: "middle".into(),
        },
        Entry::User { text: "new".into() },
    ];
    let rows = history_window(&entries, 3, 80, Theme::default(), false);
    assert_eq!(rows.len(), 3);
    assert!(row_text(&rows[0]).contains("older rows"));
    assert!(row_text(&rows[2]).contains("new"));
}

#[test]
fn build_region_clips_the_streaming_tail_to_the_screen() {
    let mut ui = ui(80, 10);
    ui.stream = Some(StreamState {
        reasoning: String::new(),
        // One giant single block: no stable split, so the display clip is
        // the only bound. The full text prints when the message
        // finalizes.
        markdown: (0..60).map(|i| format!("line {i}\n")).collect(),
        ..StreamState::default()
    });
    let input = ui.input_layout();
    let build = ui.build_region(&input);
    assert!(build.rows.len() <= 10, "region must fit the screen");
    // The newest rows are visible and the input line is at the bottom.
    let last = row_text(build.rows.last().expect("input row"));
    assert_eq!(last, format!("› {PLACEHOLDER}"));
    let text: String = build
        .rows
        .iter()
        .map(row_text)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(text.contains("line 59"));
    assert!(!text.contains("line 0"));
}

#[test]
fn build_region_clips_huge_input_around_the_cursor() {
    let mut ui = ui(80, 10);
    ui.input = (0..50)
        .map(|i| format!("row{i}"))
        .collect::<Vec<_>>()
        .join("\n");
    ui.cursor = ui.input.len();
    let input = ui.input_layout();
    assert!(input.rows.len() > 10);
    let build = ui.build_region(&input);
    assert!(build.rows.len() <= 10);
    // The cursor (at the end of the draft) stays inside the window.
    assert!(build.cursor_row < build.rows.len());
    let last = row_text(build.rows.last().expect("clipped input"));
    assert_eq!(last, "  row49");
}

#[test]
fn commit_stream_prefix_moves_stable_blocks_into_pending() {
    let mut ui = ui(80, 8);
    // A first paragraph tall enough to outgrow the height-8 tail budget
    // (~4 rows), followed by an unfinished second paragraph.
    let tall = format!("{}\n\n", "word ".repeat(100));
    ui.stream = Some(StreamState {
        reasoning: String::new(),
        markdown: tall.clone() + &"second ".repeat(30),
        ..StreamState::default()
    });
    let input = ui.input_layout();
    ui.commit_stream_prefix(input.rows.len());
    // The stable first paragraph was committed and drained from the live
    // stream.
    assert_eq!(ui.pending.len(), 1);
    assert_eq!(
        ui.stream.as_ref().expect("tail remains").markdown,
        "second ".repeat(30)
    );
    match &ui.pending[0] {
        Entry::Assistant { markdown, .. } => assert_eq!(markdown, &tall),
        other => panic!("expected assistant prefix entry, got {other:?}"),
    }
}

#[test]
fn full_repaint_prints_a_stream_prefix_committed_during_the_repaint() {
    // A full repaint prints only `transcript`. A prefix committed after
    // pending entries moved there was never printed.
    let mut ui = ui(80, 8);
    let tall = format!("{}\n\n", "word ".repeat(100));
    ui.stream = Some(StreamState {
        reasoning: String::new(),
        markdown: tall.clone() + &"second ".repeat(30),
        ..StreamState::default()
    });
    let (_build, _oversized) = ui.stage_full_repaint();
    assert!(ui.pending.is_empty());
    let prefixes = ui
        .transcript
        .iter()
        .filter(|entry| matches!(entry, Entry::Assistant { markdown, .. } if *markdown == tall))
        .count();
    assert_eq!(prefixes, 1);
}

#[test]
fn commit_stream_prefix_keeps_short_blocks_live() {
    let mut ui = ui(80, 24);
    ui.stream = Some(StreamState {
        reasoning: String::new(),
        markdown: "short\n\nanswer".into(),
        ..StreamState::default()
    });
    let input = ui.input_layout();
    ui.commit_stream_prefix(input.rows.len());
    assert!(ui.pending.is_empty());
    assert_eq!(
        ui.stream.as_ref().expect("stream remains").markdown,
        "short\n\nanswer"
    );
}

#[test]
fn continued_chunks_do_not_repeat_the_assistant_marker() {
    let mut ui = ui(80, 8);
    ui.stream = Some(StreamState {
        markdown: format!("{}\n\nsecond\n\nthird", "word ".repeat(100)),
        ..StreamState::default()
    });
    let input = ui.input_layout();
    ui.commit_stream_prefix(input.rows.len());
    ui.finalize_stream();
    let rows: Vec<String> = ui
        .pending
        .iter()
        .flat_map(|entry| entry_lines(entry, 76, ui.theme, false))
        .map(|line| row_text(&line))
        .collect();
    assert_eq!(rows.iter().filter(|row| row.starts_with('‹')).count(), 1);
    assert!(rows.iter().any(|row| row.trim() == "third"));
}

#[test]
fn long_reasoning_streams_into_scrollback_by_paragraph() {
    let mut ui = ui(80, 8);
    let paragraph = "think ".repeat(60);
    ui.stream = Some(StreamState {
        reasoning: format!("{paragraph}\n\n{paragraph}\n\nlast"),
        ..StreamState::default()
    });
    let input = ui.input_layout();
    ui.commit_stream_prefix(input.rows.len());
    assert!(!ui.pending.is_empty());
    let rest = &ui.stream.as_ref().expect("stream").reasoning;
    assert!(rest.ends_with("last"));
    assert!(
        rest.matches("think").count() <= 60,
        "committed paragraphs left the stream"
    );
}

#[test]
fn copy_returns_rejoined_reply_or_last_code_block() {
    let mut ui = ui(80, 24);
    ui.pending.push(Entry::Assistant {
        markdown: "intro\n\n```rs\nlet a = 1;\n```\n\n".into(),
        reasoning: String::new(),
        continued: false,
    });
    ui.pending.push(Entry::Assistant {
        markdown: "tail".into(),
        reasoning: String::new(),
        continued: true,
    });
    assert_eq!(
        ui.last_reply_markdown().as_deref(),
        Some("intro\n\n```rs\nlet a = 1;\n```\n\ntail")
    );
    ui.copy_last_reply(true);
    let osc = ui.clipboard_osc.take().expect("queued");
    assert_eq!(osc, clipboard_sequence("let a = 1;", false));
    assert!(osc.contains("bGV0IGEgPSAxOw=="));
    assert!(clipboard_sequence("x", true).starts_with("\u{1b}Ptmux;"));
}

#[test]
fn input_offset_at_inverts_the_layout_cursor() {
    let theme = Theme::default();
    let input = "héllo wörld 你好 wide\nsecond line that wraps around\n\nend";
    for width in [5usize, 8, 13, 40] {
        for cursor in input
            .char_indices()
            .map(|(index, _)| index)
            .chain([input.len()])
        {
            let layout = input_layout(input, cursor, width, theme, "", "", "");
            // A cursor parked at a full row's end sits on a fresh row
            // that has no text to click, so only check mid-row cursors.
            let at_end = cursor == input.len();
            if at_end {
                continue;
            }
            let got = input_offset_at(input, cursor, width, layout.cursor_row, layout.cursor_col);
            assert_eq!(got, Some(cursor), "width {width} cursor {cursor}");
        }
    }
    assert_eq!(input_offset_at("ab", 0, 10, 0, 99), Some(2));
    assert_eq!(input_offset_at("ab", 0, 10, 1, 0), None);
}

#[test]
fn input_offset_at_counts_the_cursor_only_row() {
    // `abcd` fills width 4, so a cursor at its end gets an extra empty row
    // and `x` renders on row 2.
    let input = "abcd\nx";
    let layout = input_layout(input, 4, 4, Theme::default(), "", "", "");
    assert_eq!(layout.cursor_row, 1);
    assert_eq!(input_offset_at(input, 4, 4, 1, 0), Some(4));
    assert_eq!(input_offset_at(input, 4, 4, 2, 0), Some(5));
    assert_eq!(input_offset_at(input, 4, 4, 3, 0), None);
    // With the cursor elsewhere there is no extra row.
    assert_eq!(input_offset_at(input, 6, 4, 1, 0), Some(5));
}

#[test]
fn copy_rejoins_reply_chunks_across_interleaved_notices() {
    let mut ui = ui(80, 24);
    ui.pending.push(Entry::Assistant {
        markdown: "earlier".into(),
        reasoning: String::new(),
        continued: false,
    });
    ui.pending.push(Entry::Assistant {
        markdown: "```rs\nlet a = 1;\n```\n\n".into(),
        reasoning: String::new(),
        continued: false,
    });
    ui.pending.push(Entry::Notice {
        text: "help".into(),
    });
    ui.pending.push(Entry::Assistant {
        markdown: "tail".into(),
        reasoning: String::new(),
        continued: true,
    });
    assert_eq!(
        ui.last_reply_markdown().as_deref(),
        Some("```rs\nlet a = 1;\n```\n\ntail")
    );
}

#[test]
fn clicking_the_prompt_moves_the_cursor() {
    let mut ui = ui(80, 24);
    ui.input = "hello world".into();
    ui.cursor = ui.input.len();
    ui.region_top = Some(10);
    let input = ui.input_layout();
    let build = ui.build_region(&input);
    ui.input_span = build.input;
    let first = 10 + build.input.expect("span").first_row;
    // Gutter 2 + prefix 2, then column 6 of the text.
    assert_eq!(ui.click_to_cursor(4 + 6, first as u16), Some(6));
    assert_eq!(ui.click_to_cursor(4 + 6, (first + 3) as u16), None);
    ui.region_top = None;
    assert_eq!(ui.click_to_cursor(10, first as u16), None);
}

#[test]
fn region_top_follows_scrolling_growth() {
    let mut ui = ui(80, 10);
    ui.region_top = Some(9);
    ui.add_notice("scrolls the terminal");
    let input = ui.input_layout();
    let build = ui.build_region(&input);
    let rows = build.rows.len();
    ui.write_frame(build, false).expect("frame");
    assert_eq!(ui.region_top, Some(10 - rows));
    let input = ui.input_layout();
    let build = ui.build_region(&input);
    ui.write_frame(build, true).expect("frame");
    // Short history: notice, gap, then the one-row region.
    assert_eq!(ui.region_top, Some(2));
}

#[test]
fn prewalk_is_never_shown_armed_without_a_target() {
    let mut ui = ui(80, 24);
    ui.apply_event(UiEvent::PrewalkChanged {
        armed: true,
        target: None,
        reasoning: None,
        change: PrewalkChange::Session,
    });
    assert!(!ui.prewalk_armed);
}

#[test]
fn finalize_stream_commits_the_live_message() {
    let mut live = ui(80, 24);
    live.stream = Some(StreamState {
        reasoning: "thinking".into(),
        markdown: "answer".into(),
        ..StreamState::default()
    });
    live.finalize_stream();
    assert!(live.stream.is_none());
    match &live.pending[0] {
        Entry::Assistant {
            markdown,
            reasoning,
            ..
        } => {
            assert_eq!(markdown, "answer");
            assert_eq!(reasoning, "thinking");
        }
        other => panic!("expected assistant entry, got {other:?}"),
    }

    // An empty stream finalizes to nothing.
    let mut empty = ui(80, 24);
    empty.stream = Some(StreamState::default());
    empty.finalize_stream();
    assert!(empty.pending.is_empty());
}

#[test]
fn session_events_reset_the_transcript_and_queue_a_separator() {
    let mut loaded_ui = ui(80, 24);
    loaded_ui
        .transcript
        .push(Entry::User { text: "old".into() });
    loaded_ui.stream = Some(StreamState {
        reasoning: String::new(),
        markdown: "live".into(),
        ..StreamState::default()
    });
    loaded_ui.apply_event(UiEvent::SessionChanged {
        id: "abcd1234efgh".into(),
        title: None,
        loaded: true,
    });
    assert!(loaded_ui.transcript.is_empty());
    assert!(loaded_ui.stream.is_none());
    // The live message finalizes first, then the separator follows it.
    assert!(matches!(&loaded_ui.pending[0], Entry::Assistant { .. }));
    assert!(
        matches!(&loaded_ui.pending[1], Entry::Separator { label } if label == "Loaded Session abcd1234")
    );

    let mut startup = ui(80, 24);
    startup.apply_event(UiEvent::SessionChanged {
        id: "initial".into(),
        title: None,
        loaded: false,
    });
    assert!(matches!(&startup.pending[0], Entry::Separator { label } if label.is_empty()));
    startup.apply_event(UiEvent::SessionChanged {
        id: "next".into(),
        title: None,
        loaded: false,
    });
    assert!(matches!(&startup.pending[1], Entry::Separator { label } if label == "New Session"));

    let mut minimal = ui(80, 24);
    minimal.minimal = true;
    minimal.apply_event(UiEvent::SessionChanged {
        id: "initial".into(),
        title: None,
        loaded: false,
    });
    assert!(minimal.pending.is_empty());
}

#[test]
fn direct_provider_prefix_requests_its_model_catalogue() {
    let mut ui = ui(80, 24);
    ui.input = "/model openrouter:".into();
    ui.cursor = ui.input.len();
    ui.refresh_completion();
    let (tx, mut rx) = mpsc::unbounded_channel();
    ui.request_typed_backend(&tx, false).unwrap();
    assert_eq!(
        rx.try_recv().ok(),
        Some(InputMessage::ListModels {
            provider: "openrouter".into(),
        })
    );
    // A partially typed `provider:model` token addresses the same
    // catalogue even before the prefix is Tab-accepted.
    let mut ui2 = tests::ui(80, 24);
    ui2.input = "/model openrouter:anthropic/claude".into();
    ui2.cursor = ui2.input.len();
    ui2.refresh_completion();
    let (tx, mut rx) = mpsc::unbounded_channel();
    ui2.request_typed_backend(&tx, false).unwrap();
    assert_eq!(
        rx.try_recv().ok(),
        Some(InputMessage::ListModels {
            provider: "openrouter".into(),
        })
    );
}

#[test]
fn failed_sessions_command_does_not_print_a_later_completion_list() {
    let mut ui = ui(80, 24);
    let (tx, _rx) = mpsc::unbounded_channel();
    ui.submit_command("/sessions", &tx).unwrap();
    ui.apply_event(UiEvent::Error("could not list sessions: x".into()));
    ui.apply_event(UiEvent::SessionList {
        sessions: Vec::new(),
    });
    assert!(!ui.pending.iter().any(|entry| matches!(
        entry,
        Entry::Notice { text } if text == "No sessions for this workspace"
    )));
}

fn load_ui() -> CrossTerm {
    let mut ui = ui(80, 24);
    ui.input = "/load ".into();
    ui.cursor = ui.input.len();
    ui.refresh_completion();
    ui
}

#[tokio::test]
async fn failed_session_fetch_retries_on_tab_not_on_typing() {
    let mut ui = load_ui();
    let (tx, mut rx) = mpsc::unbounded_channel();
    ui.request_typed_backend(&tx, false).unwrap();
    assert_eq!(rx.try_recv().ok(), Some(InputMessage::ListSessions));
    ui.apply_event(UiEvent::Error("could not list sessions: x".into()));
    ui.request_typed_backend(&tx, false).unwrap();
    assert!(rx.try_recv().is_err());
    ui.handle_tab(&tx).unwrap();
    assert_eq!(rx.try_recv().ok(), Some(InputMessage::ListSessions));
}

#[tokio::test]
async fn empty_session_list_is_not_refetched_per_keystroke() {
    let mut ui = load_ui();
    let (tx, mut rx) = mpsc::unbounded_channel();
    ui.request_typed_backend(&tx, false).unwrap();
    assert_eq!(rx.try_recv().ok(), Some(InputMessage::ListSessions));
    ui.apply_event(UiEvent::SessionList {
        sessions: Vec::new(),
    });
    ui.request_typed_backend(&tx, false).unwrap();
    assert!(rx.try_recv().is_err());
}

#[test]
fn failed_model_fetch_retries_on_tab_not_on_typing() {
    let mut ui = ui(80, 24);
    ui.input = "/model openrouter:".into();
    ui.cursor = ui.input.len();
    ui.refresh_completion();
    let (tx, mut rx) = mpsc::unbounded_channel();
    let request = InputMessage::ListModels {
        provider: "openrouter".into(),
    };
    ui.request_typed_backend(&tx, false).unwrap();
    assert_eq!(rx.try_recv().ok(), Some(request.clone()));
    // Still in flight: no duplicate request.
    ui.request_typed_backend(&tx, false).unwrap();
    assert!(rx.try_recv().is_err());
    ui.apply_event(UiEvent::BackgroundOperationFinished {
        id: 1,
        error: Some("model list failed".into()),
    });
    ui.request_typed_backend(&tx, false).unwrap();
    assert!(rx.try_recv().is_err());
    ui.handle_tab(&tx).unwrap();
    assert_eq!(rx.try_recv().ok(), Some(request));
}

#[tokio::test]
async fn load_completion_combines_session_and_filesystem_context() {
    let mut ui = ui(80, 24);
    ui.input = "/load ./".into();
    ui.cursor = ui.input.len();
    ui.refresh_completion();
    let completion = ui.completion.as_ref().expect("load completion");
    assert_eq!(completion.kind, CompletionKind::Path);
    assert_eq!(
        completion.context.target,
        CompletionTarget::Argument(ArgumentKind::Session)
    );
    assert_eq!(completion.context.query, "./");
    // The debounced scan merges filesystem candidates with the static
    // session-ID candidates (TUI-3: `/load` completes both).
    let scanned =
        crate::paths::find_path_candidates(&ui.environment.cwd, "./", &CancellationToken::new());
    let merged = merge_candidates(
        commands::candidates_at_cursor(
            &ui.input,
            ui.cursor_char_col(),
            &ui.providers,
            &ui.model_lists,
            &ui.provider,
            &ui.session_candidates,
            &ui.skills,
        )
        .map(|result| result.candidates)
        .unwrap_or_default(),
        scanned,
    );
    assert!(
        !merged.is_empty(),
        "`/load ./...` must produce filesystem candidates"
    );
    // And session IDs survive the merge when present: an empty query
    // (`/load ` with no prefix) matches every cached session.
    ui.input = "/load ".into();
    ui.cursor = ui.input.len();
    ui.session_candidates = vec![SessionListEntry {
        id: "abc123-full".into(),
        short_id: "abc123".into(),
        title: None,
        updated_at: "2026-08-13 12:00".into(),
        workspace: "/workspace".into(),
        provider: None,
        model: None,
    }];
    let merged_with_session = merge_candidates(
        commands::candidates_at_cursor(
            &ui.input,
            ui.cursor_char_col(),
            &ui.providers,
            &ui.model_lists,
            &ui.provider,
            &ui.session_candidates,
            &ui.skills,
        )
        .map(|result| result.candidates)
        .unwrap_or_default(),
        Vec::new(),
    );
    assert!(
        merged_with_session
            .iter()
            .any(|candidate| candidate.value.contains("abc123")),
        "session IDs must survive alongside path candidates"
    );
}

#[test]
fn completion_tab_accepts_the_best_ranked_provider() {
    let mut ui = ui(80, 24);
    ui.input = "/model open".to_owned();
    ui.cursor = ui.input.len();
    let (tx, mut rx) = mpsc::unbounded_channel::<InputMessage>();
    // Provider completion is fuzzy-ranked; Tab accepts the best match.
    ui.handle_tab(&tx).unwrap();
    assert_eq!(ui.input, "/model openrouter:");
    // Accepting the provider requested its model list on demand.
    assert_eq!(
        rx.try_recv().ok(),
        Some(InputMessage::ListModels {
            provider: "openrouter".into(),
        })
    );
}

#[test]
fn fuzzy_model_tab_accepts_a_provider_qualified_match() {
    let mut ui = ui(80, 24);
    ui.model_lists.insert(
        "openrouter".into(),
        vec![ModelEntry {
            id: "anthropic/claude-sonnet-4".into(),
            name: Some("Claude Sonnet 4".into()),
            context_length: Some(200_000),
        }],
    );
    ui.input = "/model cs4".to_owned();
    ui.cursor = ui.input.len();
    let (tx, _) = mpsc::unbounded_channel::<InputMessage>();
    ui.handle_tab(&tx).unwrap();
    assert_eq!(ui.input, "/model openrouter:anthropic/claude-sonnet-4");
}

#[test]
fn ghost_previews_a_single_candidate_suffix() {
    let mut ui = ui(80, 24);
    ui.input = "/model openc".to_owned();
    ui.cursor = ui.input.len();
    ui.refresh_completion();
    // Only `opencode-go:` matches "openc", so the ghost shows its untyped
    // remainder.
    assert_eq!(ui.ghost_text(), "ode-go:");
}

#[tokio::test]
async fn at_reference_opens_a_file_completion_context() {
    let mut ui = ui(80, 24);
    ui.input = "see @src/ma".to_owned();
    ui.cursor = ui.input.len();
    ui.refresh_completion();
    // The `@` token opens a path completion even though the draft is not
    // a slash command; the scan starts from the token's query.
    let completion = ui.completion.as_ref().expect("@ opens completion");
    assert_eq!(completion.kind, CompletionKind::Path);
    assert_eq!(completion.context.token_start, 4);
    assert_eq!(completion.context.query, "src/ma");
    assert_eq!(ui.path_completion_query.as_deref(), Some("src/ma"));

    // Moving the cursor off the token (e.g. after a space) closes it.
    ui.input.push(' ');
    ui.cursor = ui.input.len();
    ui.refresh_completion();
    assert!(ui.completion.is_none());
}

#[tokio::test]
async fn at_reference_scan_results_replace_the_candidate_list() {
    let mut ui = ui(80, 24);
    ui.input = "look at @Re".to_owned();
    ui.cursor = ui.input.len();
    ui.refresh_completion();
    let generation = ui.path_completion_generation;
    let context = ui.completion.as_ref().unwrap().context.clone();
    ui.apply_path_completion(PathCompletionResult {
        generation,
        context,
        candidates: vec![Candidate {
            value: "@src/README.md".into(),
            description: "file".into(),
            kind: CandidateKind::File,
        }],
        at_prefix: paths::extract_at_prefix("look at @Re", 11),
    });
    let completion = ui.completion.as_ref().expect("candidates applied");
    assert_eq!(completion.candidates[0].value, "@src/README.md");
    // A fuzzy match (`Re` → README) shares no literal prefix with the
    // typed token, so there is no ghost to preview.
    assert_eq!(ui.ghost_text(), "");
    // Tab accepts the candidate and appends a trailing space.
    let (tx, _) = mpsc::unbounded_channel::<InputMessage>();
    ui.handle_tab(&tx).unwrap();
    assert_eq!(ui.input, "look at @src/README.md ");
}

#[tokio::test]
async fn tab_accepts_scanned_export_and_load_paths() {
    for command in ["/export ", "/load "] {
        let mut ui = ui(80, 24);
        ui.input = format!("{command}notes");
        ui.cursor = ui.input.len();
        ui.refresh_completion();
        let generation = ui.path_completion_generation;
        let context = ui
            .completion
            .as_ref()
            .expect("path scan started")
            .context
            .clone();
        ui.apply_path_completion(PathCompletionResult {
            generation,
            context,
            candidates: vec![Candidate {
                value: "notes/today.md".into(),
                description: "file".into(),
                kind: CandidateKind::File,
            }],
            at_prefix: None,
        });
        // Tab refreshes completion first; unchanged context must keep the
        // scanned result instead of clearing it.
        let (tx, _rx) = mpsc::unbounded_channel::<InputMessage>();
        ui.handle_tab(&tx).unwrap();
        assert!(
            ui.input.starts_with(&format!("{command}notes/today.md")),
            "{}",
            ui.input
        );
    }
}

#[tokio::test]
async fn tab_accepts_the_top_ranked_at_candidate_then_cycles() {
    let mut ui = ui(80, 24);
    ui.input = "look at @REA".to_owned();
    ui.cursor = ui.input.len();
    ui.refresh_completion();
    let generation = ui.path_completion_generation;
    let context = ui.completion.as_ref().unwrap().context.clone();
    // Ranking put the root README first (see the paths tests).
    ui.apply_path_completion(PathCompletionResult {
        generation,
        context,
        candidates: vec![
            Candidate {
                value: "@README.md".into(),
                description: "file".into(),
                kind: CandidateKind::File,
            },
            Candidate {
                value: "@crates/agent/README.md".into(),
                description: "file".into(),
                kind: CandidateKind::File,
            },
        ],
        at_prefix: paths::extract_at_prefix("look at @REA", 11),
    });

    // The ghost previews exactly what Tab will insert.
    assert_eq!(ui.ghost_text(), "DME.md");

    // First Tab accepts the top-ranked candidate; the trailing space ends
    // the token so typing continues on a fresh one.
    let (tx, _) = mpsc::unbounded_channel::<InputMessage>();
    ui.handle_tab(&tx).unwrap();
    assert_eq!(ui.input, "look at @README.md ");
}

#[tokio::test]
async fn fuzzy_at_tab_accepts_the_ranked_match_and_a_repeat_cycles() {
    let mut ui = ui(80, 24);
    ui.input = "@agent.rs".to_owned();
    ui.cursor = ui.input.len();
    ui.refresh_completion();
    let generation = ui.path_completion_generation;
    let context = ui.completion.as_ref().unwrap().context.clone();
    ui.apply_path_completion(PathCompletionResult {
        generation,
        context,
        candidates: vec![
            Candidate {
                value: "@crates/agent/src/agent.rs".into(),
                description: "file".into(),
                kind: CandidateKind::File,
            },
            Candidate {
                value: "@crates/tui/src/app.rs".into(),
                description: "file".into(),
                kind: CandidateKind::File,
            },
        ],
        at_prefix: paths::extract_at_prefix("@agent.rs", 9),
    });

    // No literal extension of `@agent.rs` exists, so nothing is dimmed
    // inline — but Tab still completes to the ranked match.
    assert_eq!(ui.ghost_text(), "");
    let (tx, _) = mpsc::unbounded_channel::<InputMessage>();
    ui.handle_tab(&tx).unwrap();
    assert_eq!(ui.input, "@crates/agent/src/agent.rs ");

    // The accepted file closes the list; re-opening on the same token and
    // pressing Tab again cycles to the next candidate.
    ui.cursor = "@crates/agent/src/agent.rs".len();
    ui.refresh_completion();
    let generation = ui.path_completion_generation;
    let context = ui.completion.as_ref().unwrap().context.clone();
    ui.apply_path_completion(PathCompletionResult {
        generation,
        context,
        candidates: vec![
            Candidate {
                value: "@crates/agent/src/agent.rs".into(),
                description: "file".into(),
                kind: CandidateKind::File,
            },
            Candidate {
                value: "@crates/tui/src/app.rs".into(),
                description: "file".into(),
                kind: CandidateKind::File,
            },
        ],
        at_prefix: paths::extract_at_prefix("@crates/agent/src/agent.rs", 26),
    });
    ui.handle_tab(&tx).unwrap();
    assert_eq!(ui.input, "@crates/tui/src/app.rs ");
}

#[tokio::test]
async fn enter_still_submits_while_an_at_list_is_open() {
    let mut ui = ui(80, 24);
    ui.input = "look at @Re".to_owned();
    ui.cursor = ui.input.len();
    ui.refresh_completion();
    let generation = ui.path_completion_generation;
    let context = ui.completion.as_ref().unwrap().context.clone();
    ui.apply_path_completion(PathCompletionResult {
        generation,
        context,
        candidates: vec![Candidate {
            value: "@README.md".into(),
            description: "file".into(),
            kind: CandidateKind::File,
        }],
        at_prefix: paths::extract_at_prefix("look at @Re", 11),
    });

    // Enter is submit-only: completion stays a Tab affair.
    let (tx, mut rx) = mpsc::unbounded_channel::<InputMessage>();
    let cancel = CancellationToken::new();
    let enter = Event::Key(KeyEvent::new(KeyCode::Enter, KeyModifiers::empty()));
    ui.handle_input(&enter, &tx, &cancel).unwrap();
    assert_eq!(
        rx.try_recv().ok(),
        Some(InputMessage::Message("look at @Re".into()))
    );
}

#[tokio::test]
async fn refining_an_at_query_keeps_the_previous_candidates_visible() {
    let mut ui = ui(80, 24);
    ui.input = "look at @Re".to_owned();
    ui.cursor = ui.input.len();
    ui.refresh_completion();
    let generation = ui.path_completion_generation;
    let context = ui.completion.as_ref().unwrap().context.clone();
    ui.apply_path_completion(PathCompletionResult {
        generation,
        context,
        candidates: vec![Candidate {
            value: "@README.md".into(),
            description: "file".into(),
            kind: CandidateKind::File,
        }],
        at_prefix: paths::extract_at_prefix("look at @Re", 11),
    });

    // Keep typing: the refined scan is still in flight, but the open
    // list must not blank out while waiting for it.
    ui.input.push('a');
    ui.cursor = ui.input.len();
    ui.refresh_completion();
    let completion = ui.completion.as_ref().expect("list stays open");
    assert_eq!(completion.candidates.len(), 1);
    assert_eq!(completion.candidates[0].value, "@README.md");
    assert_eq!(ui.path_completion_query.as_deref(), Some("Rea"));
}

#[tokio::test]
async fn stale_at_scan_results_are_ignored() {
    let mut ui = ui(80, 24);
    ui.input = "@a".to_owned();
    ui.cursor = ui.input.len();
    ui.refresh_completion();
    let generation = ui.path_completion_generation;
    let context = ui.completion.as_ref().unwrap().context.clone();
    // The user kept typing before the scan landed: the result is stale.
    ui.input = "@ab".to_owned();
    ui.cursor = ui.input.len();
    ui.apply_path_completion(PathCompletionResult {
        generation,
        context,
        candidates: vec![Candidate {
            value: "@a.txt".into(),
            description: "file".into(),
            kind: CandidateKind::File,
        }],
        at_prefix: paths::extract_at_prefix("@a", 2),
    });
    let completion = ui.completion.as_ref().expect("still open for @ab");
    assert!(completion.candidates.is_empty());
}

#[test]
fn completion_hint_is_always_one_bounded_visual_row() {
    let candidates = vec![
        Candidate {
            value: "openrouter:anthropic/claude-sonnet-4-very-long-name".into(),
            description: String::new(),
            kind: CandidateKind::Slash,
        },
        Candidate {
            value: "github-copilot:claude-sonnet-4".into(),
            description: String::new(),
            kind: CandidateKind::Slash,
        },
    ];
    let hint = fit_completion_hint(&candidates, 0, 24);
    assert!(UnicodeWidthStr::width(hint.as_str()) <= 24);
    assert!(hint.ends_with('…'));
    assert!(!hint.contains(['\n', '\r']));

    // The selected candidate leads even when it is not index zero.
    let hint = fit_completion_hint(&candidates, 1, 80);
    assert!(hint.starts_with("github-copilot:"));
}

#[test]
fn input_layout_renders_ghost_and_hint_rows() {
    // Ghost suffix follows the typed text on the input row.
    let layout = input_layout("/model openc", 12, 60, Theme::default(), "", "ode-go:", "");
    assert_eq!(row_text(&layout.rows[0]), "› /model opencode-go:");
    assert_eq!(layout.cursor_row, 0);

    // The completion hint row sits below the input without moving it.
    let layout = input_layout(
        "/model",
        6,
        60,
        Theme::default(),
        "",
        "",
        "opencode-go: · openrouter:",
    );
    assert_eq!(row_text(&layout.rows[0]), "› /model");
    assert_eq!(row_text(&layout.rows[1]), "  opencode-go: · openrouter:");
    assert_eq!(layout.cursor_row, 0);
}
