
use super::*;
use crate::{ContextFileEntry, SkillEntry};
fn pane(path: &str) -> AgentPane {
    AgentPane::new(
        "m",
        "p",
        vec![],
        Vec::<SkillEntry>::new(),
        Vec::<ContextFileEntry>::new(),
        "auto",
        path.into(),
    )
}
fn key(c: char) -> Event {
    Event::Key(KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE))
}
fn strip_ansi(value: &str) -> String {
    let mut plain = String::new();
    let mut chars = value.chars();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            for code in chars.by_ref() {
                if code.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            plain.push(c);
        }
    }
    plain
}
fn add_slots(mux: &mut MuxUi, count: u64) {
    for id in 1..=count {
        mux.apply(MuxEvent::Add {
            id,
            name: format!("agent-{id}"),
            workspace: PathBuf::from(format!("/agent-{id}")),
            worktree: false,
            status: MuxStatus::Idle,
            pane: pane("/"),
        });
    }
}
#[test]
fn provisional_add_activates_but_replacement_preserves_newer_selection() {
    let mut mux = MuxUi::new("/x".into());
    mux.apply(MuxEvent::Add {
        id: 1,
        name: "first".into(),
        workspace: "/pending".into(),
        worktree: true,
        status: MuxStatus::Starting,
        pane: pane("/pending"),
    });
    assert_eq!(mux.selected_id(), Some(1));
    mux.apply(MuxEvent::Add {
        id: 2,
        name: "second".into(),
        workspace: "/second".into(),
        worktree: false,
        status: MuxStatus::Starting,
        pane: pane("/second"),
    });
    assert_eq!(mux.selected_id(), Some(2));

    mux.apply(MuxEvent::Replace {
        id: 1,
        name: "ready".into(),
        workspace: "/first".into(),
        worktree: true,
        status: MuxStatus::Starting,
        pane: pane("/first"),
    });

    assert_eq!(mux.selected_id(), Some(2));
    assert_eq!(mux.slots[0].name, "ready");
    assert_eq!(mux.slots[0].workspace, PathBuf::from("/first"));
}

#[test]
fn replacement_preserves_the_provisional_draft_cursor() {
    let mut mux = MuxUi::new("/x".into());
    mux.apply(MuxEvent::Add {
        id: 1,
        name: "pending".into(),
        workspace: "/pending".into(),
        worktree: false,
        status: MuxStatus::Idle,
        pane: pane("/pending"),
    });
    mux.slots[0].pane.set_draft("abc");
    mux.handle(Event::Key(KeyEvent::new(KeyCode::Left, KeyModifiers::NONE)))
        .unwrap();

    mux.apply(MuxEvent::Replace {
        id: 1,
        name: "ready".into(),
        workspace: "/ready".into(),
        worktree: false,
        status: MuxStatus::Idle,
        pane: pane("/ready"),
    });
    mux.handle(key('X')).unwrap();

    assert_eq!(mux.slots[0].pane.editor_text(), "abXc");
}

#[test]
fn routes_by_stable_id_and_retains_drafts() {
    let mut m = MuxUi::new("/x".into());
    m.apply(MuxEvent::Add {
        id: 9,
        name: "a".into(),
        workspace: "/a".into(),
        worktree: false,
        status: MuxStatus::Idle,
        pane: pane("/a"),
    });
    m.apply(MuxEvent::Add {
        id: 4,
        name: "b".into(),
        workspace: "/b".into(),
        worktree: false,
        status: MuxStatus::Idle,
        pane: pane("/b"),
    });
    assert!(m.handle(key('z')).unwrap().is_empty());
    assert_eq!(m.slots[1].pane.editor_text(), "z");
    m.handle(Event::Key(KeyEvent::new(
        KeyCode::Null,
        KeyModifiers::CONTROL,
    )))
    .unwrap();
    m.handle(key('k')).unwrap();
    assert_eq!(m.selected_id(), Some(9));
    assert_eq!(m.slots[1].pane.editor_text(), "z");
    m.handle(Event::Key(KeyEvent::new(
        KeyCode::Null,
        KeyModifiers::CONTROL,
    )))
    .unwrap();
    m.handle(key('j')).unwrap();
    let actions = m
        .handle(Event::Key(KeyEvent::new(
            KeyCode::Enter,
            KeyModifiers::NONE,
        )))
        .unwrap();
    assert!(matches!(
        &actions[0],
        MuxAction::AgentInput { id: 4, input: InputMessage::Message(text) } if text == "z"
    ));
}
#[test]
fn prefix_and_overlays() {
    let mut m = MuxUi::new("/x".into());
    m.handle(Event::Key(KeyEvent::new(
        KeyCode::Char(' '),
        KeyModifiers::CONTROL,
    )))
    .unwrap();
    m.handle(key('n')).unwrap();
    assert!(matches!(m.overlay, Some(Overlay::NewMenu { .. })));
    m.handle(Event::Key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE)))
        .unwrap();
    assert!(m.overlay.is_none())
}
#[test]
fn duplicate_direct_workspace_requires_confirmation_and_keeps_inheritance() {
    let cwd = std::fs::canonicalize(".").unwrap();
    let mut mux = MuxUi::new(cwd.clone());
    mux.apply(MuxEvent::Add {
        id: 12,
        name: "one".into(),
        workspace: cwd.clone(),
        worktree: false,
        status: MuxStatus::Idle,
        pane: pane(cwd.to_str().unwrap()),
    });
    let mut actions = vec![];
    mux.request_create(
        WorkspaceChoice::Directory {
            path: cwd,
            name: "two".into(),
        },
        &mut actions,
    );
    assert!(actions.is_empty());
    assert!(matches!(
        mux.overlay,
        Some(Overlay::ConfirmDuplicate { .. })
    ));
    let actions = mux
        .handle(Event::Key(KeyEvent::new(
            KeyCode::Enter,
            KeyModifiers::NONE,
        )))
        .unwrap();
    assert!(matches!(
        actions.as_slice(),
        [MuxAction::Create {
            inherit_from: Some(12),
            ..
        }]
    ));
}

#[test]
fn worktree_tab_toggles_keep_without_closing_editor() {
    let mut mux = MuxUi::new("/x".into());
    mux.overlay = Some(Overlay::Worktree {
        branch: "feature".into(),
        keep: true,
    });

    mux.handle(Event::Key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE)))
        .unwrap();

    assert!(matches!(
        mux.overlay,
        Some(Overlay::Worktree { keep: false, .. })
    ));
}

#[test]
fn paste_cancels_the_prefix_and_reaches_the_selected_pane() {
    let mut mux = MuxUi::new("/x".into());
    add_slots(&mut mux, 1);
    mux.prefix = true;

    mux.handle(Event::Paste("pasted".into())).unwrap();

    assert!(!mux.prefix);
    assert_eq!(mux.slots[0].pane.editor_text(), "pasted");
}

#[test]
fn paste_populates_text_overlays_and_refreshes_directory_completion() {
    let mut mux = MuxUi::new("/x".into());
    mux.overlay = Some(Overlay::Worktree {
        branch: String::new(),
        keep: true,
    });
    mux.handle(Event::Paste("feat/pasted\n".into())).unwrap();
    assert!(matches!(
        mux.overlay,
        Some(Overlay::Worktree { ref branch, .. }) if branch == "feat/pasted"
    ));

    mux.overlay = Some(Overlay::Directory {
        input: String::new(),
        suggestions: vec!["/stale".into()],
        selected: 0,
    });
    mux.handle(Event::Paste("/tmp/project".into())).unwrap();
    assert!(matches!(
        mux.overlay,
        Some(Overlay::Directory { ref input, ref suggestions, .. })
            if input == "/tmp/project" && suggestions.is_empty()
    ));
    assert_eq!(
        mux.directory_request
            .as_ref()
            .map(|request| request.input.as_str()),
        Some("/tmp/project")
    );
}

#[test]
fn directory_enter_prefers_an_exact_directory_over_its_suggestions() {
    let root = tempfile::tempdir().unwrap();
    let typed = root.path().join("typed");
    let child = typed.join("child");
    std::fs::create_dir_all(&child).unwrap();
    let mut mux = MuxUi::new(root.path().to_path_buf());
    mux.overlay = Some(Overlay::Directory {
        input: typed.display().to_string(),
        suggestions: vec![child],
        selected: 0,
    });

    let actions = mux
        .handle(Event::Key(KeyEvent::new(
            KeyCode::Enter,
            KeyModifiers::NONE,
        )))
        .unwrap();

    assert!(matches!(
        actions.as_slice(),
        [MuxAction::Create {
            choice: WorkspaceChoice::Directory { path, .. },
            ..
        }] if path == &typed
    ));
}

#[test]
fn directory_enter_honors_an_explicit_suggestion_selection() {
    let root = tempfile::tempdir().unwrap();
    let typed = root.path().join("typed");
    let child = typed.join("child");
    std::fs::create_dir_all(&child).unwrap();
    let mut mux = MuxUi::new(root.path().to_path_buf());
    mux.overlay = Some(Overlay::Directory {
        input: typed.display().to_string(),
        suggestions: vec![child.clone()],
        selected: 0,
    });
    mux.handle(Event::Key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE)))
        .unwrap();

    let actions = mux
        .handle(Event::Key(KeyEvent::new(
            KeyCode::Enter,
            KeyModifiers::NONE,
        )))
        .unwrap();

    assert!(matches!(
        actions.as_slice(),
        [MuxAction::Create {
            choice: WorkspaceChoice::Directory { path, .. },
            ..
        }] if path == &child
    ));
}

#[test]
fn directory_completion_is_debounced_and_stale_results_are_ignored() {
    let mut mux = MuxUi::new("/x".into());
    mux.overlay = Some(Overlay::Directory {
        input: String::new(),
        suggestions: vec![],
        selected: 0,
    });
    mux.handle(key('a')).unwrap();
    let first = mux.directory_request.take().unwrap();
    mux.handle(key('b')).unwrap();
    let second = mux.directory_request.as_ref().unwrap();
    assert!(second.generation > first.generation);
    assert_eq!(second.input, "ab");
    let generation = second.generation;

    assert!(!mux.apply_directory_completion(DirectoryCompletion {
        generation: first.generation,
        input: first.input,
        suggestions: vec!["/stale".into()],
    }));
    let Overlay::Directory { suggestions, .. } = mux.overlay.as_ref().unwrap() else {
        panic!("directory overlay closed unexpectedly");
    };
    assert!(suggestions.is_empty());

    assert!(mux.apply_directory_completion(DirectoryCompletion {
        generation,
        input: "ab".into(),
        suggestions: vec!["/fresh".into()],
    }));
    mux.handle(Event::Key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE)))
        .unwrap();
    assert!(!mux.apply_directory_completion(DirectoryCompletion {
        generation,
        input: "ab".into(),
        suggestions: vec!["/late".into()],
    }));
}

#[test]
fn layout_gives_both_panes_the_full_terminal_height() {
    let m = MuxUi::new("/x".into());
    assert!(m.layout(29, 10).sidebar.is_none());
    assert!(m.layout(41, 10).sidebar.is_some());
    let layout = m.layout(80, 20);
    assert!(layout.sidebar.is_some());
    assert_eq!(layout.body_height, 20);
    assert_eq!(layout.main_x + layout.main_width, 80);
}

#[test]
fn rendered_sidebar_boundary_spans_every_screen_row_for_any_session_count() {
    let theme = render::Theme::default();
    let muted = Style::default().fg(theme.muted_text);
    let accent = Style::default().fg(theme.accent);

    for count in [1, 6] {
        let mut mux = MuxUi::new("/".into());
        mux.handle(Event::Resize(80, 12)).unwrap();
        add_slots(&mut mux, count);
        let layout = mux.layout(80, 12);
        let (_, sidebar_width) = layout.sidebar.unwrap();
        let sidebar = mux.sidebar_lines(sidebar_width, 12, muted, accent);
        assert_eq!(sidebar.len(), 12);

        for (y, row) in sidebar.iter().enumerate() {
            let plain = strip_ansi(&line_to_ansi(row));
            assert_eq!(plain.width(), sidebar_width as usize + 1, "sidebar row {y}");
            let boundary = plain.chars().last().unwrap();
            assert!(matches!(boundary, '╮' | '│' | '┤' | '╯'), "sidebar row {y}");

            // Model final composition with a visible main-pane first cell.
            let screen = format!(
                "{plain}  A{}",
                " ".repeat(layout.main_width.saturating_sub(1) as usize)
            );
            assert_eq!(screen.width(), 80, "screen row {y}");
            assert_eq!(screen.chars().nth(layout.main_x as usize), Some('A'));
        }
        assert!(strip_ansi(&line_to_ansi(&sidebar[0])).starts_with("╭─ SESSIONS "));
        assert!(strip_ansi(&line_to_ansi(sidebar.last().unwrap())).starts_with('╰'));
    }
}

#[test]
fn normal_sidebar_uses_compact_ordinals_spacers_and_anchored_footer() {
    let mut mux = MuxUi::new("/".into());
    mux.handle(Event::Resize(80, 14)).unwrap();
    add_slots(&mut mux, 10);
    mux.selected = Some(0);
    mux.sidebar_scroll = 0;
    let lines = mux.sidebar_lines(26, 14, Style::default(), Style::default());
    let plain = lines
        .iter()
        .map(|line| strip_ansi(&line_to_ansi(line)))
        .collect::<Vec<_>>();

    assert!(plain[0].starts_with("╭─ SESSIONS "));
    assert!(plain[1].trim_matches(['│', ' ']).is_empty());
    assert!(plain[2].contains("› 1 agent-1 ✓"));
    assert!(plain[3].contains("  2 agent-2 ✓"));
    assert!(plain[7].contains("  6 agent-6 ✓"));
    assert!(plain[8].trim_matches(['│', ' ']).is_empty());
    assert!(plain[9].contains("+ New agent"));
    assert!(plain[10].trim_matches(['│', ' ']).is_empty());
    assert_eq!(plain[11], format!("├{}┤", "─".repeat(25)));
    assert!(plain[12].contains("^Space  mux"));
    assert_eq!(plain[13], format!("╰{}╯", "─".repeat(25)));
    assert_eq!(session_ordinal(8), "9");
    assert_eq!(session_ordinal(9), "10");
}

#[test]
fn sidebar_borders_are_not_session_hit_targets() {
    let mut mux = MuxUi::new("/".into());
    mux.width = 80;
    mux.height = 14;
    add_slots(&mut mux, 2);
    let sidebar_width = mux.layout(mux.width, mux.height).sidebar.unwrap().1;
    assert_eq!(mux.selected_id(), Some(2));

    mux.handle_mouse(MouseEventKind::Down(MouseButton::Left), 0, 2);
    mux.handle_mouse(MouseEventKind::Down(MouseButton::Left), sidebar_width, 2);

    assert_eq!(mux.selected_id(), Some(2));
}

#[test]
fn hidden_new_row_has_no_mouse_hit_target_in_tiny_prefix_layout() {
    let mut mux = MuxUi::new("/".into());
    mux.width = 80;
    mux.height = 8;
    mux.prefix = true;

    mux.handle_mouse(MouseEventKind::Down(MouseButton::Left), 1, 3);

    assert!(mux.overlay.is_none());
}

#[test]
fn sidebar_rows_handle_tiny_heights_without_overflow() {
    let mux = MuxUi::new("/".into());
    let style = Style::default();
    for height in 0..=2 {
        for sidebar_width in 0..=20 {
            let rows = mux.sidebar_lines(sidebar_width, height, style, style);
            assert_eq!(rows.len(), height as usize);
            assert!(
                rows.iter()
                    .all(|row| row.width() == sidebar_width as usize + 1)
            );
        }
    }
}

#[test]
fn buffer_rows_emit_wide_graphemes_once() {
    let mut buffer = Buffer::empty(Rect::new(0, 0, 4, 1));
    buffer.set_string(0, 0, "界ab", Style::default());

    let row = buffer_row(&buffer, 0, 4);
    let text = row
        .spans
        .iter()
        .map(|span| span.content.as_ref())
        .collect::<String>();
    assert_eq!(text, "界ab");
    assert_eq!(text.width(), 4);
}

#[test]
fn tiny_layout_overlay_and_cursor_are_bounded() {
    let m = MuxUi::new("/x".into());
    for height in 0..=2 {
        for width in 0..=1 {
            let layout = m.layout(width, height);
            assert!(layout.main_x <= width);
            assert!(layout.main_width <= width);
            assert!(layout.body_height <= height);
            let frame = overlay_frame(layout, &Overlay::Help);
            assert!(frame.x.saturating_add(frame.width) <= width);
            assert!(frame.y.saturating_add(frame.rows.len() as u16) <= layout.body_height);
        }
    }
    assert_eq!(bounded_cursor((9, 9), 0, 2), None);
    assert_eq!(bounded_cursor((9, 9), 1, 1), Some((0, 0)));
    assert_eq!(bounded_cursor((9, 9), 2, 2), Some((1, 1)));
}

fn visible_buffer(buffer: &Buffer) -> String {
    (0..buffer.area.height)
        .map(|y| {
            (0..buffer.area.width)
                .map(|x| buffer[(x, y)].symbol())
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn composed_new_agent_frame_is_unique_selectable_and_activates_selection() {
    let mut mux = MuxUi::new("/workspace".into());
    mux.handle(Event::Key(KeyEvent::new(
        KeyCode::Char(' '),
        KeyModifiers::CONTROL,
    )))
    .unwrap();
    mux.handle(key('n')).unwrap();

    let (initial, _) = mux.compose_frame(100, 30);
    let initial_text = visible_buffer(&initial);
    for option in ["New worktree", "Current directory", "Another directory"] {
        assert_eq!(
            initial_text.matches(option).count(),
            1,
            "{option}: {initial_text}"
        );
    }
    assert!(initial_text.contains("› New worktree"));
    assert!(!initial_text.contains("› Current directory"));

    mux.handle(Event::Key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE)))
        .unwrap();
    let (selected, _) = mux.compose_frame(100, 30);
    let selected_text = visible_buffer(&selected);
    assert!(!selected_text.contains("› New worktree"));
    assert!(
        selected_text.contains("› Current directory"),
        "{selected_text}"
    );

    mux.handle(Event::Key(KeyEvent::new(
        KeyCode::Enter,
        KeyModifiers::NONE,
    )))
    .unwrap();
    assert!(matches!(mux.overlay, Some(Overlay::Current { .. })));
    let (current, cursor) = mux.compose_frame(100, 30);
    let current_text = visible_buffer(&current);
    assert!(current_text.contains("Session name:"));
    assert!(!current_text.contains("New worktree"));
    assert!(cursor.is_some());

    mux.overlay = Some(Overlay::NewMenu { selected: 2 });
    mux.handle(Event::Key(KeyEvent::new(
        KeyCode::Enter,
        KeyModifiers::NONE,
    )))
    .unwrap();
    assert!(matches!(mux.overlay, Some(Overlay::Directory { .. })));
    let (directory, cursor) = mux.compose_frame(100, 30);
    let directory_text = visible_buffer(&directory);
    assert!(directory_text.contains("Directory:"));
    assert!(!directory_text.contains("Current directory"));
    assert!(cursor.is_some());
}

#[test]
fn overlays_are_centered_in_main_pane_closed_and_opaque() {
    let mux = MuxUi::new("/x".into());
    let layout = mux.layout(100, 30);
    let first = overlay_frame(layout, &Overlay::NewMenu { selected: 0 });
    assert!(first.x >= layout.main_x);
    assert_eq!(
        first.x - layout.main_x,
        (layout.main_width - first.width) / 2
    );
    assert!(first.rows[0].starts_with("╭─ New agent "));
    assert!(first.rows[0].ends_with('╮'));
    assert!(
        first.rows[0]
            .trim_matches(['╭', '╮', '─'])
            .contains("New agent")
    );
    assert!(
        first.rows.last().unwrap().starts_with('╰')
            && first.rows.last().unwrap().ends_with('╯')
            && first
                .rows
                .last()
                .unwrap()
                .chars()
                .skip(1)
                .take(first.width.saturating_sub(2) as usize)
                .all(|character| character == '─')
    );
    assert!(
        first
            .rows
            .iter()
            .all(|row| row.width() == first.width as usize)
    );
    assert!(
        first.rows[1..first.rows.len() - 1]
            .iter()
            .all(|row| row.starts_with('│') && row.ends_with('│'))
    );

    // A subsequent, shorter overlay still paints every cell in its rectangle;
    // no text or divider from the New Agent menu can survive underneath it.
    let second = overlay_frame(layout, &Overlay::Close);
    assert!(
        second
            .rows
            .iter()
            .all(|row| row.width() == second.width as usize)
    );
    assert!(!second.rows.join("\n").contains("New worktree"));
}

#[test]
fn editable_overlay_cursor_tracks_unicode_and_edits() {
    let mut mux = MuxUi::new("/x".into());
    mux.handle(Event::Resize(100, 30)).unwrap();
    mux.handle(Event::Key(KeyEvent::new(
        KeyCode::Null,
        KeyModifiers::CONTROL,
    )))
    .unwrap();
    mux.handle(key('n')).unwrap();
    mux.handle(Event::Key(KeyEvent::new(
        KeyCode::Enter,
        KeyModifiers::NONE,
    )))
    .unwrap();

    let layout = mux.layout(100, 30);
    let Overlay::Worktree { .. } = mux.overlay.as_ref().unwrap() else {
        panic!("new menu did not open worktree editor");
    };
    let empty = overlay_frame(layout, mux.overlay.as_ref().unwrap());
    let empty_cursor = overlay_cursor(&empty, mux.overlay.as_ref().unwrap()).unwrap();
    assert_eq!(
        empty_cursor,
        (empty.x + 2 + "Branch / name: ".width() as u16, empty.y + 2)
    );

    mux.handle(key('界')).unwrap();
    let typed = overlay_frame(layout, mux.overlay.as_ref().unwrap());
    assert_eq!(
        overlay_cursor(&typed, mux.overlay.as_ref().unwrap())
            .unwrap()
            .0,
        empty_cursor.0 + 2
    );
    mux.handle(Event::Key(KeyEvent::new(
        KeyCode::Backspace,
        KeyModifiers::NONE,
    )))
    .unwrap();
    let erased = overlay_frame(layout, mux.overlay.as_ref().unwrap());
    assert_eq!(
        overlay_cursor(&erased, mux.overlay.as_ref().unwrap()).unwrap(),
        empty_cursor
    );

    assert!(overlay_cursor(&empty, &Overlay::Help).is_none());
}

#[test]
fn sidebar_scroll_and_new_hit_use_rendered_capacity() {
    let mut mux = MuxUi::new("/".into());
    mux.handle(Event::Resize(80, 10)).unwrap();
    for id in 1..=5 {
        mux.apply(MuxEvent::Add {
            id,
            name: id.to_string(),
            workspace: "/".into(),
            worktree: false,
            status: MuxStatus::Idle,
            pane: pane("/"),
        });
    }
    for _ in 0..10 {
        mux.handle_mouse(MouseEventKind::ScrollDown, 1, 2);
    }
    assert_eq!(mux.visible_rows(), 2);
    assert_eq!(mux.sidebar_scroll, 3);

    // The session rows are 2..4, row 4 is the spacer, and New is row 5.
    mux.handle_mouse(MouseEventKind::Down(MouseButton::Left), 1, 4);
    assert!(mux.overlay.is_none());
    mux.handle_mouse(MouseEventKind::Down(MouseButton::Left), 1, 5);
    assert!(matches!(mux.overlay, Some(Overlay::NewMenu { .. })));
}

#[test]
fn prefix_command_strip_is_titled_complete_and_bottom_anchored() {
    let mut mux = MuxUi::new("/".into());
    mux.handle(Event::Resize(80, 16)).unwrap();
    assert_eq!(mux.sidebar_footer_rows(), 2);
    assert_eq!(mux.visible_rows(), 8);

    mux.handle(Event::Key(KeyEvent::new(
        KeyCode::Null,
        KeyModifiers::CONTROL,
    )))
    .unwrap();
    assert_eq!(mux.sidebar_footer_rows(), 6);
    assert_eq!(mux.visible_rows(), 4);

    let theme = render::Theme::default();
    let muted = Style::default().fg(theme.muted_text);
    let accent = Style::default().fg(theme.accent);
    let lines = mux.sidebar_lines(26, 16, muted, accent);
    let plain = lines
        .iter()
        .map(|line| strip_ansi(&line_to_ansi(line)))
        .collect::<Vec<_>>();
    assert!(plain[9].starts_with("├─ MUX › ^Space "));
    for (row, label) in [
        (10, "n  new      w  worktree"),
        (11, "c  current  d  directory"),
        (12, "x  close    j  next"),
        (13, "k  prev     1–9 jump"),
        (14, "?  help"),
    ] {
        assert!(
            plain[row].contains(label),
            "missing {label}: {}",
            plain[row]
        );
    }
    assert!(plain[15].starts_with('╰'));
}

#[test]
fn direct_workspace_prefix_shortcuts_open_editors() {
    let mut mux = MuxUi::new("/some/project".into());
    for (shortcut, expected) in [('w', "worktree"), ('c', "current"), ('d', "directory")] {
        mux.overlay = None;
        mux.handle(Event::Key(KeyEvent::new(
            KeyCode::Null,
            KeyModifiers::CONTROL,
        )))
        .unwrap();
        mux.handle(key(shortcut)).unwrap();
        match (expected, mux.overlay.as_ref()) {
            ("worktree", Some(Overlay::Worktree { branch, keep })) => {
                assert!(branch.is_empty());
                assert!(*keep);
            }
            ("current", Some(Overlay::Current { name })) => assert_eq!(name, "project"),
            (
                "directory",
                Some(Overlay::Directory {
                    input, suggestions, ..
                }),
            ) => {
                assert!(input.is_empty());
                assert!(suggestions.is_empty());
            }
            _ => panic!("{shortcut} opened the wrong overlay"),
        }
    }
}

#[test]
fn directory_overlay_has_fixed_filterable_suggestion_list() {
    let layout = MuxUi::new("/x".into()).layout(100, 30);
    let empty = Overlay::Directory {
        input: "a".into(),
        suggestions: vec![],
        selected: 0,
    };
    let many = Overlay::Directory {
        input: "a".into(),
        suggestions: vec!["/first".into(), "/second".into(), "/third".into()],
        selected: 0,
    };
    let empty_frame = overlay_frame(layout, &empty);
    let many_frame = overlay_frame(layout, &many);
    assert_eq!(empty_frame.rows.len(), many_frame.rows.len());
    assert_eq!(empty_frame.y, many_frame.y);
    assert!(many_frame.rows.join("\n").contains("› /first"));
    assert!(many_frame.rows.join("\n").contains("/second"));
    assert!(many_frame.rows.join("\n").contains("/third"));

    let theme = render::Theme::default();
    let mut buffer = Buffer::empty(Rect::new(0, 0, 100, 30));
    draw_overlay(&mut buffer, layout, &many, theme);
    let selected_style = buffer[(many_frame.x + 2, many_frame.y + 3)].style();
    assert_eq!(selected_style.fg, Some(theme.accent));
    assert!(selected_style.add_modifier.contains(Modifier::BOLD));
    let other_style = buffer[(many_frame.x + 2, many_frame.y + 4)].style();
    assert_eq!(other_style.fg, Some(theme.muted_text));
    assert!(other_style.add_modifier.contains(Modifier::DIM));
    let input_style = buffer[(many_frame.x + 2, many_frame.y + 2)].style();
    assert_eq!(input_style.fg, Some(theme.primary_text));
    assert!(!input_style.add_modifier.contains(Modifier::DIM));
}

#[test]
fn help_lists_all_prefix_shortcuts() {
    let text = overlay_frame(MuxUi::new("/".into()).layout(100, 30), &Overlay::Help)
        .rows
        .join("\n");
    for shortcut in [
        "n menu",
        "w worktree",
        "c current",
        "d directory",
        "j/k",
        "1-9",
        "x close",
        "? help",
    ] {
        assert!(text.contains(shortcut), "missing {shortcut}: {text}");
    }
}

#[test]
fn remove_selects_neighbor() {
    let mut m = MuxUi::new("/".into());
    for id in 1..=2 {
        m.apply(MuxEvent::Add {
            id,
            name: id.to_string(),
            workspace: "/".into(),
            worktree: false,
            status: MuxStatus::Starting,
            pane: pane("/"),
        })
    }
    m.apply(MuxEvent::Remove { id: 2 });
    assert_eq!(m.selected_id(), Some(1))
}

#[test]
fn cursor_starts_at_the_top_of_the_main_pane() {
    let layout = MuxLayout {
        main_x: 23,
        ..MuxLayout::default()
    };
    let frame = crate::PaneFrame {
        lines: vec![],
        cursor_row: 4,
        cursor_col: 7,
    };
    assert_eq!(pane_cursor_position(layout, &frame), (30, 4));
}

#[test]
fn page_and_mouse_wheel_route_to_main_pane() {
    let mut mux = MuxUi::new("/".into());
    mux.apply(MuxEvent::Add {
        id: 1,
        name: "one".into(),
        workspace: "/".into(),
        worktree: false,
        status: MuxStatus::Idle,
        pane: pane("/"),
    });
    mux.handle(Event::Key(KeyEvent::new(
        KeyCode::PageUp,
        KeyModifiers::NONE,
    )))
    .unwrap();
    assert_eq!(mux.slots[0].pane.scroll_offset(), 24);

    mux.handle(Event::Mouse(crossterm::event::MouseEvent {
        kind: MouseEventKind::ScrollDown,
        column: 60,
        row: 5,
        modifiers: KeyModifiers::NONE,
    }))
    .unwrap();
    assert_eq!(mux.slots[0].pane.scroll_offset(), 21);

    mux.handle(Event::Mouse(crossterm::event::MouseEvent {
        kind: MouseEventKind::ScrollUp,
        column: 2,
        row: 5,
        modifiers: KeyModifiers::NONE,
    }))
    .unwrap();
    assert_eq!(mux.slots[0].pane.scroll_offset(), 21);
    assert_eq!(mux.sidebar_scroll, 0);
}

#[test]
fn delegated_exit_requires_confirmation_after_draft_is_cleared() {
    let mut mux = MuxUi::new("/".into());
    mux.apply(MuxEvent::Add {
        id: 7,
        name: "one".into(),
        workspace: "/".into(),
        worktree: false,
        status: MuxStatus::Idle,
        pane: pane("/"),
    });
    mux.slots[0].pane.set_draft("discard me");
    let ctrl_c = Event::Key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL));
    assert!(mux.handle(ctrl_c.clone()).unwrap().is_empty());
    assert!(mux.overlay.is_none());
    assert!(mux.handle(ctrl_c.clone()).unwrap().is_empty());
    assert!(matches!(mux.overlay, Some(Overlay::ConfirmExit)));

    assert!(
        mux.handle(Event::Key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE)))
            .unwrap()
            .is_empty()
    );
    assert!(mux.overlay.is_none());

    assert!(mux.handle(ctrl_c).unwrap().is_empty());
    assert!(matches!(mux.overlay, Some(Overlay::ConfirmExit)));
    assert!(matches!(
        mux.handle(Event::Key(KeyEvent::new(
            KeyCode::Enter,
            KeyModifiers::NONE
        )))
        .unwrap()
        .as_slice(),
        [MuxAction::Exit]
    ));
}
