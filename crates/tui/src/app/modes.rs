//! Terminal mode bookkeeping: the backend seam, what was enabled, and the
//! restore paths (normal exit, `Drop`, and the panic hook).

use super::*;

/// Operations needed to change terminal state. Keeping this small seam here
/// makes setup and cleanup testable without making the rendering or input
/// protocol depend on a different terminal implementation.
pub(super) trait TerminalBackend: Send {
    fn size(&self) -> io::Result<(u16, u16)>;
    /// Cursor `(column, row)`, when the terminal answers the query. Used once
    /// at startup to locate the live region for click-to-position.
    fn cursor_position(&self) -> Option<(u16, u16)> {
        None
    }
    fn enable_raw_mode(&mut self) -> io::Result<()>;
    fn disable_raw_mode(&mut self) -> io::Result<()>;
    fn enable_bracketed_paste(&mut self, out: &mut Stdout) -> io::Result<()>;
    fn disable_bracketed_paste(&mut self, out: &mut Stdout) -> io::Result<()>;
    fn push_keyboard_flags(&mut self, out: &mut Stdout) -> io::Result<()>;
    fn pop_keyboard_flags(&mut self, out: &mut Stdout) -> io::Result<()>;
    fn write_newline(&mut self, out: &mut Stdout) -> io::Result<()>;
}

/// The production terminal operation layer. Crossterm is deliberately kept
/// behind [`TerminalBackend`], so mode bookkeeping can be tested independently
/// of the process terminal and its platform-specific implementation.
pub(super) struct CrosstermBackend;

impl TerminalBackend for CrosstermBackend {
    fn size(&self) -> io::Result<(u16, u16)> {
        terminal::size()
    }

    fn cursor_position(&self) -> Option<(u16, u16)> {
        crossterm::cursor::position().ok()
    }

    fn enable_raw_mode(&mut self) -> io::Result<()> {
        terminal::enable_raw_mode()
    }

    fn disable_raw_mode(&mut self) -> io::Result<()> {
        terminal::disable_raw_mode()
    }

    fn enable_bracketed_paste(&mut self, out: &mut Stdout) -> io::Result<()> {
        execute!(out, EnableBracketedPaste)
    }

    fn disable_bracketed_paste(&mut self, out: &mut Stdout) -> io::Result<()> {
        execute!(out, DisableBracketedPaste)
    }

    fn push_keyboard_flags(&mut self, out: &mut Stdout) -> io::Result<()> {
        execute!(
            out,
            PushKeyboardEnhancementFlags(KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES)
        )
    }

    fn pop_keyboard_flags(&mut self, out: &mut Stdout) -> io::Result<()> {
        execute!(out, PopKeyboardEnhancementFlags)
    }

    fn write_newline(&mut self, out: &mut Stdout) -> io::Result<()> {
        writeln!(out)
    }
}

/// Terminal modes successfully enabled by one UI instance. Cleanup only
/// reverses modes recorded here, so a partial setup cannot pop another
/// component's keyboard state. Failed cleanup remains recorded until a later
/// retry succeeds.
#[derive(Default)]
pub(super) struct TerminalModes {
    pub(super) raw_mode: bool,
    pub(super) bracketed_paste: bool,
    pub(super) keyboard_flags: bool,
    pub(super) newline_written: bool,
}

pub(super) struct TerminalState {
    pub(super) backend: Box<dyn TerminalBackend>,
    pub(super) modes: TerminalModes,
}

impl TerminalState {
    fn is_restored(&self) -> bool {
        !self.modes.raw_mode
            && !self.modes.bracketed_paste
            && !self.modes.keyboard_flags
            && self.modes.newline_written
    }
}

/// Restore every state component that still needs work. An operation that
/// fails remains recorded, allowing a later call from `run`, `Drop`, or a
/// panic path to retry it. All operations are attempted before the first
/// error is returned.
pub(super) fn restore_terminal(state: &mut TerminalState, out: &mut Stdout) -> Result<()> {
    if state.is_restored() {
        return Ok(());
    }

    let mut first_error = None;
    if state.modes.raw_mode {
        if let Err(error) = state
            .backend
            .disable_raw_mode()
            .context("restore terminal raw mode")
        {
            first_error.get_or_insert(error);
        } else {
            state.modes.raw_mode = false;
        }
    }
    if state.modes.bracketed_paste {
        if let Err(error) = state
            .backend
            .disable_bracketed_paste(out)
            .context("restore bracketed paste")
        {
            first_error.get_or_insert(error);
        } else {
            state.modes.bracketed_paste = false;
        }
    }
    if state.modes.keyboard_flags {
        if let Err(error) = state
            .backend
            .pop_keyboard_flags(out)
            .context("restore keyboard enhancement flags")
        {
            first_error.get_or_insert(error);
        } else {
            state.modes.keyboard_flags = false;
        }
    }
    if !state.modes.newline_written {
        if let Err(error) = state.backend.write_newline(out).context("leave terminal") {
            first_error.get_or_insert(error);
        } else {
            state.modes.newline_written = true;
        }
    }

    match first_error {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

/// Lock the shared terminal state. A panic while the lock was held must not
/// stop later restoration, so a poisoned lock is recovered: the state only
/// records which modes still need to be undone.
pub(super) fn lock_terminal(terminal: &Mutex<TerminalState>) -> MutexGuard<'_, TerminalState> {
    terminal.lock().unwrap_or_else(PoisonError::into_inner)
}

pub(super) fn install_panic_hook(terminal: &Arc<Mutex<TerminalState>>) {
    let terminal = Arc::downgrade(terminal);
    let ui_thread = std::thread::current().id();
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic| {
        if std::thread::current().id() == ui_thread
            && let Some(terminal) = terminal.upgrade()
        {
            let guard = match terminal.try_lock() {
                Ok(guard) => Some(guard),
                Err(TryLockError::Poisoned(poisoned)) => Some(poisoned.into_inner()),
                Err(TryLockError::WouldBlock) => None,
            };
            if let Some(mut guard) = guard {
                let mut stdout = io::stdout();
                // Panic hooks cannot report a cleanup error without obscuring
                // the original panic, but they still use the same best-effort
                // operation layer. A caught panic or a later Drop can retry
                // anything that failed.
                let _ = restore_terminal(&mut guard, &mut stdout);
            }
        }
        previous(panic);
    }));
}
