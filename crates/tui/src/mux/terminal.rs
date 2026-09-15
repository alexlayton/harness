use super::directory::directory_suggestions;
use super::{DirectoryCompletion, MuxAction, MuxEvent, MuxUi, Overlay};
use anyhow::{Context, Result};
use crossterm::cursor::Show;
use crossterm::event::{
    DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture,
    EventStream, KeyboardEnhancementFlags, PopKeyboardEnhancementFlags,
    PushKeyboardEnhancementFlags,
};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use futures_util::StreamExt;
use std::io::{self, Stdout, Write};
use std::sync::Once;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

impl MuxUi {
    /// Run as the sole terminal owner. Restoration is attempted on normal
    /// return, cancellation, input failure, and panic (including aborting
    /// release builds, via the process panic hook).
    pub async fn run(
        mut self,
        mut events: mpsc::UnboundedReceiver<MuxEvent>,
        actions: mpsc::UnboundedSender<MuxAction>,
        cancel: CancellationToken,
    ) -> Result<()> {
        install_mux_panic_hook();
        let mut terminal = TerminalGuard::enter()?;
        let mut input = EventStream::new();
        let mut tick = tokio::time::interval(Duration::from_millis(100));
        let (completion_tx, mut completion_rx) = mpsc::unbounded_channel();
        let mut completion_task: Option<JoinHandle<()>> = None;
        self.draw(terminal.out())?;

        let result: Result<()> = async {
            loop {
                tokio::select! {
                    event = events.recv() => {
                        let Some(event) = event else { break };
                        self.apply(event);
                    }
                    event = input.next() => {
                        let Some(event) = event else { break };
                        for action in self.handle(event.context("read terminal event")?)? {
                            if actions.send(action).is_err() {
                                return Ok(());
                            }
                        }
                        if let Some(request) = self.directory_request.take() {
                            if let Some(task) = completion_task.take() {
                                task.abort();
                            }
                            let tx = completion_tx.clone();
                            let cwd = self.cwd.clone();
                            completion_task = Some(tokio::spawn(async move {
                                tokio::time::sleep(Duration::from_millis(120)).await;
                                let input = request.input;
                                let scan_input = input.clone();
                                let suggestions = tokio::task::spawn_blocking(move || {
                                    directory_suggestions(&scan_input, &cwd)
                                }).await;
                                if let Ok(suggestions) = suggestions {
                                    let _ = tx.send(DirectoryCompletion {
                                        generation: request.generation,
                                        input,
                                        suggestions,
                                    });
                                }
                            }));
                        } else if !matches!(self.overlay, Some(Overlay::Directory { .. }))
                            && let Some(task) = completion_task.take()
                        {
                            task.abort();
                        }
                    }
                    Some(completion) = completion_rx.recv() => {
                        self.apply_directory_completion(completion);
                    }
                    _ = tick.tick() => {
                        let selected = self.selected;
                        let changed = self.slots.iter_mut().enumerate().fold(
                            false,
                            |changed, (index, slot)| slot.pane.tick(selected == Some(index)) || changed,
                        );
                        if !changed {
                            continue;
                        }
                    }
                    _ = cancel.cancelled() => {
                        let _ = actions.send(MuxAction::Exit);
                        break;
                    }
                }
                self.draw(terminal.out())?;
            }
            Ok(())
        }
        .await;
        if let Some(task) = completion_task {
            task.abort();
        }
        drop(terminal);
        result
    }
}

static MUX_RAW_MODE: AtomicBool = AtomicBool::new(false);
static MUX_ALT_SCREEN: AtomicBool = AtomicBool::new(false);
static MUX_BRACKETED_PASTE: AtomicBool = AtomicBool::new(false);
static MUX_MOUSE_CAPTURE: AtomicBool = AtomicBool::new(false);
static MUX_KEYBOARD_FLAGS: AtomicBool = AtomicBool::new(false);

fn restore_mux_terminal(out: &mut Stdout) {
    if MUX_KEYBOARD_FLAGS.swap(false, Ordering::SeqCst) {
        let _ = execute!(out, PopKeyboardEnhancementFlags);
    }
    let _ = execute!(out, Show);
    if MUX_MOUSE_CAPTURE.swap(false, Ordering::SeqCst) {
        let _ = execute!(out, DisableMouseCapture);
    }
    if MUX_BRACKETED_PASTE.swap(false, Ordering::SeqCst) {
        let _ = execute!(out, DisableBracketedPaste);
    }
    if MUX_ALT_SCREEN.swap(false, Ordering::SeqCst) {
        let _ = execute!(out, LeaveAlternateScreen);
    }
    if MUX_RAW_MODE.swap(false, Ordering::SeqCst) {
        let _ = disable_raw_mode();
    }
    let _ = out.flush();
}

fn install_mux_panic_hook() {
    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |panic| {
            // Abort builds cannot run guards, so restoration must happen in
            // the hook. Unwind builds may catch a background-task panic; in
            // that case restoring here would dismantle a still-running mux.
            // The terminal-owning future restores through its guard instead.
            #[cfg(panic = "abort")]
            restore_mux_terminal(&mut io::stdout());
            previous(panic);
        }));
    });
}

struct TerminalGuard {
    out: Stdout,
}
impl TerminalGuard {
    fn enter() -> Result<Self> {
        enable_raw_mode().context("enable raw mode")?;
        MUX_RAW_MODE.store(true, Ordering::SeqCst);
        let mut this = Self { out: io::stdout() };
        let setup = (|| -> io::Result<()> {
            execute!(this.out, EnterAlternateScreen)?;
            MUX_ALT_SCREEN.store(true, Ordering::SeqCst);
            execute!(this.out, EnableBracketedPaste)?;
            MUX_BRACKETED_PASTE.store(true, Ordering::SeqCst);
            execute!(this.out, EnableMouseCapture)?;
            MUX_MOUSE_CAPTURE.store(true, Ordering::SeqCst);
            Ok(())
        })();
        if let Err(error) = setup {
            drop(this);
            return Err(error).context("configure mux terminal");
        }
        if execute!(
            this.out,
            PushKeyboardEnhancementFlags(KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES)
        )
        .is_ok()
        {
            MUX_KEYBOARD_FLAGS.store(true, Ordering::SeqCst);
        }
        Ok(this)
    }
    fn out(&mut self) -> &mut Stdout {
        &mut self.out
    }
}
impl Drop for TerminalGuard {
    fn drop(&mut self) {
        restore_mux_terminal(&mut self.out);
    }
}
