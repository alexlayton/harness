//! Best-effort integration with Herdr's pane-agent CLI.

use std::path::PathBuf;
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio::time::{Duration, timeout};

const SOURCE: &str = "harness";
const AGENT: &str = "Harness";
static LAST_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// A `harness worktree` launch that the resume command must recreate.
///
/// Sessions are scoped to the workspace directory, and Herdr restores the
/// pane in the shell's directory (the launch directory), so the resume command
/// must enter the same worktree again before it loads the session.
#[derive(Clone, Debug)]
pub(crate) struct WorktreeResume {
    pub branch: String,
    pub dir: PathBuf,
}

/// Settings needed to reopen a session with the same provider and runtime policy.
#[derive(Clone)]
pub(crate) struct ResumeOptions {
    pub provider: String,
    pub model: String,
    pub reasoning: String,
    pub no_context_files: bool,
    pub defer_session_sync: bool,
    pub worktree: Option<WorktreeResume>,
    /// Opt-in MCP servers connected by this process, so a resumed session
    /// gets the same ones.
    pub mcp: Vec<String>,
}

impl ResumeOptions {
    /// Build the resume argv, or `None` when it cannot be expressed safely
    /// (for example a worktree path that is not UTF-8).
    fn argv(&self, id: &str) -> Option<Vec<String>> {
        let mut argv = vec![
            "harness".into(),
            "--provider".into(),
            self.provider.clone(),
            "--model".into(),
            self.model.clone(),
            "--reasoning-effort".into(),
            self.reasoning.clone(),
            "--resume-session".into(),
            id.into(),
        ];
        if self.no_context_files {
            argv.push("--no-context-files".into());
        }
        if self.defer_session_sync {
            argv.push("--defer-session-sync".into());
        }
        for name in &self.mcp {
            argv.push("--mcp".into());
            argv.push(name.clone());
        }
        // Global flags stay before the subcommand. Retention (`--keep`) is
        // sticky in the repository, so it does not need to be repeated.
        if let Some(worktree) = &self.worktree {
            argv.extend([
                "worktree".into(),
                worktree.branch.clone(),
                "--dir".into(),
                worktree.dir.to_str()?.into(),
            ]);
        }
        Some(argv)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[allow(dead_code)] // The current Harness tool policy has no approval gate to enter blocked state.
pub(crate) enum State {
    #[default]
    Idle,
    Working,
    Blocked(Option<String>),
}
impl State {
    fn label(&self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Working => "working",
            Self::Blocked(_) => "blocked",
        }
    }
}

/// Everything Harness reports for its pane. Only the newest snapshot is sent.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Snapshot {
    state: State,
    session: Option<String>,
    resume: Option<Vec<String>>,
    /// Display-only session title, sent as pane metadata.
    title: Option<String>,
}
impl Snapshot {
    /// True when the lifecycle part (`report-agent`) is unchanged.
    fn same_agent_report(&self, other: &Self) -> bool {
        self.state == other.state && self.session == other.session && self.resume == other.resume
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Herdr {
    bin: String,
    pane: String,
}
impl Herdr {
    /// Enable integration only in a complete Herdr pane environment.
    pub(crate) fn from_env() -> Option<Self> {
        if std::env::var("HERDR_ENV").ok().as_deref() != Some("1") {
            return None;
        }
        let bin = std::env::var("HERDR_BIN_PATH")
            .ok()
            .filter(|v| !v.is_empty())?;
        let pane = std::env::var("HERDR_PANE_ID")
            .ok()
            .filter(|v| !v.is_empty())?;
        std::env::var("HERDR_SOCKET_PATH")
            .ok()
            .filter(|v| !v.is_empty())?;
        Some(Self { bin, pane })
    }

    fn report_args(&self, snapshot: &Snapshot) -> Vec<String> {
        let mut args = vec![
            "pane".into(),
            "report-agent".into(),
            self.pane.clone(),
            "--source".into(),
            SOURCE.into(),
            "--agent".into(),
            AGENT.into(),
            "--state".into(),
            snapshot.state.label().into(),
            "--seq".into(),
            sequence(),
        ];
        if let State::Blocked(Some(message)) = &snapshot.state {
            // `=` keeps a message that starts with `-` from parsing as a flag.
            args.push(format!("--message={message}"));
        }
        if let Some(session) = &snapshot.session {
            args.extend(["--agent-session-id".into(), session.clone()]);
        }
        if let Some(argv) = snapshot
            .resume
            .as_deref()
            .filter(|argv| valid_resume_argv(argv))
        {
            args.push("--".into());
            args.extend(argv.iter().cloned());
        }
        args
    }

    fn metadata_args(&self, title: Option<&str>) -> Vec<String> {
        vec![
            "pane".into(),
            "report-metadata".into(),
            self.pane.clone(),
            "--source".into(),
            SOURCE.into(),
            // Guard: the title applies only while Harness holds the pane.
            "--agent".into(),
            AGENT.into(),
            // `=` keeps a title that starts with `-` from parsing as a flag.
            title.map_or_else(
                || "--clear-title".into(),
                |title| format!("--title={title}"),
            ),
            "--seq".into(),
            sequence(),
        ]
    }

    async fn invoke(&self, args: Vec<String>) {
        let mut command = tokio::process::Command::new(&self.bin);
        command
            .args(args)
            .kill_on_drop(true)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        match timeout(Duration::from_millis(750), command.status()).await {
            Ok(Ok(status)) if status.success() => {}
            Ok(Ok(status)) => tracing::warn!(%status, "Herdr command failed"),
            Ok(Err(error)) => tracing::warn!(%error, "could not run Herdr command"),
            Err(_) => tracing::warn!("Herdr command timed out"),
        }
    }

    pub(crate) fn start(&self) -> (Reporter, JoinHandle<()>) {
        let (reporter, mut rx) = Reporter::channel();
        let herdr = self.clone();
        let task = tokio::spawn(async move {
            let mut sent: Option<Snapshot> = None;
            let mut sent_title: Option<String> = None;
            // Herdr asks integrations to send only the latest state: changes
            // queued while a CLI call is pending are superseded, not replayed.
            while let Some(next) = latest(&mut rx).await {
                if sent
                    .as_ref()
                    .is_none_or(|sent| !sent.same_agent_report(&next))
                {
                    herdr.invoke(herdr.report_args(&next)).await;
                }
                if next.title != sent_title {
                    herdr
                        .invoke(herdr.metadata_args(next.title.as_deref()))
                        .await;
                    sent_title.clone_from(&next.title);
                }
                sent = Some(next);
            }
            // Release clears lifecycle state; metadata is cleared explicitly
            // so a stale title does not outlive the process.
            if sent_title.is_some() {
                herdr.invoke(herdr.metadata_args(None)).await;
            }
        });
        (reporter, task)
    }

    pub(crate) async fn release(&self) {
        self.invoke(vec![
            "pane".into(),
            "release-agent".into(),
            self.pane.clone(),
            "--source".into(),
            SOURCE.into(),
            "--agent".into(),
            AGENT.into(),
            "--seq".into(),
            sequence(),
        ])
        .await;
    }
}

/// Wait for the next snapshot, then skip to the newest one already queued.
async fn latest(rx: &mut mpsc::UnboundedReceiver<Snapshot>) -> Option<Snapshot> {
    let mut snapshot = rx.recv().await?;
    while let Ok(newer) = rx.try_recv() {
        snapshot = newer;
    }
    Some(snapshot)
}

#[derive(Clone)]
pub(crate) struct Reporter {
    tx: mpsc::UnboundedSender<Snapshot>,
    current: std::sync::Arc<std::sync::Mutex<Snapshot>>,
}
impl Reporter {
    fn channel() -> (Self, mpsc::UnboundedReceiver<Snapshot>) {
        let (tx, rx) = mpsc::unbounded_channel();
        (
            Self {
                tx,
                current: Default::default(),
            },
            rx,
        )
    }

    fn update(&self, change: impl FnOnce(&mut Snapshot)) {
        if let Ok(mut current) = self.current.lock() {
            let previous = current.clone();
            change(&mut current);
            if *current != previous {
                // Send under the lock so the newest queued snapshot is the
                // newest state even with concurrent reporters.
                let _ = self.tx.send(current.clone());
            }
        }
    }

    pub(crate) fn report(&self, state: State, session: Option<String>) {
        self.update(|current| {
            current.state = state;
            if let Some(id) = session {
                current.session = Some(id);
            }
        });
    }
    pub(crate) fn with_session(&self, id: String, options: &ResumeOptions) {
        self.update(|current| {
            current.resume = options.argv(&id);
            current.session = Some(id);
        });
    }
    pub(crate) fn resume_options(&self, options: &ResumeOptions) {
        self.update(|current| {
            if let Some(id) = &current.session {
                current.resume = options.argv(id);
            }
        });
    }
    fn title(&self, title: Option<&str>) {
        let title = title
            .map(str::trim)
            .filter(|title| !title.is_empty())
            .map(str::to_owned);
        self.update(|current| current.title = title);
    }

    /// Map one agent event to a Herdr state or resume-command update. The
    /// TUI and headless frontends share this so their reports cannot drift.
    pub(crate) fn observe(&self, event: &agent::AgentEvent, options: &mut ResumeOptions) {
        use agent::AgentEvent;
        match event {
            AgentEvent::ModelChanged { provider, model } => {
                options.provider = provider.clone();
                options.model = model.clone();
                self.resume_options(options);
            }
            AgentEvent::ReasoningChanged { level } => {
                options.reasoning = level.clone();
                self.resume_options(options);
            }
            AgentEvent::TextDelta(_)
            | AgentEvent::ReasoningDelta(_)
            | AgentEvent::ToolCallStarted { .. }
            | AgentEvent::Retrying { .. }
            | AgentEvent::AskStarted => self.report(State::Working, None),
            AgentEvent::TurnFinished | AgentEvent::OperationFinished => {
                self.report(State::Idle, None)
            }
            AgentEvent::SessionChanged { id, title, .. } => {
                self.with_session(id.clone(), options);
                self.title(title.as_deref());
            }
            AgentEvent::SessionTitleChanged { title } => self.title(Some(title)),
            _ => {}
        }
    }
}

fn sequence() -> String {
    // Unix nanoseconds are globally ordered in practice; the process-local CAS
    // also guarantees strict ordering when reports share a clock tick.
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
        .min(u64::MAX as u128) as u64;
    let mut current = LAST_SEQUENCE.load(Ordering::Relaxed);
    loop {
        let next = now.max(current.saturating_add(1));
        match LAST_SEQUENCE.compare_exchange_weak(
            current,
            next,
            Ordering::SeqCst,
            Ordering::Relaxed,
        ) {
            Ok(_) => return next.to_string(),
            Err(observed) => current = observed,
        }
    }
}

/// Herdr's `--` resume argv is deliberately stricter than a shell command:
/// it is passed directly to exec and must be safe for Herdr's persisted form.
fn valid_resume_argv(argv: &[String]) -> bool {
    if argv.is_empty()
        || argv.len() > 64
        || argv[0].is_empty()
        || (argv[0].contains('/') || argv[0].contains('\\'))
    {
        return false;
    }
    let bytes = argv.iter().map(String::len).sum::<usize>() + argv.len().saturating_sub(1);
    bytes <= 8 * 1024
        && argv
            .iter()
            .all(|arg| !arg.contains('\'') && !arg.chars().any(char::is_control))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options() -> ResumeOptions {
        ResumeOptions {
            provider: "openrouter".into(),
            model: "model-x".into(),
            reasoning: "low".into(),
            no_context_files: false,
            defer_session_sync: true,
            worktree: None,
            mcp: Vec::new(),
        }
    }

    #[test]
    fn report_uses_cli_syntax_and_resume_preserves_model_and_session() {
        let h = Herdr {
            bin: "herdr".into(),
            pane: "pane-1".into(),
        };
        let mut options = options();
        options.mcp = vec!["email".into()];
        let argv = options.argv("sess").unwrap();
        let args = h.report_args(&Snapshot {
            state: State::Working,
            session: Some("sess".into()),
            resume: Some(argv.clone()),
            title: None,
        });
        assert!(args.windows(2).any(|w| w == ["--state", "working"]));
        assert!(args.windows(2).any(|w| w == ["--source", "harness"]));
        assert!(args.windows(2).any(|w| w == ["--agent", "Harness"]));
        assert!(args.windows(2).any(|w| w == ["--agent-session-id", "sess"]));
        assert!(args.windows(2).any(|w| w == ["--model", "model-x"]));
        assert!(args.windows(2).any(|w| w == ["--provider", "openrouter"]));
        assert!(args.windows(2).any(|w| w == ["--reasoning-effort", "low"]));
        assert!(args.contains(&"--defer-session-sync".into()));
        assert!(argv.windows(2).any(|w| w == ["--mcp", "email"]));
        assert!(args.ends_with(&argv));
    }

    #[test]
    fn worktree_resume_reenters_the_worktree_after_global_flags() {
        let mut options = options();
        options.worktree = Some(WorktreeResume {
            branch: "feature/x".into(),
            dir: "/repo/.worktrees/x".into(),
        });
        let argv = options.argv("sess").unwrap();
        assert!(argv.ends_with(&[
            "worktree".into(),
            "feature/x".into(),
            "--dir".into(),
            "/repo/.worktrees/x".into(),
        ]));
        assert!(argv.windows(2).any(|w| w == ["--resume-session", "sess"]));
        assert!(valid_resume_argv(&argv));
    }

    #[test]
    fn resume_argv_obeys_herdr_limits() {
        assert!(valid_resume_argv(&[
            "harness".into(),
            "--model".into(),
            "x".into()
        ]));
        assert!(!valid_resume_argv(&["./harness".into()]));
        assert!(!valid_resume_argv(&["harness".into(), "bad'name".into()]));
        assert!(!valid_resume_argv(&["harness".into(), "bad\nname".into()]));
        assert!(!valid_resume_argv(
            &(0..64).map(|_| "x".repeat(128)).collect::<Vec<_>>()
        ));
        assert!(sequence().parse::<u64>().unwrap() < sequence().parse::<u64>().unwrap());
    }

    #[tokio::test]
    async fn queued_reports_collapse_to_the_newest_state() {
        let (reporter, mut rx) = Reporter::channel();
        reporter.report(State::Working, None);
        reporter.report(State::Working, None); // Repeated stream updates are coalesced.
        reporter.report(State::Idle, None);
        assert_eq!(latest(&mut rx).await.unwrap().state, State::Idle);
        assert!(rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn state_updates_retain_session_and_resume_argv() {
        let h = Herdr {
            bin: "herdr".into(),
            pane: "pane".into(),
        };
        let (reporter, task) = h.start();
        let mut options = ResumeOptions {
            provider: "openrouter".into(),
            model: "model-a".into(),
            reasoning: "high".into(),
            no_context_files: true,
            defer_session_sync: false,
            worktree: None,
            mcp: Vec::new(),
        };
        reporter.report(State::Working, None);
        reporter.with_session("session-1".into(), &options);
        let current = reporter.current.lock().unwrap().clone();
        let resume = current.resume.unwrap();
        assert_eq!(current.session.as_deref(), Some("session-1"));
        assert_eq!(current.state, State::Working);
        for expected in ["model-a", "openrouter", "high", "--no-context-files"] {
            assert!(resume.contains(&expected.into()), "{expected}");
        }
        options.model = "model-b".into();
        reporter.resume_options(&options);
        reporter.with_session("session-2".into(), &options);
        let current = reporter.current.lock().unwrap().clone();
        assert_eq!(current.state, State::Working);
        assert_eq!(current.session.as_deref(), Some("session-2"));
        assert!(current.resume.unwrap().contains(&"model-b".into()));
        drop(reporter);
        task.abort();
    }
}
