//! Best-effort integration with Herdr's pane-agent CLI.

use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tokio::time::{Duration, timeout};

const SOURCE: &str = "harness";
const AGENT: &str = "Harness";
static LAST_SEQUENCE: AtomicU64 = AtomicU64::new(0);

type ReportSnapshot = (State, Option<String>, Option<Vec<String>>);

#[derive(Clone, Debug, PartialEq, Eq)]
#[allow(dead_code)] // The current Harness tool policy has no approval gate to enter blocked state.
pub(crate) enum State {
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

    fn report_args(
        &self,
        state: &State,
        session: Option<&str>,
        resume: Option<&[String]>,
    ) -> Vec<String> {
        let mut args = vec![
            "pane".into(),
            "report-agent".into(),
            self.pane.clone(),
            "--source".into(),
            SOURCE.into(),
            "--agent".into(),
            AGENT.into(),
            "--state".into(),
            state.label().into(),
            "--seq".into(),
            sequence(),
        ];
        if let State::Blocked(Some(message)) = state {
            args.extend(["--message".into(), message.clone()]);
        }
        if let Some(session) = session {
            args.extend(["--agent-session-id".into(), session.into()]);
        }
        if let Some(argv) = resume.filter(|argv| valid_resume_argv(argv)) {
            args.push("--".into());
            args.extend(argv.iter().cloned());
        }
        args
    }

    async fn invoke(&self, args: Vec<String>) {
        let mut command = tokio::process::Command::new(&self.bin);
        command
            .args(args)
            .kill_on_drop(true)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        let _ = timeout(Duration::from_millis(750), command.status()).await;
    }

    async fn report(&self, state: State, session: Option<String>, resume: Option<Vec<String>>) {
        self.invoke(self.report_args(&state, session.as_deref(), resume.as_deref()))
            .await;
    }

    pub(crate) fn start(&self) -> (Reporter, JoinHandle<()>) {
        let initial = (State::Idle, None, None);
        let (tx, mut rx) = watch::channel(initial.clone());
        let herdr = self.clone();
        let task = tokio::spawn(async move {
            let mut previous: Option<ReportSnapshot> = None;
            while rx.changed().await.is_ok() {
                let next = rx.borrow_and_update().clone();
                if previous.as_ref() == Some(&next) {
                    continue;
                }
                previous = Some(next.clone());
                herdr.report(next.0, next.1, next.2).await;
            }
        });
        (
            Reporter {
                tx,
                current: std::sync::Arc::new(std::sync::Mutex::new(initial)),
            },
            task,
        )
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

#[derive(Clone)]
pub(crate) struct Reporter {
    tx: watch::Sender<ReportSnapshot>,
    current: std::sync::Arc<std::sync::Mutex<ReportSnapshot>>,
}
impl Reporter {
    pub(crate) fn report(&self, state: State, session: Option<String>) {
        if let Ok(mut current) = self.current.lock() {
            current.0 = state;
            if let Some(id) = session {
                current.1 = Some(id);
            }
            let _ = self.tx.send(current.clone());
        }
    }
    pub(crate) fn with_session(&self, id: String, model: String) {
        if let Ok(mut current) = self.current.lock() {
            current.0 = State::Idle;
            current.1 = Some(id.clone());
            current.2 = Some(vec![
                "harness".into(),
                "--model".into(),
                model,
                "--resume-session".into(),
                id,
            ]);
            let _ = self.tx.send(current.clone());
        }
    }
    pub(crate) fn model(&self, model: String) {
        if let Ok(mut current) = self.current.lock()
            && let Some(id) = current.1.clone()
        {
            current.2 = Some(vec![
                "harness".into(),
                "--model".into(),
                model,
                "--resume-session".into(),
                id,
            ]);
            let _ = self.tx.send(current.clone());
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
    #[test]
    fn report_uses_cli_syntax_and_resume_preserves_model_and_session() {
        let h = Herdr {
            bin: "herdr".into(),
            pane: "pane-1".into(),
        };
        let argv = vec![
            "harness".into(),
            "--model".into(),
            "model-x".into(),
            "--resume-session".into(),
            "sess".into(),
        ];
        let args = h.report_args(&State::Working, Some("sess"), Some(&argv));
        assert!(args.windows(2).any(|w| w == ["--state", "working"]));
        assert!(args.windows(2).any(|w| w == ["--source", "harness"]));
        assert!(args.windows(2).any(|w| w == ["--agent", "Harness"]));
        assert!(args.windows(2).any(|w| w == ["--agent-session-id", "sess"]));
        assert!(args.windows(2).any(|w| w == ["--model", "model-x"]));
        assert!(args.ends_with(&argv));
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
    async fn state_updates_retain_session_and_resume_argv() {
        let h = Herdr {
            bin: "herdr".into(),
            pane: "pane".into(),
        };
        let (reporter, task) = h.start();
        reporter.with_session("session-1".into(), "model-a".into());
        reporter.report(State::Working, None);
        let current = reporter.current.lock().unwrap().clone();
        assert_eq!(current.1.as_deref(), Some("session-1"));
        assert!(current.2.as_ref().unwrap().contains(&"model-a".into()));
        reporter.model("model-b".into());
        assert!(
            reporter
                .current
                .lock()
                .unwrap()
                .2
                .as_ref()
                .unwrap()
                .contains(&"model-b".into())
        );
        drop(reporter);
        task.abort();
    }
}
