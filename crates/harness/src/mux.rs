//! Host-side lifecycle supervisor for the terminal multiplexer.

use crate::config::{
    Config, ProviderArg, WorktreeArgs, build_provider_with_auths, provider_factory,
};
use crate::{context, tui_adapter, worktree};
use agent::assembly::AgentBuilder;
use agent::{AgentEvent, InputMessage};
use anyhow::{Context, Result};
use llm::ReasoningPolicy;
use session::{SessionCreateOptions, SessionStore};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::task::{JoinHandle, JoinSet};
use tokio_util::sync::CancellationToken;
use tools::{ToolConfig, ToolExecutionGate, default_registry};
use tui::{
    AgentPane, ContextFileEntry, MuxAction, MuxEvent, MuxId, MuxStatus, MuxUi, WorkspaceChoice,
};

#[derive(Clone)]
struct SlotSettings {
    provider: String,
    model: String,
    reasoning: ReasoningPolicy,
}

#[derive(Clone, Default)]
struct WorkspaceMutationCoordinator {
    workspace_gates: Arc<Mutex<HashMap<PathBuf, ToolExecutionGate>>>,
}

impl WorkspaceMutationCoordinator {
    fn gate_for(&self, workspace: &Path) -> ToolExecutionGate {
        let mut gates = self
            .workspace_gates
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        gates
            .entry(workspace.to_path_buf())
            .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(())))
            .clone()
    }
}

struct SlotHandle {
    input: mpsc::UnboundedSender<InputMessage>,
    cancel: CancellationToken,
    task: JoinHandle<()>,
    settings: SlotSettings,
}

/// A mux-created worktree is retained on every exit path, including setup errors.
struct RetainedLease(Option<worktree::WorktreeLease>);

impl Drop for RetainedLease {
    fn drop(&mut self) {
        if let Some(lease) = self.0.take() {
            lease.release_without_cleanup();
        }
    }
}

enum RuntimeMessage {
    Prepared {
        id: MuxId,
        name: String,
        workspace: PathBuf,
        worktree: bool,
        pane: Box<AgentPane>,
    },
    Ready(MuxId),
    Event(MuxId, AgentEvent),
    AssemblyError(MuxId, String),
    Stopped(MuxId),
}

/// Run one terminal mux rooted permanently at the canonical launch directory.
pub(crate) async fn run(config: Config, cli: &crate::config::Cli, launch: PathBuf) -> Result<()> {
    let (event_tx, event_rx) = mpsc::unbounded_channel();
    let (action_tx, mut action_rx) = mpsc::unbounded_channel();
    let (runtime_tx, mut runtime_rx) = mpsc::unbounded_channel();
    let cancel = CancellationToken::new();
    let mut ui_task =
        tokio::spawn(MuxUi::new(launch.clone()).run(event_rx, action_tx, cancel.clone()));
    let mut slots = HashMap::<MuxId, SlotHandle>::new();
    let mut reapers = JoinSet::new();
    let mutation_coordinator = WorkspaceMutationCoordinator::default();
    let mut next_id = 1;
    let defaults = SlotSettings {
        provider: config.provider.to_string(),
        model: config.model.clone(),
        reasoning: config.reasoning,
    };

    create_slot(
        next_id,
        WorkspaceChoice::Directory {
            path: launch.clone(),
            name: basename(&launch),
        },
        defaults.clone(),
        &config,
        cli,
        &launch,
        &event_tx,
        &runtime_tx,
        &mut slots,
        &mutation_coordinator,
    );
    next_id += 1;

    loop {
        tokio::select! {
            action = action_rx.recv() => match action {
                Some(MuxAction::AgentInput { id, input }) => {
                    let starts_work = matches!(
                        &input,
                        tui::InputMessage::Message(_)
                            | tui::InputMessage::InvokeSkill { .. }
                            | tui::InputMessage::CompactSession
                    );
                    if let Some(slot) = slots.get(&id)
                        && slot.input.send(tui_adapter::into_agent_input(input)).is_ok()
                        && starts_work
                    {
                        let _ = event_tx.send(MuxEvent::Status { id, status: MuxStatus::Running });
                    }
                }
                    Some(MuxAction::Create { choice, inherit_from }) => {
                    let settings = inherit_from.and_then(|id| slots.get(&id)).map(|s| s.settings.clone()).unwrap_or_else(|| defaults.clone());
                    create_slot(next_id, choice, settings, &config, cli, &launch, &event_tx, &runtime_tx, &mut slots, &mutation_coordinator);
                    next_id += 1;
                }
                Some(MuxAction::Close { id }) => close_slot(id, &mut slots, &mut reapers, &event_tx),
                Some(MuxAction::Exit) | None => break,
            },
            message = runtime_rx.recv() => match message {
                Some(RuntimeMessage::Prepared { id, name, workspace, worktree, pane }) => if slots.contains_key(&id) {
                    let _ = event_tx.send(MuxEvent::Replace {
                        id,
                        name,
                        workspace,
                        worktree,
                        status: MuxStatus::Starting,
                        pane: *pane,
                    });
                },
                Some(RuntimeMessage::Ready(id)) => if slots.contains_key(&id) {
                    let _ = event_tx.send(MuxEvent::Status { id, status: MuxStatus::Idle });
                },
                Some(RuntimeMessage::Event(id, event)) => {
                    if let Some(slot) = slots.get_mut(&id) {
                        match &event {
                            AgentEvent::ModelChanged { provider, model } => { slot.settings.provider = provider.clone(); slot.settings.model = model.clone(); }
                            AgentEvent::ReasoningChanged { level } => if let Ok(value) = level.parse() { slot.settings.reasoning = value; },
                            _ => {}
                        }
                        if let Some(status) = status_for_agent_event(&event) {
                            let _ = event_tx.send(MuxEvent::Status { id, status });
                        }
                        let _ = event_tx.send(MuxEvent::Ui { id, event: tui_adapter::into_ui_event(event) });
                    }
                }
                Some(RuntimeMessage::AssemblyError(id, error)) => if slots.contains_key(&id) {
                    let _ = event_tx.send(MuxEvent::Status { id, status: MuxStatus::Error });
                    let _ = event_tx.send(MuxEvent::Ui { id, event: tui::UiEvent::Error(error) });
                },
                Some(RuntimeMessage::Stopped(id)) => if slots.contains_key(&id) {
                    let _ = event_tx.send(MuxEvent::Status { id, status: MuxStatus::Error });
                },
                None => break,
            }
        }
    }
    // Stop terminal ownership independently of agent teardown. In particular, a
    // slot may currently be inside uninterruptible spawn_blocking preparation.
    cancel.cancel();
    drop(event_tx);
    for (_, slot) in slots.drain() {
        start_reap(slot, &mut reapers);
    }
    let mut shutdown_task =
        tokio::spawn(async move { while reapers.join_next().await.is_some() {} });

    let terminal_result = match tokio::time::timeout(Duration::from_secs(2), &mut ui_task).await {
        Ok(result) => result.context("join mux terminal task")?,
        Err(_) => {
            ui_task.abort();
            let _ = ui_task.await;
            Ok(())
        }
    };

    if tokio::time::timeout(Duration::from_secs(8), &mut shutdown_task)
        .await
        .is_err()
    {
        // Dropping the reaper JoinSet aborts its Tokio tasks. An already-running
        // blocking closure cannot be force-aborted; its owned RetainedLease will
        // still retain the worktree when that closure eventually returns.
        shutdown_task.abort();
        let _ = shutdown_task.await;
    }
    terminal_result
}

#[allow(clippy::too_many_arguments)]
fn create_slot(
    id: MuxId,
    choice: WorkspaceChoice,
    settings: SlotSettings,
    config: &Config,
    cli: &crate::config::Cli,
    launch: &Path,
    event_tx: &mpsc::UnboundedSender<MuxEvent>,
    runtime_tx: &mpsc::UnboundedSender<RuntimeMessage>,
    slots: &mut HashMap<MuxId, SlotHandle>,
    mutation_coordinator: &WorkspaceMutationCoordinator,
) {
    let (provisional_name, provisional_workspace, provisional_worktree) = match &choice {
        WorkspaceChoice::Directory { path, name } => (name.clone(), path.clone(), false),
        WorkspaceChoice::Worktree { branch, .. } => (branch.clone(), launch.to_path_buf(), true),
    };
    let provisional_pane = pane(
        &settings,
        provisional_workspace.clone(),
        vec![],
        vec![],
        config.tui_minimal,
    );
    let _ = event_tx.send(MuxEvent::Add {
        id,
        name: provisional_name,
        workspace: provisional_workspace,
        worktree: provisional_worktree,
        status: MuxStatus::Starting,
        pane: provisional_pane,
    });

    let cancel = CancellationToken::new();
    let (input_tx, input_rx) = mpsc::unbounded_channel();
    let task_cancel = cancel.clone();
    let runtime_tx = runtime_tx.clone();
    let config = config.clone();
    let task_settings = settings.clone();
    let launch = launch.to_path_buf();
    let no_context_files = cli.no_context_files;
    let deferred = cli.defer_session_sync;
    let mutation_coordinator = mutation_coordinator.clone();
    let task = tokio::spawn(async move {
        let setup_settings = task_settings.clone();
        let setup_config = config.clone();
        let preparation_cancel = task_cancel.clone();
        let preparation = tokio::task::spawn_blocking(move || -> Result<_> {
            anyhow::ensure!(
                !preparation_cancel.is_cancelled(),
                "slot preparation cancelled"
            );
            let (workspace, name, lease) = prepare_workspace(choice, &launch)?;
            let lease = RetainedLease(lease);
            anyhow::ensure!(
                !preparation_cancel.is_cancelled(),
                "slot preparation cancelled"
            );
            let context_bundle = context::load_context_bundle(&workspace, no_context_files, true);
            anyhow::ensure!(
                !preparation_cancel.is_cancelled(),
                "slot preparation cancelled"
            );
            let mut tools = default_registry(ToolConfig::new(&workspace, setup_config.rtk))?;
            // Canonical path identity, rather than creation origin, is the
            // safety boundary: a worktree can also be opened through the
            // direct-directory picker.
            tools.set_execution_gate(mutation_coordinator.gate_for(&workspace));
            let skills = tools
                .skills()
                .map(|catalog| {
                    catalog
                        .entries()
                        .into_iter()
                        .map(|skill| tui::SkillEntry {
                            name: skill.name,
                            description: skill.description,
                        })
                        .collect()
                })
                .unwrap_or_default();
            let context_files = context_bundle
                .display_paths
                .iter()
                .cloned()
                .map(|path| ContextFileEntry { path })
                .collect();
            anyhow::ensure!(
                !preparation_cancel.is_cancelled(),
                "slot preparation cancelled"
            );
            let provider = build_provider_with_auths(
                &setup_settings.provider,
                setup_config.copilot_auth.clone(),
                setup_config.codex_auth.clone(),
            )?;
            anyhow::ensure!(
                !preparation_cancel.is_cancelled(),
                "slot preparation cancelled"
            );
            let store =
                SessionStore::default_for_workspace(&workspace)?.with_deferred_sync(deferred);
            let session = store.create(SessionCreateOptions {
                provider: Some(setup_settings.provider.clone()),
                model: Some(setup_settings.model.clone()),
                ..Default::default()
            })?;
            Ok((
                workspace,
                name,
                lease,
                context_bundle,
                tools,
                skills,
                context_files,
                provider,
                store,
                session,
            ))
        })
        .await;

        let result: Result<()> = match preparation {
            Err(error) => Err(anyhow::Error::new(error).context("join workspace preparation task")),
            Ok(Err(error)) => Err(error),
            Ok(Ok((
                workspace,
                name,
                lease,
                context_bundle,
                tools,
                skills,
                context_files,
                provider,
                store,
                session,
            ))) => {
                let result = async {
                    if task_cancel.is_cancelled() {
                        return Ok(());
                    }
                    let worktree = lease.0.is_some();
                    let ready_pane = pane(
                        &task_settings,
                        workspace.clone(),
                        skills,
                        context_files,
                        config.tui_minimal,
                    );
                    let _ = runtime_tx.send(RuntimeMessage::Prepared {
                        id,
                        name,
                        workspace: workspace.clone(),
                        worktree,
                        pane: Box::new(ready_pane),
                    });
                    let agent = AgentBuilder::new(
                        provider,
                        task_settings.model.clone(),
                        tools,
                        task_cancel.clone(),
                    )
                    .with_reasoning(task_settings.reasoning)
                    .with_project_context(context_bundle.rendered)
                    .with_compaction(config.compaction)
                    .with_subagents(config.subagents, config.rtk)
                    .with_mcp_servers(config.mcp_servers.clone())
                    .with_session(store, session)
                    .with_provider_factory(provider_factory(
                        config.copilot_auth.clone(),
                        config.codex_auth.clone(),
                    ))
                    .build()
                    .await?;
                    if task_cancel.is_cancelled() {
                        return Ok(());
                    }
                    let (agent_tx, mut agent_rx) = mpsc::unbounded_channel();
                    let forward = runtime_tx.clone();
                    let drain = tokio::spawn(async move {
                        while let Some(event) = agent_rx.recv().await {
                            let _ = forward.send(RuntimeMessage::Event(id, event));
                        }
                    });
                    let _ = runtime_tx.send(RuntimeMessage::Ready(id));
                    agent.run(input_rx, agent_tx).await;
                    let _ = drain.await;
                    Ok(())
                }
                .await;
                drop(lease);
                result
            }
        };
        if let Err(error) = result
            && !task_cancel.is_cancelled()
        {
            let _ = runtime_tx.send(RuntimeMessage::AssemblyError(id, format!("{error:#}")));
        }
        let _ = runtime_tx.send(RuntimeMessage::Stopped(id));
    });
    slots.insert(
        id,
        SlotHandle {
            input: input_tx,
            cancel,
            task,
            settings,
        },
    );
}

fn prepare_workspace(
    choice: WorkspaceChoice,
    launch: &Path,
) -> Result<(PathBuf, String, Option<worktree::WorktreeLease>)> {
    match choice {
        WorkspaceChoice::Directory { path, name } => {
            let path = std::fs::canonicalize(&path)
                .with_context(|| format!("resolve directory `{}`", path.display()))?;
            anyhow::ensure!(
                path.is_dir(),
                "workspace is not a directory: {}",
                path.display()
            );
            Ok((path, name, None))
        }
        WorkspaceChoice::Worktree { branch, keep } => {
            let args = WorktreeArgs {
                branch: branch.clone(),
                start_point: None,
                dir: None,
                keep,
                ephemeral: false,
                command: None,
            };
            let lease = worktree::prepare(&args, launch)?;
            Ok((lease.workspace_path().to_path_buf(), branch, Some(lease)))
        }
    }
}

fn close_slot(
    id: MuxId,
    slots: &mut HashMap<MuxId, SlotHandle>,
    reapers: &mut JoinSet<()>,
    events: &mpsc::UnboundedSender<MuxEvent>,
) {
    if let Some(slot) = slots.remove(&id) {
        // Remove first so rendering and routing change in this supervisor turn.
        let _ = events.send(MuxEvent::Remove { id });
        start_reap(slot, reapers);
    }
}

fn start_reap(slot: SlotHandle, reapers: &mut JoinSet<()>) {
    slot.cancel.cancel();
    drop(slot.input);
    reapers.spawn(async move {
        let mut task = slot.task;
        if tokio::time::timeout(Duration::from_secs(6), &mut task)
            .await
            .is_err()
        {
            task.abort();
            let _ = task.await;
        }
    });
}

fn status_for_agent_event(event: &AgentEvent) -> Option<MuxStatus> {
    match event {
        AgentEvent::TextDelta(_)
        | AgentEvent::ReasoningDelta(_)
        | AgentEvent::ToolCallStarted { .. }
        | AgentEvent::Retrying { .. } => Some(MuxStatus::Running),
        AgentEvent::OperationFinished
        | AgentEvent::TurnFinished
        | AgentEvent::CompactionFinished { .. } => Some(MuxStatus::Idle),
        _ => None,
    }
}

fn pane(
    settings: &SlotSettings,
    workspace: PathBuf,
    skills: Vec<tui::SkillEntry>,
    context: Vec<ContextFileEntry>,
    minimal: bool,
) -> AgentPane {
    AgentPane::new_with_minimal(
        &settings.model,
        &settings.provider,
        ProviderArg::ALL.iter().map(ToString::to_string).collect(),
        skills,
        context,
        settings.reasoning.as_str(),
        minimal,
        workspace,
    )
}
fn basename(path: &Path) -> String {
    path.file_name()
        .and_then(|v| v.to_str())
        .unwrap_or("workspace")
        .to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn prepares_and_canonicalizes_directory_workspace() {
        let root = tempdir().unwrap();
        let nested = root.path().join("nested");
        std::fs::create_dir(&nested).unwrap();
        let choice = WorkspaceChoice::Directory {
            path: nested.join(".."),
            name: "project".into(),
        };

        let (workspace, name, lease) = prepare_workspace(choice, root.path()).unwrap();

        assert_eq!(workspace, std::fs::canonicalize(root.path()).unwrap());
        assert_eq!(name, "project");
        assert!(lease.is_none());
    }

    #[test]
    fn mutation_coordinator_keys_every_slot_by_canonical_workspace() {
        let coordinator = WorkspaceMutationCoordinator::default();
        let root = Path::new("/canonical/workspace");
        let shared_a = coordinator.gate_for(root);
        let shared_b = coordinator.clone().gate_for(root);
        let distinct = coordinator.gate_for(Path::new("/canonical/other"));

        assert!(Arc::ptr_eq(&shared_a, &shared_b));
        assert!(!Arc::ptr_eq(&shared_a, &distinct));
    }

    #[test]
    fn operation_completion_restores_idle_without_misclassifying_errors() {
        assert_eq!(
            status_for_agent_event(&AgentEvent::Error("retrying".into())),
            None
        );
        assert_eq!(
            status_for_agent_event(&AgentEvent::OperationFinished),
            Some(MuxStatus::Idle)
        );
        assert_eq!(
            status_for_agent_event(&AgentEvent::CompactionFinished {
                compacted_through: 1,
                summary_bytes: 2,
                auto: false,
                reason: agent::CompactionReason::Manual,
            }),
            Some(MuxStatus::Idle)
        );
        assert_eq!(
            status_for_agent_event(&AgentEvent::TextDelta("working".into())),
            Some(MuxStatus::Running)
        );
        assert_eq!(
            status_for_agent_event(&AgentEvent::Retrying {
                attempt: 1,
                message: "again".into(),
            }),
            Some(MuxStatus::Running)
        );
    }

    #[test]
    fn rejects_a_file_as_directory_workspace() {
        let root = tempdir().unwrap();
        let file = root.path().join("file");
        std::fs::write(&file, "x").unwrap();
        let choice = WorkspaceChoice::Directory {
            path: file,
            name: "file".into(),
        };

        let error = match prepare_workspace(choice, root.path()) {
            Ok(_) => panic!("file workspace unexpectedly succeeded"),
            Err(error) => error,
        };

        assert!(error.to_string().contains("workspace is not a directory"));
    }
}
