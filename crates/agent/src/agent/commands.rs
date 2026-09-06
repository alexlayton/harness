use super::persistence::{ui_snapshot_entries, usage_event};
use super::{
    Agent, AgentEvent, AgentSessionState, CompactionReason, InputMessage, SessionListItem,
    TurnControl, TurnError, send,
};
use llm::Provider;
use session::{ExportOptions, SessionCreateOptions, SessionEvent, export_jsonl, snapshot_entries};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tools::SkillEntry;

impl Agent {
    pub(crate) fn handle_new_session(&mut self, events: &mpsc::UnboundedSender<AgentEvent>) {
        let Some(store) = self.session.as_ref().map(|state| state.store.clone()) else {
            self.history.clear();
            send(
                events,
                AgentEvent::Notice("Started a new conversation".into()),
            );
            return;
        };
        let session = match store.create(SessionCreateOptions {
            provider: Some(self.provider.name().to_owned()),
            model: Some(self.model.clone()),
            ..SessionCreateOptions::default()
        }) {
            Ok(session) => session,
            Err(error) => {
                send(
                    events,
                    AgentEvent::Error(format!("could not create session: {error}")),
                );
                return;
            }
        };
        let id = session.id().to_string();
        let parent_session_id = session.id();
        let title = session.metadata.title.clone();
        self.history.clear();
        self.last_context_tokens = None;
        self.session = Some(AgentSessionState { store, session });
        if let Some(runner) = &self.subagent_runner {
            runner.update_parent_session(Some(parent_session_id));
        }
        send(
            events,
            AgentEvent::SessionChanged {
                id,
                title,
                loaded: false,
            },
        );
        send(
            events,
            AgentEvent::SessionSnapshot {
                entries: Vec::new(),
            },
        );
        if let Some(state) = self.session.as_ref() {
            send(events, usage_event(&state.session.metadata.usage));
        }
        send(events, self.context_usage_event());
        send(
            events,
            AgentEvent::Notice("Started a new conversation".into()),
        );
    }

    pub(crate) fn handle_load_session(
        &mut self,
        selector: String,
        events: &mpsc::UnboundedSender<AgentEvent>,
    ) {
        let Some(store) = self.session.as_ref().map(|state| state.store.clone()) else {
            send(events, AgentEvent::Error("sessions are not enabled".into()));
            return;
        };
        let mut session = match store.load(&selector) {
            Ok(session) => session,
            Err(error) => {
                send(
                    events,
                    AgentEvent::Error(format!("could not load session: {error}")),
                );
                return;
            }
        };
        if !session
            .file_path()
            .is_some_and(|path| store.is_path_in_store(path))
        {
            match store.adopt(&session) {
                Ok(adopted) => session = adopted,
                Err(error) => {
                    send(
                        events,
                        AgentEvent::Error(format!("could not adopt loaded session: {error}")),
                    );
                    return;
                }
            }
        }
        if let Err(error) = store.repair_incomplete_tool_calls(&mut session) {
            send(
                events,
                AgentEvent::Error(format!("could not repair loaded session: {error}")),
            );
            return;
        }
        let id = session.id().to_string();
        let parent_session_id = session.id();
        let title = session.metadata.title.clone();
        self.history = session.context_messages();
        self.last_context_tokens = None;
        let snapshot = ui_snapshot_entries(snapshot_entries(&session));
        self.session = Some(AgentSessionState { store, session });
        if let Some(runner) = &self.subagent_runner {
            runner.update_parent_session(Some(parent_session_id));
        }
        send(
            events,
            AgentEvent::SessionChanged {
                id,
                title,
                loaded: true,
            },
        );
        send(events, AgentEvent::SessionSnapshot { entries: snapshot });
        if let Some(state) = self.session.as_ref() {
            send(events, usage_event(&state.session.metadata.usage));
        }
        send(events, self.context_usage_event());
        send(
            events,
            AgentEvent::Notice(format!(
                "Loaded session; active model remains {} · {}",
                self.provider.name(),
                self.model
            )),
        );
    }

    pub(crate) fn handle_list_sessions(&self, events: &mpsc::UnboundedSender<AgentEvent>) {
        let Some(store) = self.session.as_ref().map(|state| state.store.clone()) else {
            send(events, AgentEvent::Error("sessions are not enabled".into()));
            return;
        };
        match store.list() {
            Ok(entries) => send(
                events,
                AgentEvent::SessionList {
                    sessions: entries
                        .into_iter()
                        .filter(|entry| entry.has_conversation)
                        .map(|entry| SessionListItem {
                            id: entry.id.to_string(),
                            short_id: entry.short_id,
                            title: entry.title,
                            updated_at: entry.updated_at,
                            workspace: entry.workspace_root.display().to_string(),
                            provider: entry.provider,
                            model: entry.model,
                        })
                        .collect(),
                },
            ),
            Err(error) => send(
                events,
                AgentEvent::Error(format!("could not list sessions: {error}")),
            ),
        }
    }

    pub(crate) fn handle_export_session(
        &self,
        destination: Option<String>,
        events: &mpsc::UnboundedSender<AgentEvent>,
    ) {
        let Some(session) = self.session.as_ref().map(|state| state.session.clone()) else {
            send(events, AgentEvent::Error("sessions are not enabled".into()));
            return;
        };
        let destination = destination.map(PathBuf::from);
        match export_jsonl(&session, destination.as_deref(), &ExportOptions::default()) {
            Ok(path) => {
                let path = path.display().to_string();
                send(events, AgentEvent::SessionExported { path: path.clone() });
                send(
                    events,
                    AgentEvent::Notice(format!("Exported session to {path}")),
                );
            }
            Err(error) => send(
                events,
                AgentEvent::Error(format!("could not export session: {error}")),
            ),
        }
    }

    pub(crate) async fn handle_compact_session(
        &mut self,
        events: &mpsc::UnboundedSender<AgentEvent>,
        cancel: &CancellationToken,
    ) -> Result<(), TurnError> {
        if self.session.is_none() {
            send(
                events,
                AgentEvent::Notice("compaction is unavailable without a session".into()),
            );
            return Ok(());
        }
        self.compact_and_reload(events, cancel, CompactionReason::Manual, 0)
            .await
            .map(|_| ())
    }

    /// Boundary wrapper for manual compaction: same deferred-sync flush and
    /// quarantine policy as turns, with exactly one terminal event owned by
    /// the run loop.
    pub(crate) async fn handle_compact_session_boundary(
        &mut self,
        events: &mpsc::UnboundedSender<AgentEvent>,
        input: &mut mpsc::UnboundedReceiver<InputMessage>,
    ) -> TurnControl {
        let cancel = self.cancel.child_token();
        let application = self.cancel.clone();
        let mut buffered = VecDeque::new();
        let mut input_open = self.input_open;
        let mut interrupted = false;
        let outcome = {
            let mut application_open = true;
            let operation = self.handle_compact_session(events, &cancel);
            tokio::pin!(operation);
            loop {
                tokio::select! {
                    biased;
                    result = &mut operation => break result,
                    _ = application.cancelled(), if application_open => {
                        application_open = false;
                        cancel.cancel();
                    }
                    message = input.recv(), if input_open => match message {
                        Some(InputMessage::Interrupt) => {
                            interrupted = true;
                            cancel.cancel();
                        }
                        Some(message) => buffered.push_back(message),
                        None => input_open = false,
                    },
                }
            }
        };
        self.input_open = input_open;
        self.queued.extend(buffered);
        if self.flush_deferred_sync(events).is_err() {
            return TurnControl::Quarantine;
        }
        if interrupted && !application.is_cancelled() {
            // `handle_compact_session` has already observed the cancelled
            // operation and deliberately persisted no summary.
            return TurnControl::Continue;
        }
        match outcome {
            Ok(()) => TurnControl::Continue,
            Err(TurnError::Shutdown) => TurnControl::Shutdown,
            Err(TurnError::Persist(_)) => TurnControl::Quarantine,
        }
    }

    /// Boundary wrapper for model changes: persist-first atomic commit
    /// (see `handle_set_model`), deferred-sync flush, and quarantine on
    /// persistence failure — the same boundary policy as turns.
    pub(crate) async fn handle_set_model_boundary(
        &mut self,
        provider: Option<String>,
        model: String,
        events: &mpsc::UnboundedSender<AgentEvent>,
    ) -> TurnControl {
        // Resolve without mutating live state; `handle_set_model` persists
        // first and only commits after persistence succeeds.
        let outcome = self.handle_set_model(provider, model, events).await;
        if self.flush_deferred_sync(events).is_err() {
            return TurnControl::Quarantine;
        }
        match outcome {
            Ok(()) => TurnControl::Continue,
            Err(TurnError::Shutdown) => TurnControl::Shutdown,
            Err(TurnError::Persist(_)) => TurnControl::Quarantine,
        }
    }

    pub(crate) async fn handle_set_model(
        &mut self,
        provider: Option<String>,
        model: String,
        events: &mpsc::UnboundedSender<AgentEvent>,
    ) -> Result<(), TurnError> {
        // Resolve the candidate provider and canonical model name without
        // mutating live state; commit only after persistence succeeds, so
        // a failed `ModelChange` persist leaves parent and subagent
        // selection unchanged (AGENT-4 atomicity).
        let requested = provider.unwrap_or_else(|| self.provider.name().to_owned());
        let current = self.provider.name().to_owned();
        let next_provider = if requested.eq_ignore_ascii_case(&current) {
            None
        } else {
            let Some(factory) = &self.provider_factory else {
                send(
                    events,
                    AgentEvent::Error("provider switching is unavailable".into()),
                );
                return Ok(());
            };
            match factory(&requested) {
                Ok(provider) => Some(provider),
                Err(error) => {
                    send(events, AgentEvent::Error(error.to_string()));
                    return Ok(());
                }
            }
        };
        let canonical = next_provider
            .as_ref()
            .map(|provider| provider.name().to_owned())
            .unwrap_or_else(|| current.clone());
        // Persist first: only after this succeeds do live parent/provider
        // state and the subagent runner move.
        self.persist_event(
            SessionEvent::ModelChange {
                provider: canonical.clone(),
                model: model.clone(),
            },
            events,
        )?;
        if let Some(provider) = next_provider {
            self.provider = provider;
        }
        self.model = model.clone();
        // Future subagents must follow the parent's active selection; a
        // failed switch already returned above, so children never see a
        // half-applied state. Running children keep their own snapshot.
        if let Some(runner) = &self.subagent_runner {
            runner.update_model(self.provider.clone(), self.model.clone());
        }
        send(
            events,
            AgentEvent::ModelChanged {
                provider: canonical.clone(),
                model: model.clone(),
            },
        );
        send(
            events,
            AgentEvent::Notice(format!("Using {canonical} · {model}")),
        );
        // A different model may have a different context window and stale
        // token counts; reset both so the next trigger re-baselines. Metadata
        // is fetched in the background so a slow catalogue cannot block the
        // command loop, and one response supplies both UI models and context.
        self.last_context_tokens = None;
        self.context_window = self.compaction.resolved_window(0);
        if let Some(metadata_tx) = &self.model_metadata_tx {
            spawn_model_metadata(
                metadata_tx.clone(),
                self.provider.clone(),
                canonical,
                self.model.clone(),
                self.cancel.clone(),
            );
        }
        Ok(())
    }

    pub(crate) fn handle_set_reasoning(
        &mut self,
        level: String,
        events: &mpsc::UnboundedSender<AgentEvent>,
    ) {
        let reasoning = match level.parse::<llm::ReasoningPolicy>() {
            Ok(reasoning) => reasoning,
            Err(error) => {
                send(events, AgentEvent::Error(error));
                return;
            }
        };
        self.reasoning = reasoning;
        if let Some(runner) = &self.subagent_runner {
            runner.update_reasoning(reasoning);
        }
        send(
            events,
            AgentEvent::ReasoningChanged {
                level: reasoning.to_string(),
            },
        );
        send(
            events,
            AgentEvent::Notice(format!("Reasoning effort: {reasoning}")),
        );
    }

    /// Fetch subscription allowance usage from the provider active when the
    /// command was submitted. The request runs in the background so a slow
    /// account endpoint cannot block subsequent input processing.
    pub(crate) fn handle_subscription_usage(&self, events: &mpsc::UnboundedSender<AgentEvent>) {
        let provider = self.provider.clone();
        let provider_name = provider.name().to_owned();
        let events = events.clone();
        tokio::spawn(async move {
            match provider.subscription_usage().await {
                Ok(Some(usage)) => send(
                    &events,
                    AgentEvent::SubscriptionUsageLoaded {
                        provider: provider_name,
                        usage,
                    },
                ),
                Ok(None) => send(
                    &events,
                    AgentEvent::Notice(format!(
                        "subscription usage is not available for {provider_name}"
                    )),
                ),
                Err(error) => send(
                    &events,
                    AgentEvent::Error(format!(
                        "could not fetch {provider_name} subscription usage: {error}"
                    )),
                ),
            }
        });
    }

    pub(crate) fn handle_list_models(
        &self,
        provider: String,
        events: &mpsc::UnboundedSender<AgentEvent>,
    ) {
        let Some(factory) = &self.provider_factory else {
            send(
                events,
                AgentEvent::Notice(
                    "could not fetch model list: provider construction is unavailable".into(),
                ),
            );
            return;
        };
        let provider = match factory(&provider) {
            Ok(provider) => provider,
            Err(error) => {
                send(
                    events,
                    AgentEvent::Notice(format!("could not fetch model list: {error}")),
                );
                return;
            }
        };
        spawn_model_list(provider.name().to_owned(), provider, events.clone());
    }

    /// Reply to `/skills` with the discovered-skill view: invocable skills
    /// first, then discovery diagnostics so broken skills are visible to the
    /// user (they never reach the model prompt).
    pub(crate) fn handle_list_skills(&self, events: &mpsc::UnboundedSender<AgentEvent>) {
        let Some(catalog) = self.tools.skills() else {
            send(
                events,
                AgentEvent::SkillsLoaded {
                    skills: Vec::new(),
                    diagnostics: Vec::new(),
                    empty: true,
                },
            );
            return;
        };
        let empty = catalog.is_empty();
        send(
            events,
            AgentEvent::SkillsLoaded {
                skills: catalog
                    .skills
                    .iter()
                    .map(|skill| SkillEntry {
                        name: skill.name.clone(),
                        description: skill.description.clone(),
                    })
                    .collect(),
                diagnostics: catalog
                    .diagnostics
                    .iter()
                    .map(|diagnostic| match &diagnostic.path {
                        Some(path) => format!("{}: {}", path.display(), diagnostic.message),
                        None => diagnostic.message.clone(),
                    })
                    .collect(),
                empty,
            },
        );
    }

    /// Start a turn from a skill's instructions: the `SKILL.md` body without
    /// frontmatter, prefixed with a line naming the skill so both the model
    /// and the session transcript show what was invoked. Runs through the
    /// single shared turn executor, so shutdown propagates, persistence
    /// failures quarantine, deferred writes flush, and exactly one terminal
    /// event is emitted — identical to a normal user message.
    pub(crate) async fn handle_invoke_skill(
        &mut self,
        name: String,
        events: &mpsc::UnboundedSender<AgentEvent>,
        input: &mut mpsc::UnboundedReceiver<InputMessage>,
    ) -> TurnControl {
        let found = self.tools.skills().and_then(|catalog| {
            catalog
                .invocable()
                .into_iter()
                .find(|skill| skill.name.eq_ignore_ascii_case(&name))
                .map(|skill| (skill.file_path.clone(), skill.name.clone()))
        });
        let Some((file_path, name)) = found else {
            send(events, AgentEvent::Error(format!("unknown skill: {name}")));
            return TurnControl::Continue;
        };
        let raw = match std::fs::read_to_string(&file_path) {
            Ok(raw) => raw,
            Err(error) => {
                send(
                    events,
                    AgentEvent::Error(format!("could not read {name}: {error}")),
                );
                return TurnControl::Continue;
            }
        };
        let (_, body) = tools::parse_frontmatter(&raw);
        let body = body.trim();
        if body.is_empty() {
            send(events, AgentEvent::Error(format!("skill {name} is empty")));
            return TurnControl::Continue;
        }
        self.execute_turn(format!("/{name}\n\n{body}"), events, input)
            .await
    }
}

/// Host-owned provider construction used for model selection commands.
pub type ProviderFactory =
    Arc<dyn Fn(&str) -> anyhow::Result<Arc<dyn Provider>> + Send + Sync + 'static>;

/// Fetch model metadata for the active selection on a bounded background
/// task. The run loop consumes the result and derives both context-window and
/// model-list updates from this one request.
pub(crate) fn spawn_model_metadata(
    sender: mpsc::UnboundedSender<(String, String, Vec<llm::ModelInfo>)>,
    provider: Arc<dyn Provider>,
    provider_name: String,
    model: String,
    cancel: CancellationToken,
) {
    tokio::spawn(async move {
        let models = tokio::select! {
            result = tokio::time::timeout(
                std::time::Duration::from_secs(5),
                provider.list_models(),
            ) => result.ok().and_then(Result::ok),
            _ = cancel.cancelled() => None,
        };
        if let Some(models) = models {
            let _ = sender.send((provider_name, model, models));
        }
    });
}

/// Fetch a provider's model list on a background task, reporting
/// `AgentEvent::ModelList` on success and a notice on failure. Shared by the
/// `/model` and `/models` handlers.
pub fn spawn_model_list(
    provider_name: String,
    provider: Arc<dyn Provider>,
    events: mpsc::UnboundedSender<AgentEvent>,
) {
    tokio::spawn(async move {
        match tokio::time::timeout(std::time::Duration::from_secs(5), provider.list_models()).await
        {
            Ok(Ok(models)) => send(
                &events,
                AgentEvent::ModelList {
                    provider: provider_name,
                    models,
                },
            ),
            Ok(Err(error)) => send(
                &events,
                AgentEvent::Notice(format!("could not fetch model list: {error}")),
            ),
            Err(_) => send(
                &events,
                AgentEvent::Notice("could not fetch model list: request timed out".into()),
            ),
        }
    });
}
