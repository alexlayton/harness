use super::persistence::{ui_snapshot_entries, usage_event};
use super::{
    Agent, AgentEvent, AgentSessionState, CompactionReason, SessionListItem, TurnError, send,
};
use crate::config::{build_provider_with_auth, save_settings};
use crate::tools::SkillEntry;
use auth::CopilotAuth;
use llm::Provider;
use session::{ExportOptions, SessionCreateOptions, SessionEvent, export_jsonl, snapshot_entries};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tui::InputMessage;

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
            send(events, AgentEvent::Error("sessions are not enabled".into()));
            return Ok(());
        }
        self.compact_and_reload(events, cancel, CompactionReason::Manual)
            .await
            .map(|_| ())
    }

    pub(crate) async fn handle_set_model(
        &mut self,
        provider: Option<String>,
        model: String,
        events: &mpsc::UnboundedSender<AgentEvent>,
    ) {
        let needs_auth = provider
            .as_deref()
            .and_then(crate::config::ProviderArg::from_name)
            == Some(crate::config::ProviderArg::GithubCopilot)
            || self.provider.name() == "github-copilot";
        let mut auth = self.copilot_auth.clone();
        if needs_auth && auth.is_none() {
            match CopilotAuth::from_default() {
                Ok(value) => {
                    let value = Arc::new(value);
                    self.copilot_auth = Some(value.clone());
                    auth = Some(value);
                }
                Err(error) => {
                    send(events, AgentEvent::Error(error.to_string()));
                    return;
                }
            }
        }
        self.handle_set_model_with_factory(
            provider,
            model,
            events,
            Box::new(move |name| {
                let copilot_auth = if crate::config::ProviderArg::from_name(name)
                    == Some(crate::config::ProviderArg::GithubCopilot)
                {
                    auth.clone()
                } else {
                    None
                };
                build_provider_with_auth(name, copilot_auth)
            }),
        );
        // A different model may have a different context window and stale
        // token counts; reset both so the next trigger re-baselines.
        self.last_context_tokens = None;
        self.refresh_context_window().await;
    }

    pub(crate) fn handle_set_model_with_factory(
        &mut self,
        provider: Option<String>,
        model: String,
        events: &mpsc::UnboundedSender<AgentEvent>,
        factory: ProviderFactory,
    ) {
        let explicit_provider = provider.is_some();
        let requested = provider.unwrap_or_else(|| self.provider.name().to_owned());
        let known_provider = crate::config::ProviderArg::ALL
            .iter()
            .find(|known| known.to_string().eq_ignore_ascii_case(&requested));
        let canonical = known_provider
            .map(ToString::to_string)
            .unwrap_or_else(|| requested.clone());
        let current = self.provider.name().to_owned();
        let provider_changed =
            (explicit_provider && known_provider.is_none()) || current != canonical;
        let next_provider = if provider_changed {
            match factory(&canonical) {
                Ok(provider) => Some(provider),
                Err(error) => {
                    send(events, AgentEvent::Error(error.to_string()));
                    return;
                }
            }
        } else {
            None
        };

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
        if self
            .persist_event(
                SessionEvent::ModelChange {
                    provider: canonical.clone(),
                    model: model.clone(),
                },
                events,
            )
            .is_err()
        {
            return;
        }
        if let Err(error) = save_settings(&canonical, &model) {
            tracing::warn!(error = %error, "could not persist model settings");
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
        spawn_model_list(canonical, self.provider.clone(), events.clone());
    }

    pub(crate) fn handle_list_models(
        &self,
        provider: String,
        events: &mpsc::UnboundedSender<AgentEvent>,
    ) {
        let provider_name = crate::config::ProviderArg::ALL
            .iter()
            .find(|known| known.to_string().eq_ignore_ascii_case(&provider))
            .map(ToString::to_string)
            .unwrap_or(provider.clone());
        let auth = if provider_name == "github-copilot" {
            self.copilot_auth.clone()
        } else {
            None
        };
        let provider = match build_provider_with_auth(&provider_name, auth) {
            Ok(provider) => provider,
            Err(error) => {
                send(
                    events,
                    AgentEvent::Notice(format!("could not fetch model list: {error}")),
                );
                return;
            }
        };
        spawn_model_list(provider_name, provider, events.clone());
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
    /// and the session transcript show what was invoked.
    pub(crate) async fn handle_invoke_skill(
        &mut self,
        name: String,
        events: &mpsc::UnboundedSender<AgentEvent>,
        input: &mut mpsc::UnboundedReceiver<InputMessage>,
    ) {
        let found = self.tools.skills().and_then(|catalog| {
            catalog
                .invocable()
                .into_iter()
                .find(|skill| skill.name.eq_ignore_ascii_case(&name))
                .map(|skill| (skill.file_path.clone(), skill.name.clone()))
        });
        let Some((file_path, name)) = found else {
            send(events, AgentEvent::Error(format!("unknown skill: {name}")));
            return;
        };
        let raw = match std::fs::read_to_string(&file_path) {
            Ok(raw) => raw,
            Err(error) => {
                send(
                    events,
                    AgentEvent::Error(format!("could not read {name}: {error}")),
                );
                return;
            }
        };
        let (_, body) = tools::parse_frontmatter(&raw);
        let body = body.trim();
        if body.is_empty() {
            send(events, AgentEvent::Error(format!("skill {name} is empty")));
            return;
        }
        let turn_cancel = CancellationToken::new();
        let result = self
            .run_turn(format!("/{name}\n\n{body}"), events, input, &turn_cancel)
            .await;
        match result {
            Err(TurnError::Shutdown) => {}
            Err(TurnError::Persist(_)) | Ok(()) => {}
        }
    }
}

/// Factory used to build providers when the model/provider selection changes.
/// A `Box<dyn Fn>` (rather than a generic) keeps the call site simple; the
/// dispatch cost is negligible because it only runs when `/model` is used.
type ProviderFactory = Box<dyn Fn(&str) -> anyhow::Result<Arc<dyn Provider>>>;

/// Fetch a provider's model list on a background task, reporting
/// `AgentEvent::ModelList` on success and a notice on failure.  Shared by the
/// startup fetch in `main` and the `/model` and `/models` handlers.
pub fn spawn_model_list(
    provider_name: String,
    provider: Arc<dyn Provider>,
    events: mpsc::UnboundedSender<AgentEvent>,
) {
    tokio::spawn(async move {
        match provider.list_models().await {
            Ok(models) => send(
                &events,
                AgentEvent::ModelList {
                    provider: provider_name,
                    models,
                },
            ),
            Err(error) => send(
                &events,
                AgentEvent::Notice(format!("could not fetch model list: {error}")),
            ),
        }
    });
}
