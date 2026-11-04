//! One-shot prewalk handoff from the planning model to a target model.
//!
//! While armed, the planning model investigates, writes a visible plan, and
//! starts the change. The first `edit` or `write` call, including one nested
//! in `codemode`, ends the planning phase: after that tool round is persisted,
//! the agent switches to the target model and nudges it to continue the plan.
//! Harness has no todo tool, so the plan must be visible assistant text; opaque
//! or provider-specific reasoning does not survive a cross-provider switch.

use super::turn::TurnControl;
use super::{Agent, AgentEvent, TurnError, send};
use llm::ReasoningPolicy;
use std::fmt;
use tokio::sync::mpsc;

/// Result of returning to the planning selection after a handoff.
pub(crate) enum RestoreOutcome {
    /// Nothing to restore, or the restore succeeded.
    Ready,
    /// The switch was refused (and reported); the restore point is kept.
    Rejected,
    /// The loop must stop (shutdown or quarantine).
    Stop(TurnControl),
}

/// Appended to the system prompt only while a handoff is pending.
const PLANNING_NUDGE: &str = "\n\nPrewalk: you are the planning model for this task. Investigate first. Before your first edit or write call, state a short numbered implementation plan in your visible reply: files to change, ordered steps, and how to verify. Then make the first edit. After that tool round, the implementation model (possibly with a different reasoning effort) continues from your plan and this conversation, so keep the plan concrete and self-contained.";

/// In-memory note for the target model, like the turn recovery notes. The
/// durable session records the model change itself.
const HANDOFF_NOTE: &str = "[system note: prewalk handoff. You are now the implementation model for this task. The previous model investigated the workspace, wrote the plan above, and started the change. Follow that plan: finish the remaining steps, keep new changes consistent with the edits already made, run the verification it describes, and report the result. Do not restart the investigation unless the plan proves wrong.]";

/// A prewalk target as `[<provider>:]<model>`. `None` keeps the provider that
/// is active at handoff time.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrewalkTarget {
    pub provider: Option<String>,
    pub model: String,
}

impl fmt::Display for PrewalkTarget {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.provider {
            Some(provider) => write!(formatter, "{provider}:{}", self.model),
            None => formatter.write_str(&self.model),
        }
    }
}

/// Host-resolved prewalk configuration for one agent.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PrewalkSettings {
    /// Persisted preference: arm each new conversation started with `/new`.
    pub enabled: bool,
    /// Model to hand off to. Arming without a target is ignored.
    pub target: Option<PrewalkTarget>,
    /// Arm the startup conversation. Hosts derive this from `enabled` for new
    /// sessions and from explicit requests for resumed ones.
    pub armed: bool,
    /// Reasoning for the target. `None` keeps the active reasoning policy.
    pub reasoning: Option<ReasoningPolicy>,
}

/// Which preference a [`AgentEvent::PrewalkChanged`] reports. Hosts may
/// persist `Enabled`, `Target`, and `Reasoning`; `Session` is live state only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrewalkChange {
    /// Startup, `/new`, `/load`, or another live-only arm-state change.
    Session,
    /// `/prewalk on|off` changed the preference and this conversation.
    Enabled,
    /// `/prewalk-model` changed the target.
    Target,
    /// `/prewalk-reasoning` changed the target reasoning.
    Reasoning,
}

/// Planning selection that `/new` returns to after a handoff.
#[derive(Clone, Debug)]
pub(crate) struct PrewalkRestore {
    provider: String,
    model: String,
    reasoning: ReasoningPolicy,
}

/// Live prewalk state owned by the agent.
#[derive(Clone, Debug, Default)]
pub(crate) struct PrewalkState {
    enabled: bool,
    target: Option<PrewalkTarget>,
    armed: bool,
    reasoning: Option<ReasoningPolicy>,
    /// Planning selection restored by `/new` after a handoff, until the user
    /// selects another model or reasoning policy explicitly.
    restore: Option<PrewalkRestore>,
}

impl PrewalkState {
    pub(crate) fn new(settings: PrewalkSettings) -> Self {
        let armed = settings.armed && settings.target.is_some();
        Self {
            enabled: settings.enabled,
            target: settings.target,
            armed,
            reasoning: settings.reasoning,
            restore: None,
        }
    }

    /// Whether startup has state worth reporting to the frontend.
    pub(crate) fn is_configured(&self) -> bool {
        self.armed || self.target.is_some() || self.reasoning.is_some()
    }

    /// A user model or reasoning selection takes ownership of the session.
    pub(crate) fn forget_restore(&mut self) {
        self.restore = None;
    }
}

/// Whether a tool call ends the planning phase. Unlike a todo gate, the
/// call does not need to succeed: the attempt shows the plan has settled.
pub(crate) fn triggers_handoff(name: &str) -> bool {
    matches!(name, "edit" | "write")
}

impl Agent {
    /// Replace the startup prewalk configuration.
    pub fn with_prewalk(mut self, settings: PrewalkSettings) -> Self {
        self.prewalk = PrewalkState::new(settings);
        self
    }

    /// System-prompt suffix for the current request.
    pub(crate) fn prewalk_prompt_suffix(&self) -> &'static str {
        if self.prewalk.armed {
            PLANNING_NUDGE
        } else {
            ""
        }
    }

    pub(crate) fn send_prewalk_state(
        &self,
        change: PrewalkChange,
        events: &mpsc::UnboundedSender<AgentEvent>,
    ) {
        send(
            events,
            AgentEvent::PrewalkChanged {
                armed: self.prewalk.armed,
                target: self.prewalk.target.as_ref().map(ToString::to_string),
                reasoning: self
                    .prewalk
                    .reasoning
                    .map(|reasoning| reasoning.to_string()),
                change,
            },
        );
    }

    /// `/prewalk on|off`: set the preference and arm or cancel this
    /// conversation's handoff. After a handoff, `on` first returns to the
    /// planning selection so the new cycle does not plan on the target.
    /// Turning it off never switches models back.
    pub(crate) async fn handle_set_prewalk(
        &mut self,
        enabled: bool,
        events: &mpsc::UnboundedSender<AgentEvent>,
    ) -> TurnControl {
        if enabled && self.prewalk.target.is_none() {
            send(
                events,
                AgentEvent::Error(
                    "prewalk needs a target model; set one with /prewalk-model [<provider>:]<model>"
                        .into(),
                ),
            );
            return TurnControl::Continue;
        }
        if enabled {
            match self.restore_prewalk_planning(events).await {
                RestoreOutcome::Ready => {}
                // Already reported; arming now would plan on the target.
                RestoreOutcome::Rejected => return TurnControl::Continue,
                RestoreOutcome::Stop(control) => return control,
            }
        }
        self.prewalk.enabled = enabled;
        self.prewalk.armed = enabled;
        self.send_prewalk_state(PrewalkChange::Enabled, events);
        TurnControl::Continue
    }

    /// `/prewalk-model`: validate and store the handoff target. A pending
    /// handoff keeps its arm state and uses the new target.
    pub(crate) fn handle_set_prewalk_model(
        &mut self,
        provider: Option<String>,
        model: String,
        events: &mpsc::UnboundedSender<AgentEvent>,
    ) {
        let model = model.trim().to_owned();
        if model.is_empty() {
            send(
                events,
                AgentEvent::Error("prewalk model must not be blank".into()),
            );
            return;
        }
        if self.secret_masker.mask_text(&model) != model
            || provider
                .as_deref()
                .is_some_and(|name| self.secret_masker.mask_text(name) != name)
        {
            send(
                events,
                AgentEvent::Error(
                    "provider and model names cannot contain a configured secret".into(),
                ),
            );
            return;
        }
        // Resolve the provider now so a typo fails at the command instead
        // of at the handoff, and store its canonical name.
        let provider = match provider {
            Some(name) if !name.eq_ignore_ascii_case(self.provider.name()) => {
                let Some(factory) = &self.provider_factory else {
                    send(
                        events,
                        AgentEvent::Error("provider switching is unavailable".into()),
                    );
                    return;
                };
                match factory(&name) {
                    Ok(candidate) => Some(candidate.name().to_owned()),
                    Err(error) => {
                        send(events, AgentEvent::Error(error.to_string()));
                        return;
                    }
                }
            }
            Some(_) => Some(self.provider.name().to_owned()),
            None => None,
        };
        self.prewalk.target = Some(PrewalkTarget { provider, model });
        self.send_prewalk_state(PrewalkChange::Target, events);
    }

    /// `/prewalk-reasoning`: `None` keeps the active reasoning at handoff.
    pub(crate) fn handle_set_prewalk_reasoning(
        &mut self,
        level: Option<String>,
        events: &mpsc::UnboundedSender<AgentEvent>,
    ) {
        let reasoning = match level
            .map(|level| level.parse::<ReasoningPolicy>())
            .transpose()
        {
            Ok(reasoning) => reasoning,
            Err(error) => {
                send(events, AgentEvent::Error(error));
                return;
            }
        };
        self.prewalk.reasoning = reasoning;
        self.send_prewalk_state(PrewalkChange::Reasoning, events);
    }

    /// Before `/new`, return to the planning selection when the preference
    /// is enabled; otherwise a later `/prewalk on` must not reach back into
    /// an earlier conversation.
    pub(crate) async fn restore_prewalk_for_new_session(
        &mut self,
        events: &mpsc::UnboundedSender<AgentEvent>,
    ) -> RestoreOutcome {
        if self.prewalk.enabled {
            self.restore_prewalk_planning(events).await
        } else {
            self.prewalk.forget_restore();
            RestoreOutcome::Ready
        }
    }

    /// Return to the planning model and reasoning after a handoff, unless the
    /// user replaced them. Like the handoff, this is not a user selection, so
    /// it is announced as [`AgentEvent::PrewalkRestored`].
    async fn restore_prewalk_planning(
        &mut self,
        events: &mpsc::UnboundedSender<AgentEvent>,
    ) -> RestoreOutcome {
        // Keep the restore point until the switch succeeds so a rejected
        // switch can be retried instead of losing the planning selection.
        let Some(restore) = self.prewalk.restore.clone() else {
            return RestoreOutcome::Ready;
        };
        let same_model = restore.provider.eq_ignore_ascii_case(self.provider.name())
            && restore.model == self.model;
        if !same_model {
            // Mirror the `/model` boundary: flush deferred sync on every
            // outcome, and quarantine when durability is uncertain.
            let switched = self
                .switch_model_for_prewalk(Some(restore.provider), restore.model, events)
                .await;
            if !matches!(switched, Ok(true)) {
                if self.flush_deferred_sync(events).is_err() {
                    return RestoreOutcome::Stop(TurnControl::Quarantine);
                }
                return match switched {
                    Err(TurnError::Shutdown) => RestoreOutcome::Stop(TurnControl::Shutdown),
                    Err(TurnError::Persist(_)) => RestoreOutcome::Stop(TurnControl::Quarantine),
                    // The rejected switch is already reported; stay here.
                    Ok(_) => RestoreOutcome::Rejected,
                };
            }
        }
        self.prewalk.restore = None;
        let reasoning = (restore.reasoning != self.reasoning).then_some(restore.reasoning);
        if let Some(reasoning) = reasoning {
            self.set_reasoning_policy(reasoning);
        }
        if !same_model || reasoning.is_some() {
            send(
                events,
                AgentEvent::PrewalkRestored {
                    provider: self.provider.name().to_owned(),
                    model: self.model.clone(),
                    reasoning: reasoning.map(|reasoning| reasoning.to_string()),
                },
            );
        }
        RestoreOutcome::Ready
    }

    /// `/new` re-arms from the preference; a consumed handoff is not
    /// inherited. `allow_arm` is false when the planning restore was rejected.
    pub(crate) fn rearm_prewalk_for_new_session(
        &mut self,
        events: &mpsc::UnboundedSender<AgentEvent>,
        allow_arm: bool,
    ) {
        let armed = allow_arm && self.prewalk.enabled && self.prewalk.target.is_some();
        if armed || self.prewalk.armed {
            self.prewalk.armed = armed;
            self.send_prewalk_state(PrewalkChange::Session, events);
        }
    }

    /// Loaded sessions are never armed automatically.
    pub(crate) fn disarm_prewalk_for_loaded_session(
        &mut self,
        events: &mpsc::UnboundedSender<AgentEvent>,
    ) {
        self.prewalk.restore = None;
        if self.prewalk.armed {
            self.prewalk.armed = false;
            self.send_prewalk_state(PrewalkChange::Session, events);
        }
    }

    /// Perform the one-shot handoff after a persisted tool round that called
    /// `edit` or `write`. A failed switch disarms and keeps the current model.
    pub(crate) async fn prewalk_handoff(
        &mut self,
        events: &mpsc::UnboundedSender<AgentEvent>,
    ) -> Result<(), TurnError> {
        if !self.prewalk.armed {
            return Ok(());
        }
        self.prewalk.armed = false;
        let Some(target) = self.prewalk.target.clone() else {
            return Ok(());
        };
        let restore = PrewalkRestore {
            provider: self.provider.name().to_owned(),
            model: self.model.clone(),
            reasoning: self.reasoning,
        };
        let same_model = target
            .provider
            .as_deref()
            .is_none_or(|provider| provider.eq_ignore_ascii_case(&restore.provider))
            && target.model == restore.model;
        let reasoning = self
            .prewalk
            .reasoning
            .filter(|reasoning| *reasoning != restore.reasoning);
        // A same-model handoff can still change reasoning; with nothing to
        // change, it only disarms.
        if same_model && reasoning.is_none() {
            self.send_prewalk_state(PrewalkChange::Session, events);
            return Ok(());
        }
        if !same_model
            && !self
                .switch_model_for_prewalk(target.provider, target.model, events)
                .await?
        {
            // The switch error is already reported; show the disarm too.
            self.send_prewalk_state(PrewalkChange::Session, events);
            return Ok(());
        }
        if let Some(reasoning) = reasoning {
            self.set_reasoning_policy(reasoning);
        }
        self.prewalk.restore = Some(restore);
        self.push_ephemeral_note(HANDOFF_NOTE.into());
        send(
            events,
            AgentEvent::PrewalkHandoff {
                provider: self.provider.name().to_owned(),
                model: self.model.clone(),
                reasoning: reasoning.map(|reasoning| reasoning.to_string()),
            },
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::InputMessage;
    use async_trait::async_trait;
    use futures_util::stream;
    use llm::{
        CompletionRequest, Content, EventStream, LlmError, ModelInfo, Provider, Role, StreamEvent,
        ToolCall,
    };
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use tokio_util::sync::CancellationToken;
    use tools::ToolRegistry;

    /// Records the model, nudge presence, and final message of each request.
    struct Recorder {
        calls: AtomicUsize,
        scripts: Vec<Vec<StreamEvent>>,
        seen: Mutex<Vec<(String, bool, Option<llm::Message>)>>,
    }

    #[async_trait]
    impl Provider for Recorder {
        fn name(&self) -> &str {
            "mock"
        }

        async fn stream(&self, request: &CompletionRequest) -> Result<EventStream, LlmError> {
            let index = self.calls.fetch_add(1, Ordering::SeqCst);
            let nudged = request
                .system
                .as_deref()
                .is_some_and(|system| system.contains(PLANNING_NUDGE));
            self.seen.lock().unwrap().push((
                request.model.clone(),
                nudged,
                request.messages.last().cloned(),
            ));
            let script = self.scripts.get(index).cloned().unwrap_or_default();
            Ok(Box::pin(stream::iter(script.into_iter().map(Ok))))
        }

        async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
            Ok(Vec::new())
        }
    }

    fn call(id: &str, name: &str) -> StreamEvent {
        StreamEvent::ToolCallComplete(ToolCall {
            id: id.into(),
            name: name.into(),
            arguments: json!({}),
        })
    }

    #[tokio::test]
    async fn first_edit_round_hands_off_to_the_target_once() {
        let done = StreamEvent::Done {
            stop_reason: None,
            usage: None,
        };
        let provider = Arc::new(Recorder {
            calls: AtomicUsize::new(0),
            scripts: vec![
                vec![call("r", "read"), done.clone()],
                vec![call("e", "edit"), done.clone()],
                vec![call("w", "write"), done.clone()],
                vec![StreamEvent::TextDelta("done".into()), done.clone()],
                vec![StreamEvent::TextDelta("next".into()), done],
            ],
            seen: Mutex::new(Vec::new()),
        });
        let (input_tx, input_rx) = tokio::sync::mpsc::unbounded_channel();
        let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
        for input in [
            InputMessage::Message("change it".into()),
            InputMessage::NewConversation,
            InputMessage::Message("next task".into()),
        ] {
            input_tx.send(input).unwrap();
        }
        drop(input_tx);
        Agent::new(
            provider.clone(),
            ToolRegistry::empty(),
            "planner",
            CancellationToken::new(),
        )
        .with_prewalk(PrewalkSettings {
            enabled: true,
            target: Some(PrewalkTarget {
                provider: None,
                model: "cheap".into(),
            }),
            armed: true,
            reasoning: Some(ReasoningPolicy::Off),
        })
        .run(input_rx, event_tx)
        .await;

        let seen = provider.seen.lock().unwrap();
        let routing = seen
            .iter()
            .map(|(model, nudged, _)| (model.as_str(), *nudged))
            .collect::<Vec<_>>();
        // Other tools keep planning; the (failed) edit ends it; the later
        // write does not trigger a second handoff. `/new` returns to the
        // planning model and arms again.
        assert_eq!(
            routing,
            [
                ("planner", true),
                ("planner", true),
                ("cheap", false),
                ("cheap", false),
                ("planner", true)
            ]
        );
        let note = seen[2].2.as_ref().expect("handoff request has messages");
        assert_eq!(note.role, Role::User);
        assert!(matches!(
            note.content.as_slice(),
            [Content::Text(text)] if text == HANDOFF_NOTE
        ));
        drop(seen);

        let mut handoffs = Vec::new();
        while let Ok(event) = event_rx.try_recv() {
            match event {
                AgentEvent::PrewalkHandoff {
                    provider,
                    model,
                    reasoning,
                }
                | AgentEvent::PrewalkRestored {
                    provider,
                    model,
                    reasoning,
                } => handoffs.push((provider, model, reasoning)),
                AgentEvent::ModelChanged { .. } | AgentEvent::ReasoningChanged { .. } => {
                    panic!("a handoff is not a user selection")
                }
                _ => {}
            }
        }
        assert_eq!(
            handoffs,
            [
                (
                    "mock".to_owned(),
                    "cheap".to_owned(),
                    Some("off".to_owned())
                ),
                (
                    "mock".to_owned(),
                    "planner".to_owned(),
                    Some("auto".to_owned())
                ),
            ]
        );
    }
    /// An agent whose restore point names a provider it cannot switch to.
    fn agent_with_unreachable_restore() -> Agent {
        let provider = Arc::new(Recorder {
            calls: AtomicUsize::new(0),
            scripts: Vec::new(),
            seen: Mutex::new(Vec::new()),
        });
        let mut agent = Agent::new(
            provider,
            ToolRegistry::empty(),
            "cheap",
            CancellationToken::new(),
        )
        .with_prewalk(PrewalkSettings {
            enabled: false,
            target: Some(PrewalkTarget {
                provider: None,
                model: "cheap".into(),
            }),
            armed: false,
            reasoning: None,
        });
        agent.prewalk.restore = Some(PrewalkRestore {
            provider: "other".into(),
            model: "planner".into(),
            reasoning: ReasoningPolicy::Auto,
        });
        agent
    }

    #[tokio::test]
    async fn rejected_restore_does_not_arm_and_keeps_the_restore_point() {
        let (events, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let mut agent = agent_with_unreachable_restore();
        agent.handle_set_prewalk(true, &events).await;
        assert!(!agent.prewalk.armed && !agent.prewalk.enabled);
        assert!(agent.prewalk.restore.is_some());
        assert!(matches!(rx.try_recv(), Ok(AgentEvent::Error(_))));

        // `/new` has the same rule: the preference is on, but arming would
        // plan on the target model.
        agent.prewalk.enabled = true;
        let outcome = agent.restore_prewalk_for_new_session(&events).await;
        assert!(matches!(outcome, RestoreOutcome::Rejected));
        agent.rearm_prewalk_for_new_session(&events, false);
        assert!(!agent.prewalk.armed);
        assert!(agent.prewalk.restore.is_some());
    }

    #[tokio::test]
    async fn rejected_user_commands_keep_the_restore_point() {
        let (events, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let mut agent = agent_with_unreachable_restore()
            .with_provider_factory(Arc::new(|name| anyhow::bail!("no provider {name}")));
        agent
            .handle_set_model_boundary(Some("bogus".into()), "x".into(), &events)
            .await;
        agent.handle_set_reasoning("nonsense".into(), &events);
        assert!(agent.prewalk.restore.is_some());
        assert!(matches!(rx.try_recv(), Ok(AgentEvent::Error(_))));

        // A valid reasoning selection takes ownership of the session.
        agent.handle_set_reasoning("off".into(), &events);
        assert!(agent.prewalk.restore.is_none());
    }
    /// An `edit` tool that shuts the application down while it runs.
    struct ShutdownEdit(CancellationToken);

    #[async_trait]
    impl tools::Tool for ShutdownEdit {
        fn spec(&self) -> tools::ToolSpec {
            tools::ToolSpec {
                definition: llm::ToolDefinition {
                    name: "edit".into(),
                    description: "probe".into(),
                    parameters: json!({"type": "object"}),
                },
                prompt: tools::ToolPrompt::default(),
            }
        }

        fn concurrency(&self, _args: &serde_json::Value) -> tools::Concurrency {
            tools::Concurrency::Exclusive
        }

        async fn execute(
            &self,
            _args: serde_json::Value,
            _cancel: CancellationToken,
        ) -> tools::ToolOutput {
            self.0.cancel();
            tools::ToolOutput {
                content: "done".into(),
                is_error: false,
                summary: "edit".into(),
            }
        }
    }

    #[tokio::test]
    async fn no_handoff_switch_is_persisted_during_application_shutdown() {
        let done = StreamEvent::Done {
            stop_reason: None,
            usage: None,
        };
        let provider = Arc::new(Recorder {
            calls: AtomicUsize::new(0),
            scripts: vec![vec![call("e", "edit"), done]],
            seen: Mutex::new(Vec::new()),
        });
        let app = CancellationToken::new();
        let (input_tx, input_rx) = tokio::sync::mpsc::unbounded_channel();
        let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();
        input_tx
            .send(InputMessage::Message("change it".into()))
            .unwrap();
        drop(input_tx);
        Agent::new(
            provider.clone(),
            ToolRegistry::new(vec![Box::new(ShutdownEdit(app.clone()))]),
            "planner",
            app,
        )
        .with_prewalk(PrewalkSettings {
            enabled: true,
            target: Some(PrewalkTarget {
                provider: None,
                model: "cheap".into(),
            }),
            armed: true,
            reasoning: None,
        })
        .run(input_rx, event_tx)
        .await;
        while let Ok(event) = event_rx.try_recv() {
            assert!(
                !matches!(event, AgentEvent::PrewalkHandoff { .. }),
                "handoff must not run during shutdown"
            );
        }
        assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
    }
}
