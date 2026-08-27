use agent::{AgentEvent, InputMessage, SessionSnapshotEntry};
use tokio::sync::mpsc;

/// Convert the terminal UI's independent input protocol into runtime input.
pub fn into_agent_input(message: tui::InputMessage) -> InputMessage {
    match message {
        tui::InputMessage::Message(value) => InputMessage::Message(value),
        tui::InputMessage::Interrupt => InputMessage::Interrupt,
        tui::InputMessage::NewConversation => InputMessage::NewConversation,
        tui::InputMessage::LoadSession { selector } => InputMessage::LoadSession { selector },
        tui::InputMessage::ListSessions => InputMessage::ListSessions,
        tui::InputMessage::ExportSession { destination } => {
            InputMessage::ExportSession { destination }
        }
        tui::InputMessage::CompactSession => InputMessage::CompactSession,
        tui::InputMessage::SetModel { provider, model } => {
            InputMessage::SetModel { provider, model }
        }
        tui::InputMessage::SetReasoning { level } => InputMessage::SetReasoning { level },
        tui::InputMessage::ListModels { provider } => InputMessage::ListModels { provider },
        tui::InputMessage::InvokeSkill { name } => InputMessage::InvokeSkill { name },
        tui::InputMessage::SubscriptionUsage => InputMessage::SubscriptionUsage,
        tui::InputMessage::ListSkills => InputMessage::ListSkills,
    }
}

/// Forward terminal input until either side closes. This task must own the
/// only runtime sender so dropping the TUI sender closes the runtime input.
pub async fn forward_inputs(
    mut input: mpsc::UnboundedReceiver<tui::InputMessage>,
    output: mpsc::UnboundedSender<InputMessage>,
) {
    loop {
        tokio::select! {
            _ = output.closed() => break,
            message = input.recv() => {
                let Some(message) = message else { break };
                if output.send(into_agent_input(message)).is_err() {
                    break;
                }
            }
        }
    }
}

/// Convert frontend-neutral runtime facts into terminal presentation records.
pub fn into_ui_event(event: AgentEvent) -> tui::UiEvent {
    match event {
        AgentEvent::TextDelta(value) => tui::UiEvent::TextDelta(value),
        AgentEvent::ReasoningDelta(value) => tui::UiEvent::ReasoningDelta(value),
        AgentEvent::ToolCallStarted {
            call_id,
            name,
            summary,
        } => tui::UiEvent::ToolCallStarted {
            call_id,
            name,
            summary,
        },
        AgentEvent::ToolCallFinished {
            call_id,
            name,
            summary,
            ok,
            duration_ms,
            output,
            error,
        } => tui::UiEvent::ToolCallFinished {
            call_id,
            name,
            summary,
            ok,
            duration_ms,
            output,
            error,
        },
        AgentEvent::Retrying { attempt, message } => tui::UiEvent::Retrying { attempt, message },
        AgentEvent::TurnFinished => tui::UiEvent::TurnFinished,
        AgentEvent::Error(value) => tui::UiEvent::Error(value),
        AgentEvent::Notice(value) => tui::UiEvent::Notice(value),
        AgentEvent::ModelChanged { provider, model } => {
            tui::UiEvent::ModelChanged { provider, model }
        }
        AgentEvent::ReasoningChanged { level } => tui::UiEvent::ReasoningChanged { level },
        AgentEvent::ModelList { provider, models } => tui::UiEvent::ModelList {
            provider,
            models: models
                .into_iter()
                .map(|model| tui::ModelEntry {
                    id: model.id,
                    name: model.name,
                    context_length: model.context_length,
                })
                .collect(),
        },
        AgentEvent::SessionChanged { id, title, loaded } => {
            tui::UiEvent::SessionChanged { id, title, loaded }
        }
        AgentEvent::SessionSnapshot { entries } => tui::UiEvent::SessionSnapshot {
            entries: entries
                .into_iter()
                .map(|entry| match entry {
                    SessionSnapshotEntry::User { text } => tui::SessionSnapshotEntry::User { text },
                    SessionSnapshotEntry::Assistant {
                        markdown,
                        reasoning,
                    } => tui::SessionSnapshotEntry::Assistant {
                        markdown,
                        reasoning,
                    },
                    SessionSnapshotEntry::Tool {
                        name,
                        summary,
                        ok,
                        duration_ms,
                        output,
                        error,
                    } => tui::SessionSnapshotEntry::Tool {
                        name,
                        summary,
                        ok,
                        duration_ms,
                        output,
                        error,
                    },
                })
                .collect(),
        },
        AgentEvent::SessionList { sessions } => tui::UiEvent::SessionList {
            sessions: sessions
                .into_iter()
                .map(|session| tui::SessionListEntry {
                    id: session.id,
                    short_id: session.short_id,
                    title: session.title,
                    updated_at: session.updated_at,
                    workspace: session.workspace,
                    provider: session.provider,
                    model: session.model,
                })
                .collect(),
        },
        AgentEvent::SessionExported { path } => tui::UiEvent::SessionExported { path },
        AgentEvent::SkillsLoaded {
            skills,
            diagnostics,
            empty,
        } => tui::UiEvent::SkillsLoaded {
            skills: skills
                .into_iter()
                .map(|skill| tui::SkillEntry {
                    name: skill.name,
                    description: skill.description,
                })
                .collect(),
            diagnostics,
            empty,
        },
        AgentEvent::UsageUpdated {
            input_tokens,
            output_tokens,
            cached_tokens,
            reasoning_tokens,
            cost,
        } => tui::UiEvent::UsageUpdated {
            input_tokens,
            output_tokens,
            cached_tokens,
            reasoning_tokens,
            cost,
        },
        AgentEvent::SubscriptionUsageLoaded { provider, usage } => {
            tui::UiEvent::SubscriptionUsageLoaded {
                provider,
                usage: tui::SubscriptionUsage {
                    plan: usage.plan,
                    windows: usage
                        .windows
                        .into_iter()
                        .map(|window| tui::SubscriptionUsageWindow {
                            label: window.label,
                            used_percent: window.used_percent,
                            status: window.status,
                            resets_at: window.resets_at,
                            resets_after_seconds: window.resets_after_seconds,
                        })
                        .collect(),
                },
            }
        }
        AgentEvent::CompactionFinished {
            compacted_through,
            summary_bytes,
            auto,
            reason,
        } => tui::UiEvent::CompactionFinished {
            compacted_through,
            summary_bytes,
            auto,
            reason: reason.to_string(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_every_input_variant() {
        let cases = [
            (
                tui::InputMessage::Message("hi".into()),
                InputMessage::Message("hi".into()),
            ),
            (tui::InputMessage::Interrupt, InputMessage::Interrupt),
            (
                tui::InputMessage::NewConversation,
                InputMessage::NewConversation,
            ),
            (
                tui::InputMessage::LoadSession {
                    selector: "latest".into(),
                },
                InputMessage::LoadSession {
                    selector: "latest".into(),
                },
            ),
            (tui::InputMessage::ListSessions, InputMessage::ListSessions),
            (
                tui::InputMessage::ExportSession {
                    destination: Some("x".into()),
                },
                InputMessage::ExportSession {
                    destination: Some("x".into()),
                },
            ),
            (
                tui::InputMessage::CompactSession,
                InputMessage::CompactSession,
            ),
            (
                tui::InputMessage::SetModel {
                    provider: Some("p".into()),
                    model: "m".into(),
                },
                InputMessage::SetModel {
                    provider: Some("p".into()),
                    model: "m".into(),
                },
            ),
            (
                tui::InputMessage::SetReasoning {
                    level: "high".into(),
                },
                InputMessage::SetReasoning {
                    level: "high".into(),
                },
            ),
            (
                tui::InputMessage::ListModels {
                    provider: "p".into(),
                },
                InputMessage::ListModels {
                    provider: "p".into(),
                },
            ),
            (
                tui::InputMessage::InvokeSkill { name: "s".into() },
                InputMessage::InvokeSkill { name: "s".into() },
            ),
            (
                tui::InputMessage::SubscriptionUsage,
                InputMessage::SubscriptionUsage,
            ),
            (tui::InputMessage::ListSkills, InputMessage::ListSkills),
        ];
        for (source, expected) in cases {
            assert_eq!(into_agent_input(source), expected);
        }
    }

    #[tokio::test]
    async fn closing_tui_input_closes_runtime_input() {
        let (tui_tx, tui_rx) = mpsc::unbounded_channel();
        let (runtime_tx, mut runtime_rx) = mpsc::unbounded_channel();
        let task = tokio::spawn(forward_inputs(tui_rx, runtime_tx));
        tui_tx
            .send(tui::InputMessage::Message("hi".into()))
            .unwrap();
        assert_eq!(
            runtime_rx.recv().await,
            Some(InputMessage::Message("hi".into()))
        );
        drop(tui_tx);
        tokio::time::timeout(std::time::Duration::from_secs(1), task)
            .await
            .expect("forwarder should stop")
            .unwrap();
        assert_eq!(runtime_rx.recv().await, None);
    }

    #[test]
    fn converts_every_event_variant() {
        let events = vec![
            AgentEvent::TextDelta("text".into()),
            AgentEvent::ReasoningDelta("reason".into()),
            AgentEvent::ToolCallStarted {
                call_id: "call".into(),
                name: "tool".into(),
                summary: "start".into(),
            },
            AgentEvent::ToolCallFinished {
                call_id: "call".into(),
                name: "tool".into(),
                summary: "done".into(),
                ok: false,
                duration_ms: 9,
                output: "output".into(),
                error: Some("error".into()),
            },
            AgentEvent::Retrying {
                attempt: 2,
                message: "retry".into(),
            },
            AgentEvent::TurnFinished,
            AgentEvent::Error("error".into()),
            AgentEvent::Notice("notice".into()),
            AgentEvent::ModelChanged {
                provider: "provider".into(),
                model: "model".into(),
            },
            AgentEvent::ReasoningChanged {
                level: "high".into(),
            },
            AgentEvent::ModelList {
                provider: "provider".into(),
                models: vec![llm::ModelInfo {
                    id: "id".into(),
                    name: Some("name".into()),
                    context_length: Some(42),
                }],
            },
            AgentEvent::SessionChanged {
                id: "id".into(),
                title: Some("title".into()),
                loaded: true,
            },
            AgentEvent::SessionSnapshot {
                entries: vec![
                    SessionSnapshotEntry::User {
                        text: "user".into(),
                    },
                    SessionSnapshotEntry::Assistant {
                        markdown: "answer".into(),
                        reasoning: "thought".into(),
                    },
                    SessionSnapshotEntry::Tool {
                        name: "tool".into(),
                        summary: "summary".into(),
                        ok: true,
                        duration_ms: 3,
                        output: "output".into(),
                        error: None,
                    },
                ],
            },
            AgentEvent::SessionList { sessions: vec![] },
            AgentEvent::SessionExported { path: "out".into() },
            AgentEvent::SkillsLoaded {
                skills: vec![tools::SkillEntry {
                    name: "skill".into(),
                    description: "description".into(),
                }],
                diagnostics: vec!["diagnostic".into()],
                empty: false,
            },
            AgentEvent::UsageUpdated {
                input_tokens: 1,
                output_tokens: 2,
                cached_tokens: 3,
                reasoning_tokens: 4,
                cost: "$0.01".into(),
            },
            AgentEvent::SubscriptionUsageLoaded {
                provider: "provider".into(),
                usage: llm::SubscriptionUsage {
                    plan: Some("plan".into()),
                    windows: vec![llm::SubscriptionUsageWindow {
                        label: "weekly".into(),
                        used_percent: 25,
                        status: Some("ok".into()),
                        resets_at: Some("tomorrow".into()),
                        resets_after_seconds: Some(60),
                    }],
                },
            },
            AgentEvent::CompactionFinished {
                compacted_through: 7,
                summary_bytes: 11,
                auto: true,
                reason: agent::CompactionReason::Auto,
            },
        ];
        let converted = events.into_iter().map(into_ui_event).collect::<Vec<_>>();
        assert!(matches!(converted[0], tui::UiEvent::TextDelta(_)));
        assert!(matches!(converted[1], tui::UiEvent::ReasoningDelta(_)));
        assert!(matches!(converted[2], tui::UiEvent::ToolCallStarted { .. }));
        assert!(matches!(
            converted[3],
            tui::UiEvent::ToolCallFinished { .. }
        ));
        assert!(matches!(converted[4], tui::UiEvent::Retrying { .. }));
        assert_eq!(converted[5], tui::UiEvent::TurnFinished);
        assert!(matches!(converted[6], tui::UiEvent::Error(_)));
        assert!(matches!(converted[7], tui::UiEvent::Notice(_)));
        assert!(matches!(converted[8], tui::UiEvent::ModelChanged { .. }));
        assert!(matches!(
            converted[9],
            tui::UiEvent::ReasoningChanged { .. }
        ));
        assert!(matches!(converted[10], tui::UiEvent::ModelList { .. }));
        assert!(matches!(converted[11], tui::UiEvent::SessionChanged { .. }));
        assert!(matches!(
            converted[12],
            tui::UiEvent::SessionSnapshot { .. }
        ));
        assert!(matches!(converted[13], tui::UiEvent::SessionList { .. }));
        assert!(matches!(
            converted[14],
            tui::UiEvent::SessionExported { .. }
        ));
        assert!(matches!(converted[15], tui::UiEvent::SkillsLoaded { .. }));
        assert!(matches!(converted[16], tui::UiEvent::UsageUpdated { .. }));
        assert!(matches!(
            converted[17],
            tui::UiEvent::SubscriptionUsageLoaded { .. }
        ));
        assert!(matches!(
            converted[18],
            tui::UiEvent::CompactionFinished { .. }
        ));
    }

    #[test]
    fn preserves_nested_event_fields() {
        let event = AgentEvent::SessionList {
            sessions: vec![agent::SessionListItem {
                id: "full".into(),
                short_id: "short".into(),
                title: Some("title".into()),
                updated_at: "now".into(),
                workspace: "/workspace".into(),
                provider: Some("provider".into()),
                model: Some("model".into()),
            }],
        };
        assert_eq!(
            into_ui_event(event),
            tui::UiEvent::SessionList {
                sessions: vec![tui::SessionListEntry {
                    id: "full".into(),
                    short_id: "short".into(),
                    title: Some("title".into()),
                    updated_at: "now".into(),
                    workspace: "/workspace".into(),
                    provider: Some("provider".into()),
                    model: Some("model".into()),
                }],
            }
        );

        assert_eq!(
            into_ui_event(AgentEvent::CompactionFinished {
                compacted_through: 7,
                summary_bytes: 11,
                auto: true,
                reason: agent::CompactionReason::Overflow,
            }),
            tui::UiEvent::CompactionFinished {
                compacted_through: 7,
                summary_bytes: 11,
                auto: true,
                reason: "overflow".into(),
            }
        );
    }
}
