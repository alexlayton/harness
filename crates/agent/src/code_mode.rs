//! Agent policy for code-mode callbacks. The code itself always stays masked;
//! only an opted-in tool receives a temporary restored argument view.
//!
//! Nested starts are acknowledged only after their durable record is written.
//! A start without a finish after cancellation or future drop means that the
//! execution status is unknown, not that the tool failed or did not run.

use crate::secrets::SecretMasker;
use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;
use tools::{CodeModeEvent, CodeModeExecutor, ToolOutput, ToolRegistry};

/// A callback waits for the dispatch loop to persist and acknowledge its event.
pub(crate) struct NestedNotification {
    pub(crate) parent_call_id: String,
    pub(crate) event: CodeModeEvent,
    pub(crate) ack: oneshot::Sender<Result<(), String>>,
}

/// Per-workflow notification channel and cancellation scope.
pub(crate) struct NestedEventSink {
    pub(crate) parent_call_id: String,
    pub(crate) sender: mpsc::Sender<NestedNotification>,
    pub(crate) cancel: CancellationToken,
    pub(crate) admission: CancellationToken,
}

/// Applies the same secret boundary as ordinary dispatch, without reacquiring
/// the registry execution gate already held for the complete workflow.
pub(crate) struct AgentCodeModeExecutor<'a> {
    registry: &'a ToolRegistry,
    secrets: Option<&'a SecretMasker>,
    restore_secrets: bool,
    notifications: NestedEventSink,
}

impl<'a> AgentCodeModeExecutor<'a> {
    pub(crate) fn new(
        registry: &'a ToolRegistry,
        secrets: Option<&'a SecretMasker>,
        restore_secrets: bool,
        notifications: NestedEventSink,
    ) -> Self {
        Self {
            registry,
            secrets,
            restore_secrets,
            notifications,
        }
    }

    fn mask_event(&self, event: CodeModeEvent) -> CodeModeEvent {
        // Namespace bridge-local IDs so successive workflows cannot collide
        // in the frontend, even if each interpreter starts its counter at zero.
        match event {
            CodeModeEvent::Started {
                call_id,
                name,
                arguments,
            } => CodeModeEvent::Started {
                call_id: format!("{}:{call_id}", self.notifications.parent_call_id),
                name: self
                    .secrets
                    .map_or(name.clone(), |masker| masker.mask_text(&name)),
                arguments: match self.secrets {
                    Some(masker) => masker.mask_json(&arguments),
                    None => arguments,
                },
            },
            CodeModeEvent::Finished {
                call_id,
                name,
                output,
            } => CodeModeEvent::Finished {
                call_id: format!("{}:{call_id}", self.notifications.parent_call_id),
                name: self
                    .secrets
                    .map_or(name.clone(), |masker| masker.mask_text(&name)),
                output: match self.secrets {
                    Some(masker) => masker.mask_tool_output(output),
                    None => output,
                },
            },
        }
    }
}

#[async_trait]
impl CodeModeExecutor for AgentCodeModeExecutor<'_> {
    async fn call(
        &self,
        name: &str,
        args: Value,
        cancel: CancellationToken,
    ) -> (ToolOutput, Option<Value>) {
        if cancel.is_cancelled()
            || self.notifications.cancel.is_cancelled()
            || self.notifications.admission.is_cancelled()
        {
            return (rejected("cancelled"), None);
        }
        let arguments = if let Some(masker) = self.secrets {
            // First mask raw values, then use the existing out-of-band key
            // rejection flag. This rejects both raw secrets and placeholders
            // in keys without allowing an ordinary JSON field to forge it.
            let masked = masker.mask_json(&args);
            let (_, rejected_key) = masker.restore_json_for_execution(&args);
            if rejected_key || masked_keys_changed(masker, &args) {
                return (
                    rejected(
                        "tool arguments were rejected because secret values and placeholders are not supported in JSON object keys",
                    ),
                    None,
                );
            }
            if self.restore_secrets && self.registry.accepts_restored_secrets(name) {
                masker.restore_json_for_execution(&masked).0
            } else {
                masked
            }
        } else {
            args
        };
        let result = tokio::select! {
            biased;
            _ = self.notifications.cancel.cancelled() => (rejected("cancelled"), None),
            result = self.registry.execute_code_mode_tool(name, arguments, cancel) => result,
        };
        match self.secrets {
            Some(masker) => (
                masker.mask_tool_output(result.0),
                result.1.map(|value| masker.mask_json(&value)),
            ),
            None => result,
        }
    }

    async fn record(&self, event: CodeModeEvent) -> Result<(), String> {
        let (ack, received) = oneshot::channel();
        let notification = NestedNotification {
            parent_call_id: self.notifications.parent_call_id.clone(),
            event: self.mask_event(event),
            ack,
        };
        tokio::select! {
            biased;
            _ = self.notifications.cancel.cancelled() => return Err("code-mode workflow cancelled".into()),
            _ = self.notifications.admission.cancelled() => return Err("code-mode admission closed".into()),
            sent = self.notifications.sender.send(notification) => {
                sent.map_err(|_| "code-mode dispatch closed".to_owned())?;
            }
        }
        tokio::select! {
            biased;
            _ = self.notifications.cancel.cancelled() => Err("code-mode workflow cancelled".into()),
            _ = self.notifications.admission.cancelled() => Err("code-mode admission closed".into()),
            result = received => result.map_err(|_| "code-mode acknowledgement dropped".to_owned())?,
        }
    }
}

// `mask_json` replaces an entire object with a rejection marker when a key
// contains a raw secret. Check keys directly using that same exact-value
// transformation; comparing complete JSON values would also reject ordinary
// secret-bearing string values, which are allowed for opted-in tools.
fn masked_keys_changed(masker: &SecretMasker, value: &Value) -> bool {
    match value {
        Value::Object(values) => values.iter().any(|(key, value)| {
            masker.mask_text(key) != *key || masked_keys_changed(masker, value)
        }),
        Value::Array(values) => values
            .iter()
            .any(|value| masked_keys_changed(masker, value)),
        _ => false,
    }
}

fn rejected(content: &str) -> ToolOutput {
    ToolOutput {
        content: content.into(),
        is_error: true,
        summary: "code-mode callback rejected".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::FutureExt;
    use serde_json::json;
    use std::sync::{Arc, Mutex};
    use tools::{Tool, ToolSpec};

    struct Echo {
        name: &'static str,
        restore: bool,
        received: Arc<Mutex<Vec<Value>>>,
    }

    #[async_trait]
    impl Tool for Echo {
        fn spec(&self) -> ToolSpec {
            ToolSpec {
                definition: llm::ToolDefinition {
                    name: self.name.into(),
                    description: "echo arguments".into(),
                    parameters: json!({"type": "object"}),
                },
                prompt: tools::ToolPrompt::default(),
            }
        }

        fn accepts_restored_secrets(&self) -> bool {
            self.restore
        }

        async fn execute(&self, args: Value, _cancel: CancellationToken) -> ToolOutput {
            self.received.lock().unwrap().push(args.clone());
            ToolOutput {
                content: args.to_string(),
                summary: args.to_string(),
                is_error: false,
            }
        }

        async fn execute_structured(
            &self,
            args: Value,
            cancel: CancellationToken,
        ) -> (ToolOutput, Option<Value>) {
            (self.execute(args.clone(), cancel).await, Some(args))
        }
    }

    #[tokio::test]
    async fn nested_tools_restore_only_opted_in_values_and_mask_both_result_forms() {
        let received = Arc::new(Mutex::new(Vec::new()));
        let mut registry = ToolRegistry::try_new(vec![
            Box::new(Echo {
                name: "local",
                restore: true,
                received: received.clone(),
            }),
            Box::new(Echo {
                name: "external",
                restore: false,
                received: received.clone(),
            }),
        ])
        .unwrap();
        registry.enable_code_mode(&[]).unwrap();
        let masker = SecretMasker::new([("TOKEN".into(), "secret-value".into())]).unwrap();
        let (sender, _receiver) = mpsc::channel(1);
        let executor = AgentCodeModeExecutor::new(
            &registry,
            Some(&masker),
            true,
            NestedEventSink {
                parent_call_id: "parent".into(),
                sender,
                cancel: CancellationToken::new(),
                admission: CancellationToken::new(),
            },
        );
        let placeholder = "{{harness-secret:TOKEN}}";
        for name in ["local", "external"] {
            let (output, structured) = executor
                .call(
                    name,
                    json!({"value": placeholder}),
                    CancellationToken::new(),
                )
                .await;
            assert!(!output.is_error);
            assert!(!output.content.contains("secret-value"));
            assert!(!output.summary.contains("secret-value"));
            assert_eq!(structured.unwrap()["value"], placeholder);
        }
        assert_eq!(received.lock().unwrap()[0]["value"], "secret-value");
        assert_eq!(received.lock().unwrap()[1]["value"], placeholder);
        for key in [placeholder, "secret-value"] {
            let (output, _) = executor
                .call("local", json!({key: "value"}), CancellationToken::new())
                .await;
            assert!(output.is_error);
        }
        assert_eq!(received.lock().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn failed_start_acknowledgement_prevents_callback_side_effects() {
        let received = Arc::new(Mutex::new(Vec::new()));
        let mut registry = ToolRegistry::try_new(vec![Box::new(Echo {
            name: "local",
            restore: false,
            received: received.clone(),
        })])
        .unwrap();
        registry.enable_code_mode(&[]).unwrap();
        let (sender, mut receiver) = mpsc::channel(1);
        let executor = AgentCodeModeExecutor::new(
            &registry,
            None,
            true,
            NestedEventSink {
                parent_call_id: "parent".into(),
                sender,
                cancel: CancellationToken::new(),
                admission: CancellationToken::new(),
            },
        );
        let workflow = async {
            executor
                .record(CodeModeEvent::Started {
                    call_id: "nested".into(),
                    name: "local".into(),
                    arguments: json!({}),
                })
                .await?;
            executor
                .call("local", json!({}), CancellationToken::new())
                .await;
            Ok::<_, String>(())
        };
        tokio::pin!(workflow);
        assert!(workflow.as_mut().now_or_never().is_none());
        let notification = receiver.recv().await.unwrap();
        assert_eq!(notification.parent_call_id, "parent");
        assert!(
            matches!(notification.event, CodeModeEvent::Started { call_id, .. } if call_id == "parent:nested")
        );
        assert!(received.lock().unwrap().is_empty());
        notification.ack.send(Err("persist failed".into())).unwrap();
        assert_eq!(workflow.await, Err("persist failed".into()));
        assert!(received.lock().unwrap().is_empty());
    }
}
