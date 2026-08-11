//! Pure slash-command parsing and completion candidate generation.
//!
//! This module intentionally has no terminal or asynchronous code.  The app
//! owns the editor and cached lists, while the agent remains responsible for
//! validating providers and executing the resulting semantic messages.

use crate::ModelEntry;
use std::collections::{HashMap, HashSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommandSpec {
    pub name: &'static str,
    pub description: &'static str,
    pub usage: &'static str,
}

pub const COMMANDS: &[CommandSpec] = &[
    CommandSpec {
        name: "/new",
        description: "Reset the current conversation",
        usage: "/new",
    },
    CommandSpec {
        name: "/model",
        description: "Set the model (and provider) to use",
        usage: "/model [<provider>:]<model>",
    },
];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Candidate {
    pub value: String,
    pub description: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ParsedCommand {
    New,
    SetModel {
        provider: Option<String>,
        model: String,
    },
}

/// A command is deliberately restricted to one line.  A multiline message
/// beginning with `/` is ordinary user text and should still be sendable.
pub fn is_command_input(text: &str) -> bool {
    text.starts_with('/') && !text.contains(['\n', '\r'])
}

pub fn parse_command(text: &str) -> Result<ParsedCommand, String> {
    if !is_command_input(text) {
        return Err("commands must start with / and fit on one line".into());
    }
    let input = text.trim();
    let mut words = input.split_whitespace();
    let command = words.next().unwrap_or("");
    match command.to_ascii_lowercase().as_str() {
        "/new" => {
            if words.next().is_some() {
                Err("usage: /new".into())
            } else {
                Ok(ParsedCommand::New)
            }
        }
        "/model" => {
            let Some(token) = words.next() else {
                // A bare command is represented explicitly so the TUI can
                // provide a useful hint without sending anything to the agent.
                return Ok(ParsedCommand::SetModel {
                    provider: None,
                    model: String::new(),
                });
            };
            if words.next().is_some() {
                return Err("usage: /model [<provider>:]<model>".into());
            }
            if let Some((provider, model)) = token.split_once(':') {
                if provider.is_empty() || model.is_empty() {
                    return Err("usage: /model [<provider>:]<model>".into());
                }
                Ok(ParsedCommand::SetModel {
                    provider: Some(provider.to_owned()),
                    model: model.to_owned(),
                })
            } else if token.is_empty() {
                Err("usage: /model [<provider>:]<model>".into())
            } else {
                Ok(ParsedCommand::SetModel {
                    provider: None,
                    model: token.to_owned(),
                })
            }
        }
        _ => Err(format!("unknown command: {command}")),
    }
}

/// Return completion candidates for the current single-line editor value.
pub fn candidates(
    input: &str,
    providers: &[String],
    model_lists: &HashMap<String, Vec<ModelEntry>>,
    current_provider: &str,
) -> Vec<Candidate> {
    if !is_command_input(input) {
        return Vec::new();
    }

    let Some(command_end) = input.find(char::is_whitespace) else {
        let token = input.to_ascii_lowercase();
        return COMMANDS
            .iter()
            .filter(|command| command.name.to_ascii_lowercase().starts_with(&token))
            .map(|command| Candidate {
                value: command.name.to_owned(),
                description: command.description.to_owned(),
            })
            .collect();
    };

    let command = &input[..command_end];
    if !command.eq_ignore_ascii_case("/model") {
        return Vec::new();
    }

    let token_start = input[command_end..]
        .find(|character: char| !character.is_whitespace())
        .map(|offset| command_end + offset)
        .unwrap_or(input.len());
    let token_end = input[token_start..]
        .find(char::is_whitespace)
        .map(|offset| token_start + offset)
        .unwrap_or(input.len());
    let token = &input[token_start..token_end];

    let mut results = Vec::<(bool, Candidate)>::new();
    let mut seen = HashSet::new();

    let (explicit_provider, model_partial) = match token.split_once(':') {
        Some((prefix, partial)) => {
            let provider = providers
                .iter()
                .find(|provider| provider.eq_ignore_ascii_case(prefix));
            let Some(provider) = provider else {
                // A colon commits the token to provider-qualified syntax. Do
                // not suggest models from the current provider for an unknown
                // provider prefix.
                return Vec::new();
            };
            (Some(provider.as_str()), partial)
        }
        None => (None, token),
    };

    if explicit_provider.is_none() {
        let prefix = token.to_ascii_lowercase();
        for provider in providers {
            if provider.to_ascii_lowercase().starts_with(&prefix) {
                let value = format!("{provider}:");
                if seen.insert(value.clone()) {
                    results.push((
                        true,
                        Candidate {
                            value,
                            description: "provider".into(),
                        },
                    ));
                }
            }
        }
    }

    let list_provider = explicit_provider.unwrap_or(current_provider);
    if let Some(models) = model_lists.get(list_provider) {
        let partial = model_partial.to_ascii_lowercase();
        for model in models {
            let id_matches = model.id.to_ascii_lowercase().contains(&partial);
            let name_matches = model
                .name
                .as_deref()
                .is_some_and(|name| name.to_ascii_lowercase().contains(&partial));
            if !id_matches && !name_matches {
                continue;
            }
            let value = match explicit_provider {
                Some(provider) => format!("{provider}:{}", model.id),
                None => model.id.clone(),
            };
            if seen.insert(value.clone()) {
                results.push((
                    false,
                    Candidate {
                        value,
                        description: model_description(model),
                    },
                ));
            }
        }
    }

    results.sort_by(|(provider_a, candidate_a), (provider_b, candidate_b)| {
        provider_b
            .cmp(provider_a)
            .then_with(|| {
                candidate_a
                    .value
                    .to_ascii_lowercase()
                    .cmp(&candidate_b.value.to_ascii_lowercase())
            })
            .then_with(|| candidate_a.value.cmp(&candidate_b.value))
    });
    results
        .into_iter()
        .map(|(_, candidate)| candidate)
        .collect()
}

fn model_description(model: &ModelEntry) -> String {
    let label = model
        .name
        .as_deref()
        .filter(|name| !name.trim().is_empty())
        .unwrap_or(&model.id);
    match model.context_length {
        Some(context) => format!("{label} · {} ctx", format_context_length(context)),
        None => label.to_owned(),
    }
}

fn format_context_length(context: u64) -> String {
    if context >= 1_000_000 {
        let value = context as f64 / 1_000_000.0;
        if value.fract() == 0.0 {
            format!("{}M", value as u64)
        } else {
            format!("{value:.1}M")
        }
    } else if context >= 1_000 {
        format!("{}k", context / 1_000)
    } else {
        context.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(id: &str, name: Option<&str>, context_length: Option<u64>) -> ModelEntry {
        ModelEntry {
            id: id.into(),
            name: name.map(str::to_owned),
            context_length,
        }
    }

    fn providers() -> Vec<String> {
        vec!["opencode-go".into(), "openrouter".into()]
    }

    #[test]
    fn command_candidates_filter_the_command_token_case_insensitively() {
        let lists = HashMap::new();
        let all = candidates("/", &providers(), &lists, "opencode-go");
        assert_eq!(all.len(), 2);
        assert_eq!(
            candidates("/NE", &providers(), &lists, "opencode-go")[0].value,
            "/new"
        );
        assert!(candidates("hello", &providers(), &lists, "opencode-go").is_empty());
    }

    #[test]
    fn argument_phase_starts_only_after_whitespace() {
        let lists = HashMap::from([(
            "opencode-go".into(),
            vec![model("gpt-5.6-luna", Some("GPT 5.6 Luna"), Some(400_000))],
        )]);
        assert_eq!(
            candidates("/model", &providers(), &lists, "opencode-go")[0].value,
            "/model"
        );
        let values = candidates("/model ", &providers(), &lists, "opencode-go")
            .into_iter()
            .map(|candidate| candidate.value)
            .collect::<Vec<_>>();
        assert!(values.contains(&"opencode-go:".into()));
        assert!(values.contains(&"openrouter:".into()));
        assert!(values.contains(&"gpt-5.6-luna".into()));
    }

    #[test]
    fn provider_prefixes_and_substring_model_filtering_work() {
        let lists = HashMap::from([(
            "opencode-go".into(),
            vec![
                model("gpt-5.6-luna", Some("GPT 5.6 Luna"), Some(400_000)),
                model("minimax-m3", Some("MiniMax"), None),
            ],
        )]);
        let prefix = candidates("/model open", &providers(), &lists, "opencode-go");
        assert_eq!(prefix[0].value, "opencode-go:");
        assert!(
            prefix
                .iter()
                .any(|candidate| candidate.value == "openrouter:")
        );
        let filtered = candidates("/model luna", &providers(), &lists, "opencode-go");
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].value, "gpt-5.6-luna");
        assert!(filtered[0].description.contains("GPT 5.6 Luna"));
        assert!(filtered[0].description.contains("400k ctx"));
    }

    #[test]
    fn explicit_provider_uses_that_provider_and_keeps_the_prefix() {
        let lists = HashMap::from([(
            "openrouter".into(),
            vec![model(
                "openai/gpt-5.6-luna",
                Some("GPT Luna"),
                Some(128_000),
            )],
        )]);
        let values = candidates("/model openrouter:gpt", &providers(), &lists, "opencode-go");
        assert_eq!(values.len(), 1);
        assert_eq!(values[0].value, "openrouter:openai/gpt-5.6-luna");
    }

    #[test]
    fn parses_success_and_usage_errors() {
        assert_eq!(parse_command("/new"), Ok(ParsedCommand::New));
        assert_eq!(
            parse_command("/model gpt-5.6-luna"),
            Ok(ParsedCommand::SetModel {
                provider: None,
                model: "gpt-5.6-luna".into()
            })
        );
        assert_eq!(
            parse_command("/model openrouter:openai/demo"),
            Ok(ParsedCommand::SetModel {
                provider: Some("openrouter".into()),
                model: "openai/demo".into()
            })
        );
        assert_eq!(
            parse_command("/model"),
            Ok(ParsedCommand::SetModel {
                provider: None,
                model: String::new()
            })
        );
        assert_eq!(parse_command("/new now"), Err("usage: /new".into()));
        assert_eq!(
            parse_command("/model one two"),
            Err("usage: /model [<provider>:]<model>".into())
        );
        assert_eq!(parse_command("/foo"), Err("unknown command: /foo".into()));
    }
}
