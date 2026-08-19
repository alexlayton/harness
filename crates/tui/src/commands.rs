//! Pure slash-command parsing and completion candidate generation.
//!
//! This module intentionally has no terminal or asynchronous code.  The app
//! owns the editor and cached lists, while the agent remains responsible for
//! validating providers and executing the resulting semantic messages.

use crate::{ModelEntry, SessionListEntry};
use std::collections::{HashMap, HashSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArgumentKind {
    None,
    Model,
    Session,
    Path,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommandSpec {
    pub name: &'static str,
    pub description: &'static str,
    pub usage: &'static str,
    pub argument_kind: ArgumentKind,
}

pub const COMMANDS: &[CommandSpec] = &[
    CommandSpec {
        name: "/auth",
        description: "Authenticate GitHub Copilot",
        usage: "/auth",
        argument_kind: ArgumentKind::None,
    },
    CommandSpec {
        name: "/new",
        description: "Start a new persisted conversation",
        usage: "/new",
        argument_kind: ArgumentKind::None,
    },
    CommandSpec {
        name: "/load",
        description: "Load a session by ID, path, or latest",
        usage: "/load [<id>|latest|<path>]",
        argument_kind: ArgumentKind::Session,
    },
    CommandSpec {
        name: "/sessions",
        description: "List sessions for this workspace",
        usage: "/sessions",
        argument_kind: ArgumentKind::None,
    },
    CommandSpec {
        name: "/export",
        description: "Export the current session to JSONL",
        usage: "/export [<path>]",
        argument_kind: ArgumentKind::Path,
    },
    CommandSpec {
        name: "/compact",
        description: "Compact older local session context",
        usage: "/compact",
        argument_kind: ArgumentKind::None,
    },
    CommandSpec {
        name: "/model",
        description: "Set the model (and provider) to use",
        usage: "/model [<provider>:]<model>",
        argument_kind: ArgumentKind::Model,
    },
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CandidateKind {
    Slash,
    File,
    Directory,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Candidate {
    pub value: String,
    pub description: String,
    pub kind: CandidateKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompletionTarget {
    Command,
    Argument(ArgumentKind),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompletionContext {
    pub target: CompletionTarget,
    /// Byte offsets into the single-line command being completed.
    pub token_start: usize,
    pub token_end: usize,
    /// Text in the active token before the cursor.
    pub query: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompletionResult {
    pub context: CompletionContext,
    pub candidates: Vec<Candidate>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ParsedCommand {
    Auth,
    New,
    Load {
        selector: String,
    },
    Sessions,
    Export {
        destination: Option<String>,
    },
    Compact,
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
        "/auth" => {
            if words.next().is_some() {
                Err("usage: /auth".into())
            } else {
                Ok(ParsedCommand::Auth)
            }
        }
        "/new" => {
            if words.next().is_some() {
                Err("usage: /new".into())
            } else {
                Ok(ParsedCommand::New)
            }
        }
        "/load" => {
            let selector = words.collect::<Vec<_>>();
            if selector.len() > 1 {
                Err("usage: /load [<id>|latest|<path>]".into())
            } else {
                Ok(ParsedCommand::Load {
                    selector: selector.first().copied().unwrap_or("latest").to_owned(),
                })
            }
        }
        "/sessions" => {
            if words.next().is_some() {
                Err("usage: /sessions".into())
            } else {
                Ok(ParsedCommand::Sessions)
            }
        }
        "/export" => {
            let destination = words.collect::<Vec<_>>();
            if destination.len() > 1 {
                Err("usage: /export [<path>]".into())
            } else {
                Ok(ParsedCommand::Export {
                    destination: destination.first().map(|value| (*value).to_owned()),
                })
            }
        }
        "/compact" => {
            if words.next().is_some() {
                Err("usage: /compact".into())
            } else {
                Ok(ParsedCommand::Compact)
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

/// Find the command completion context at a character-based cursor column.
///
/// `tui-textarea` reports character columns while Rust string slicing uses
/// byte offsets, so the context deliberately stores byte offsets and performs
/// the conversion in one place.
pub fn completion_context(input: &str, cursor_col: usize) -> Option<CompletionContext> {
    if !is_command_input(input) {
        return None;
    }

    let cursor_byte = byte_index_at_char(input, cursor_col);
    let Some(command_end) = input.find(char::is_whitespace) else {
        return Some(CompletionContext {
            target: CompletionTarget::Command,
            token_start: 0,
            token_end: input.len(),
            query: input[..cursor_byte].to_owned(),
        });
    };

    if cursor_byte <= command_end {
        return Some(CompletionContext {
            target: CompletionTarget::Command,
            token_start: 0,
            token_end: command_end,
            query: input[..cursor_byte].to_owned(),
        });
    }

    let command = &input[..command_end];
    let spec = command_spec(command)?;
    if spec.argument_kind == ArgumentKind::None {
        return None;
    }

    let token_start = input[command_end..]
        .find(|character: char| !character.is_whitespace())
        .map(|offset| command_end + offset)
        .unwrap_or(input.len());
    let token_end = input[token_start..]
        .find(char::is_whitespace)
        .map(|offset| token_start + offset)
        .unwrap_or(input.len());

    // These commands accept one argument. Once the cursor has moved into a
    // later argument, it is better to close completion than to replace the
    // wrong token.
    if cursor_byte > token_end {
        return None;
    }

    let query_start = cursor_byte.clamp(token_start, token_end);
    Some(CompletionContext {
        target: CompletionTarget::Argument(spec.argument_kind),
        token_start,
        token_end,
        query: input[token_start..query_start].to_owned(),
    })
}

/// Return completion candidates for a cursor-aware command input.
pub fn candidates_at_cursor(
    input: &str,
    cursor_col: usize,
    providers: &[String],
    model_lists: &HashMap<String, Vec<ModelEntry>>,
    current_provider: &str,
    sessions: &[SessionListEntry],
) -> Option<CompletionResult> {
    let context = completion_context(input, cursor_col)?;
    let cursor_byte = byte_index_at_char(input, cursor_col);
    let query = token_prefix(input, &context, cursor_byte);
    let candidates = match context.target {
        CompletionTarget::Command => command_candidates(query),
        CompletionTarget::Argument(ArgumentKind::Model) => {
            model_candidates(query, providers, model_lists, current_provider)
        }
        CompletionTarget::Argument(ArgumentKind::Session) => session_candidates(query, sessions),
        CompletionTarget::Argument(ArgumentKind::Path)
        | CompletionTarget::Argument(ArgumentKind::None) => Vec::new(),
    };

    Some(CompletionResult {
        context,
        candidates,
    })
}

/// Compatibility wrapper for callers that only need candidates at the end of
/// the input and do not have cached session data.
pub fn candidates(
    input: &str,
    providers: &[String],
    model_lists: &HashMap<String, Vec<ModelEntry>>,
    current_provider: &str,
) -> Vec<Candidate> {
    candidates_at_cursor(
        input,
        input.chars().count(),
        providers,
        model_lists,
        current_provider,
        &[],
    )
    .map(|result| result.candidates)
    .unwrap_or_default()
}

pub fn command_spec(name: &str) -> Option<&'static CommandSpec> {
    COMMANDS
        .iter()
        .find(|command| command.name.eq_ignore_ascii_case(name))
}

/// Apply a slash-command candidate and return the new line plus a character
/// cursor column. The command name gets a separator only when its command
/// accepts arguments; file candidates get a separator when completed at the
/// end of their token.
pub fn apply_completion(
    line: &str,
    cursor_col: usize,
    context: &CompletionContext,
    candidate: &Candidate,
) -> Option<(String, usize)> {
    if context.token_start > context.token_end
        || context.token_end > line.len()
        || !line.is_char_boundary(context.token_start)
        || !line.is_char_boundary(context.token_end)
    {
        return None;
    }

    let cursor_byte = byte_index_at_char(line, cursor_col);
    let before = &line[..context.token_start];
    let after = &line[context.token_end..];
    let next_is_whitespace = after.chars().next().is_some_and(char::is_whitespace);
    let at_token_end = cursor_byte == context.token_end;

    let separator = match context.target {
        CompletionTarget::Command => command_spec(&candidate.value)
            .is_some_and(|command| command.argument_kind != ArgumentKind::None)
            .then_some(" ")
            .filter(|_| !next_is_whitespace)
            .unwrap_or(""),
        CompletionTarget::Argument(_) if candidate.kind == CandidateKind::File && at_token_end => {
            if next_is_whitespace {
                ""
            } else {
                " "
            }
        }
        CompletionTarget::Argument(_) => "",
    };

    let replacement = format!("{}{}{}{}", before, candidate.value, separator, after);
    let new_cursor =
        before.chars().count() + candidate.value.chars().count() + separator.chars().count();
    Some((replacement, new_cursor))
}

fn command_candidates(query: &str) -> Vec<Candidate> {
    let query = query.to_ascii_lowercase();
    COMMANDS
        .iter()
        .filter(|command| command.name.to_ascii_lowercase().starts_with(&query))
        .map(|command| Candidate {
            value: command.name.to_owned(),
            description: command.description.to_owned(),
            kind: CandidateKind::Slash,
        })
        .collect()
}

fn model_candidates(
    token: &str,
    providers: &[String],
    model_lists: &HashMap<String, Vec<ModelEntry>>,
    current_provider: &str,
) -> Vec<Candidate> {
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
                            kind: CandidateKind::Slash,
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
                        kind: CandidateKind::Slash,
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

fn session_candidates(query: &str, sessions: &[SessionListEntry]) -> Vec<Candidate> {
    let query = query.to_ascii_lowercase();
    let mut candidates = Vec::new();
    if "latest".starts_with(&query) {
        candidates.push(Candidate {
            value: "latest".into(),
            description: "most recent non-empty session".into(),
            kind: CandidateKind::Slash,
        });
    }

    let mut seen = HashSet::from(["latest".to_owned()]);
    for session in sessions {
        let value = if session.short_id.is_empty() {
            session.id.clone()
        } else {
            session.short_id.clone()
        };
        if value.is_empty()
            || !value.to_ascii_lowercase().starts_with(&query)
            || !seen.insert(value.clone())
        {
            continue;
        }
        let title = session
            .title
            .as_deref()
            .filter(|title| !title.trim().is_empty())
            .unwrap_or("untitled");
        candidates.push(Candidate {
            value,
            description: format!("{title} · {}", session.updated_at),
            kind: CandidateKind::Slash,
        });
    }
    candidates.sort_by(|a, b| {
        (a.value != "latest")
            .cmp(&(b.value != "latest"))
            .then_with(|| {
                a.value
                    .to_ascii_lowercase()
                    .cmp(&b.value.to_ascii_lowercase())
            })
            .then_with(|| a.value.cmp(&b.value))
    });
    candidates
}

fn token_prefix<'a>(input: &'a str, context: &CompletionContext, cursor_byte: usize) -> &'a str {
    let start = context.token_start.min(input.len());
    let end = cursor_byte.clamp(start, context.token_end.min(input.len()));
    &input[start..end]
}

fn byte_index_at_char(input: &str, column: usize) -> usize {
    input
        .char_indices()
        .nth(column)
        .map(|(index, _)| index)
        .unwrap_or(input.len())
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

    fn session(short_id: &str, title: Option<&str>) -> SessionListEntry {
        SessionListEntry {
            id: format!("{short_id}-full"),
            short_id: short_id.into(),
            title: title.map(str::to_owned),
            updated_at: "2026-08-13 12:00".into(),
            workspace: "/workspace".into(),
            provider: Some("openrouter".into()),
            model: Some("demo".into()),
        }
    }

    #[test]
    fn command_candidates_filter_the_command_token_case_insensitively() {
        let lists = HashMap::new();
        let all = candidates("/", &providers(), &lists, "opencode-go");
        assert!(all.len() >= 7);
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
    fn command_completion_adds_the_argument_space_and_keeps_the_cursor_after_it() {
        let lists = HashMap::new();
        let result = candidates_at_cursor(
            "/mod",
            "/mod".chars().count(),
            &providers(),
            &lists,
            "opencode-go",
            &[],
        )
        .unwrap();
        assert_eq!(result.context.target, CompletionTarget::Command);
        let candidate = result
            .candidates
            .iter()
            .find(|candidate| candidate.value == "/model")
            .unwrap();
        let (line, cursor) = apply_completion("/mod", 4, &result.context, candidate).unwrap();
        assert_eq!(line, "/model ");
        assert_eq!(cursor, 7);

        let result =
            candidates_at_cursor("/au", 3, &providers(), &lists, "opencode-go", &[]).unwrap();
        let candidate = result
            .candidates
            .iter()
            .find(|candidate| candidate.value == "/auth")
            .unwrap();
        let (line, cursor) = apply_completion("/au", 3, &result.context, candidate).unwrap();
        assert_eq!(line, "/auth");
        assert_eq!(cursor, 5);
    }

    #[test]
    fn argument_completion_replaces_only_the_active_token_and_preserves_suffix() {
        let lists = HashMap::from([(
            "opencode-go".into(),
            vec![model("gpt-new", Some("GPT New"), None)],
        )]);
        let input = "/model gpt tail";
        let cursor = "/model gpt".chars().count();
        let result =
            candidates_at_cursor(input, cursor, &providers(), &lists, "opencode-go", &[]).unwrap();
        let candidate = result
            .candidates
            .iter()
            .find(|candidate| candidate.value == "gpt-new")
            .unwrap();
        let (line, new_cursor) =
            apply_completion(input, cursor, &result.context, candidate).unwrap();
        assert_eq!(line, "/model gpt-new tail");
        assert_eq!(new_cursor, "/model gpt-new".chars().count());
    }

    #[test]
    fn provider_completion_leaves_the_cursor_after_the_colon() {
        let lists = HashMap::new();
        let input = "/model open";
        let result = candidates_at_cursor(
            input,
            input.chars().count(),
            &providers(),
            &lists,
            "opencode-go",
            &[],
        )
        .unwrap();
        let candidate = result
            .candidates
            .iter()
            .find(|candidate| candidate.value == "openrouter:")
            .unwrap();
        let (line, cursor) =
            apply_completion(input, input.chars().count(), &result.context, candidate).unwrap();
        assert_eq!(line, "/model openrouter:");
        assert_eq!(cursor, line.chars().count());
    }

    #[test]
    fn path_completion_adds_a_space_after_a_file_but_not_a_directory() {
        let lists = HashMap::new();
        let input = "/export src/ma";
        let result = candidates_at_cursor(
            input,
            input.chars().count(),
            &providers(),
            &lists,
            "opencode-go",
            &[],
        )
        .unwrap();
        let file = Candidate {
            value: "src/main.rs".into(),
            description: "file".into(),
            kind: CandidateKind::File,
        };
        let (line, cursor) =
            apply_completion(input, input.chars().count(), &result.context, &file).unwrap();
        assert_eq!(line, "/export src/main.rs ");
        assert_eq!(cursor, line.chars().count());

        let directory = Candidate {
            value: "src/".into(),
            description: "directory".into(),
            kind: CandidateKind::Directory,
        };
        let (line, cursor) =
            apply_completion(input, input.chars().count(), &result.context, &directory).unwrap();
        assert_eq!(line, "/export src/");
        assert_eq!(cursor, line.chars().count());
    }

    #[test]
    fn session_candidates_include_latest_and_cached_sessions() {
        let lists = HashMap::new();
        let sessions = vec![session("abc123", Some("Fix completion"))];
        let result = candidates_at_cursor(
            "/load ",
            "/load ".chars().count(),
            &providers(),
            &lists,
            "opencode-go",
            &sessions,
        )
        .unwrap();
        let values = result
            .candidates
            .iter()
            .map(|candidate| candidate.value.as_str())
            .collect::<Vec<_>>();
        assert_eq!(values.first(), Some(&"latest"));
        assert!(values.contains(&"abc123"));
        assert!(
            result
                .candidates
                .iter()
                .any(|candidate| candidate.description.contains("Fix completion"))
        );
    }

    #[test]
    fn unicode_cursor_columns_are_kept_character_based() {
        let lists = HashMap::new();
        let input = "/model ☃";
        let result = candidates_at_cursor(
            input,
            input.chars().count(),
            &providers(),
            &lists,
            "opencode-go",
            &[],
        )
        .unwrap();
        let candidate = Candidate {
            value: "replacement".into(),
            description: String::new(),
            kind: CandidateKind::Slash,
        };
        let (line, cursor) =
            apply_completion(input, input.chars().count(), &result.context, &candidate).unwrap();
        assert_eq!(line, "/model replacement");
        assert_eq!(cursor, line.chars().count());
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
        assert_eq!(parse_command("/auth"), Ok(ParsedCommand::Auth));
        assert_eq!(parse_command("/auth now"), Err("usage: /auth".into()));
        assert_eq!(parse_command("/new"), Ok(ParsedCommand::New));
        assert_eq!(
            parse_command("/load"),
            Ok(ParsedCommand::Load {
                selector: "latest".into()
            })
        );
        assert_eq!(
            parse_command("/load 1234"),
            Ok(ParsedCommand::Load {
                selector: "1234".into()
            })
        );
        assert_eq!(parse_command("/sessions"), Ok(ParsedCommand::Sessions));
        assert_eq!(
            parse_command("/export transcript.jsonl"),
            Ok(ParsedCommand::Export {
                destination: Some("transcript.jsonl".into())
            })
        );
        assert_eq!(parse_command("/compact"), Ok(ParsedCommand::Compact));
        assert_eq!(parse_command("/auth"), Ok(ParsedCommand::Auth));
        assert_eq!(parse_command("/auth now"), Err("usage: /auth".into()));
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
