//! Pure slash-command parsing and completion candidate generation.
//!
//! This module intentionally has no terminal or asynchronous code.  The app
//! owns the editor and cached lists, while the agent remains responsible for
//! validating providers and executing the resulting semantic messages.

use crate::{ModelEntry, SessionListEntry, SkillEntry};
use std::collections::{HashMap, HashSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArgumentKind {
    None,
    Model,
    Session,
    Path,
    Skill,
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
        name: "/help",
        description: "List available slash commands",
        usage: "/help",
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
    CommandSpec {
        name: "/usage",
        description: "Show active subscription allowance usage",
        usage: "/usage",
        argument_kind: ArgumentKind::None,
    },
    CommandSpec {
        name: "/skill",
        description: "Start a turn from a discovered skill",
        usage: "/skill <name>",
        argument_kind: ArgumentKind::Skill,
    },
    CommandSpec {
        name: "/skills",
        description: "List discovered skills and diagnostics",
        usage: "/skills",
        argument_kind: ArgumentKind::None,
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
    Help,
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
    Usage,
    Skills,
    /// `/skill <name>` or a bare `/<name>` alias. `name` is the skill's
    /// catalog name (without the leading slash); `alias` distinguishes the
    /// two spellings for the echoed notice.
    InvokeSkill {
        name: String,
        alias: bool,
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
        "/help" => {
            if words.next().is_some() {
                Err("usage: /help".into())
            } else {
                Ok(ParsedCommand::Help)
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
        "/usage" => {
            if words.next().is_some() {
                Err("usage: /usage".into())
            } else {
                Ok(ParsedCommand::Usage)
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

/// Parse a slash command against the static command set plus an optional
/// dynamic skill catalog. Skill aliases (`/<skill-name>`) are only recognized
/// when the name does not collide with a static command, so `/model` and
/// `/skill` can never be shadowed by a skill.
pub fn parse_command_with_skills(
    text: &str,
    skills: &[SkillEntry],
) -> Result<ParsedCommand, String> {
    let input = text.trim();
    let mut words = input.split_whitespace();
    let command = words.next().unwrap_or("");
    let rest = words.collect::<Vec<_>>();
    match command.to_ascii_lowercase().as_str() {
        "/skill" => {
            let [name] = rest[..] else {
                return Err("usage: /skill <name>".into());
            };
            let Some(entry) = skill_entry(skills, name) else {
                return Err(format!("unknown skill: {name}"));
            };
            Ok(ParsedCommand::InvokeSkill {
                name: entry.name.clone(),
                alias: false,
            })
        }
        "/skills" => {
            if !rest.is_empty() {
                return Err("usage: /skills".into());
            }
            Ok(ParsedCommand::Skills)
        }
        other => {
            if let Some(name) = other.strip_prefix('/')
                && !name.is_empty()
                && command_spec(command).is_none()
                && let Some(entry) = skill_entry(skills, name)
            {
                return Ok(ParsedCommand::InvokeSkill {
                    name: entry.name.clone(),
                    alias: true,
                });
            }
            parse_command(text)
        }
    }
}

/// The skill entry matching `name` case-insensitively, if any.
fn skill_entry<'a>(skills: &'a [SkillEntry], name: &str) -> Option<&'a SkillEntry> {
    skills
        .iter()
        .find(|entry| entry.name.eq_ignore_ascii_case(name))
}

/// Find the command completion context at a character-based cursor column.
///
/// The context deliberately stores byte offsets (used for Rust string
/// slicing) and performs the char-column conversion in one place.
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
    skills: &[SkillEntry],
) -> Option<CompletionResult> {
    let context = completion_context(input, cursor_col)?;
    let cursor_byte = byte_index_at_char(input, cursor_col);
    let query = token_prefix(input, &context, cursor_byte);
    let candidates = match context.target {
        CompletionTarget::Command => command_candidates(query, skills),
        CompletionTarget::Argument(ArgumentKind::Model) => {
            model_candidates(query, providers, model_lists, current_provider)
        }
        CompletionTarget::Argument(ArgumentKind::Session) => session_candidates(query, sessions),
        CompletionTarget::Argument(ArgumentKind::Skill) => skill_candidates(query, skills),
        CompletionTarget::Argument(ArgumentKind::Path)
        | CompletionTarget::Argument(ArgumentKind::None) => Vec::new(),
    };

    Some(CompletionResult {
        context,
        candidates,
    })
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
    skills: &[SkillEntry],
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
        CompletionTarget::Command => {
            let takes_argument = match command_spec(&candidate.value) {
                Some(command) => command.argument_kind != ArgumentKind::None,
                // A skill alias keeps the argument phase open so `/skill <name>`
                // and `/<skill>` behave identically after acceptance.
                None => candidate
                    .value
                    .strip_prefix('/')
                    .and_then(|name| skill_entry(skills, name))
                    .is_some(),
            };
            takes_argument
                .then_some(" ")
                .filter(|_| !next_is_whitespace)
                .unwrap_or("")
        }
        // `@` file references and command path arguments end the token with a
        // space once completed, so typing continues on a fresh token.
        // Directories keep the completion context alive instead.
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

fn command_candidates(query: &str, skills: &[SkillEntry]) -> Vec<Candidate> {
    let query = query.to_ascii_lowercase();
    let mut candidates: Vec<Candidate> = COMMANDS
        .iter()
        .filter(|command| command.name.to_ascii_lowercase().starts_with(&query))
        .map(|command| Candidate {
            value: command.name.to_owned(),
            description: command.description.to_owned(),
            kind: CandidateKind::Slash,
        })
        .collect();
    // Skill aliases join the command list only when the name is free —
    // static commands (including `/skill` itself) always win.
    for skill in skills {
        let value = format!("/{}", skill.name);
        if !value.to_ascii_lowercase().starts_with(&query) {
            continue;
        }
        if candidates
            .iter()
            .any(|candidate| candidate.value.eq_ignore_ascii_case(&value))
        {
            continue;
        }
        candidates.push(Candidate {
            value,
            description: skill.description.clone(),
            kind: CandidateKind::Slash,
        });
    }
    candidates
}

fn skill_candidates(query: &str, skills: &[SkillEntry]) -> Vec<Candidate> {
    let query = query.to_ascii_lowercase();
    skills
        .iter()
        .filter(|skill| skill.name.to_ascii_lowercase().starts_with(&query))
        .map(|skill| Candidate {
            value: skill.name.clone(),
            description: skill.description.clone(),
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
    // score, provider candidate, current provider, candidate
    let mut results = Vec::<(i32, bool, bool, Candidate)>::new();
    let mut seen = HashSet::new();

    if let Some((prefix, query)) = token.split_once(':') {
        let Some(provider) = providers
            .iter()
            .find(|provider| provider.eq_ignore_ascii_case(prefix))
        else {
            // A colon commits the token to provider-qualified syntax. Do not
            // silently fall back to the active provider for a typo.
            return Vec::new();
        };
        if let Some(models) = model_list(model_lists, provider) {
            push_model_candidates(
                &mut results,
                &mut seen,
                provider,
                query,
                models,
                current_provider,
            );
        }
    } else {
        // Provider names and every cached catalog share one search. Model
        // values are always provider-qualified here: when several catalogs
        // contain similar IDs, the completion itself makes the destination
        // unambiguous instead of relying on a dim description.
        for provider in providers {
            if let Some(score) = fuzzy_score(provider, token) {
                let value = format!("{provider}:");
                if seen.insert(value.clone()) {
                    results.push((
                        score,
                        true,
                        provider.eq_ignore_ascii_case(current_provider),
                        Candidate {
                            value,
                            description: "provider".into(),
                            kind: CandidateKind::Slash,
                        },
                    ));
                }
            }
            if let Some(models) = model_list(model_lists, provider) {
                push_model_candidates(
                    &mut results,
                    &mut seen,
                    provider,
                    token,
                    models,
                    current_provider,
                );
            }
        }
    }

    results.sort_by(
        |(score_a, provider_a, current_a, candidate_a),
         (score_b, provider_b, current_b, candidate_b)| {
            score_b
                .cmp(score_a)
                .then_with(|| provider_b.cmp(provider_a))
                .then_with(|| current_b.cmp(current_a))
                .then_with(|| {
                    candidate_a
                        .value
                        .to_ascii_lowercase()
                        .cmp(&candidate_b.value.to_ascii_lowercase())
                })
                .then_with(|| candidate_a.value.cmp(&candidate_b.value))
        },
    );
    results
        .into_iter()
        .map(|(_, _, _, candidate)| candidate)
        .collect()
}

fn model_list<'a>(
    model_lists: &'a HashMap<String, Vec<ModelEntry>>,
    provider: &str,
) -> Option<&'a [ModelEntry]> {
    model_lists
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case(provider))
        .map(|(_, models)| models.as_slice())
}

fn push_model_candidates(
    results: &mut Vec<(i32, bool, bool, Candidate)>,
    seen: &mut HashSet<String>,
    provider: &str,
    query: &str,
    models: &[ModelEntry],
    current_provider: &str,
) {
    for model in models {
        let score = fuzzy_score(&model.id, query)
            .into_iter()
            .chain(
                model
                    .name
                    .as_deref()
                    .and_then(|name| fuzzy_score(name, query)),
            )
            .max();
        let Some(score) = score else {
            continue;
        };
        let value = format!("{provider}:{}", model.id);
        if seen.insert(value.clone()) {
            results.push((
                score,
                false,
                provider.eq_ignore_ascii_case(current_provider),
                Candidate {
                    value,
                    description: model_description(model),
                    kind: CandidateKind::Slash,
                },
            ));
        }
    }
}

/// Rank exact, prefix, substring, then ordered-subsequence matches. The final
/// tier makes compact queries such as `gpt56` useful for IDs containing
/// punctuation without letting a loose subsequence outrank a literal match.
fn fuzzy_score(candidate: &str, query: &str) -> Option<i32> {
    let candidate = candidate.to_ascii_lowercase();
    let query = query.to_ascii_lowercase();
    if query.is_empty() {
        return Some(0);
    }
    if candidate == query {
        return Some(10_000);
    }
    if candidate.starts_with(&query) {
        return Some(8_000 - candidate.len().saturating_sub(query.len()) as i32);
    }
    if let Some(index) = candidate.find(&query) {
        return Some(6_000 - index.min(1_000) as i32);
    }

    let candidate_chars = candidate.chars().collect::<Vec<_>>();
    let query_chars = query.chars().collect::<Vec<_>>();
    let mut positions = Vec::with_capacity(query_chars.len());
    let mut next = 0usize;
    for query_char in query_chars.iter().copied() {
        let offset = candidate_chars[next..]
            .iter()
            .position(|candidate_char| *candidate_char == query_char)?;
        let position = next + offset;
        positions.push(position);
        next = position + 1;
    }
    let first = positions[0];
    let span = positions.last().copied().unwrap_or(first) - first + 1;
    let gaps = span.saturating_sub(query_chars.len());
    Some(3_000 - first.min(500) as i32 - (gaps.min(200) * 5) as i32)
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

/// The longest common character prefix shared by every candidate value, or
/// empty when they agree on nothing. Used to extend the typed token to the
/// common prefix on Tab and to drive the fish-style ghost preview.
pub fn common_prefix(candidates: &[Candidate]) -> String {
    let mut iter = candidates.iter().map(|candidate| candidate.value.as_str());
    let Some(first) = iter.next() else {
        return String::new();
    };
    let mut shared = first.to_owned();
    for value in iter {
        while !shared.is_empty() && !value.starts_with(shared.as_str()) {
            shared.pop();
        }
    }
    shared
}

/// The untyped remainder of a candidate value relative to the active token at
/// `cursor_col`. Mirrors `apply_completion`'s token math: the replacement
/// covers the whole token (`token_start`..`token_end`), so given the token
/// `open` and the candidate `openrouter:` the suffix is `router:`. Returns
/// empty when the token is already the candidate or the completion is closed.
pub fn candidate_suffix(
    input: &str,
    cursor_col: usize,
    context: &CompletionContext,
    candidate: &Candidate,
) -> String {
    if context.token_start > context.token_end
        || context.token_end > input.len()
        || !input.is_char_boundary(context.token_start)
        || !input.is_char_boundary(context.token_end)
    {
        return String::new();
    }
    let cursor_byte = byte_index_at_char(input, cursor_col);
    let end = cursor_byte.clamp(context.token_start, context.token_end);
    if !candidate
        .value
        .starts_with(&input[context.token_start..end])
    {
        return String::new();
    }
    candidate.value[end - context.token_start..].to_owned()
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
            provider: None,
            model: None,
        }
    }

    /// Complete at the end of `input` with no cached sessions.
    fn candidates(
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
            &[],
        )
        .map(|result| result.candidates)
        .unwrap_or_default()
    }

    fn skill(name: &str, description: &str) -> SkillEntry {
        SkillEntry {
            name: name.into(),
            description: description.into(),
        }
    }

    fn skills() -> Vec<SkillEntry> {
        vec![
            skill("greeter", "Says hello"),
            skill("model", "Collides with /model"),
            skill("skill", "Collides with /skill"),
            skill("skills", "Collides with /skills"),
        ]
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
        assert!(values.contains(&"opencode-go:gpt-5.6-luna".into()));
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
            &[],
        )
        .unwrap();
        assert_eq!(result.context.target, CompletionTarget::Command);
        let candidate = result
            .candidates
            .iter()
            .find(|candidate| candidate.value == "/model")
            .unwrap();
        let (line, cursor) = apply_completion("/mod", 4, &result.context, candidate, &[]).unwrap();
        assert_eq!(line, "/model ");
        assert_eq!(cursor, 7);
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
            candidates_at_cursor(input, cursor, &providers(), &lists, "opencode-go", &[], &[])
                .unwrap();
        let candidate = result
            .candidates
            .iter()
            .find(|candidate| candidate.value == "opencode-go:gpt-new")
            .unwrap();
        let (line, new_cursor) =
            apply_completion(input, cursor, &result.context, candidate, &[]).unwrap();
        assert_eq!(line, "/model opencode-go:gpt-new tail");
        assert_eq!(new_cursor, "/model opencode-go:gpt-new".chars().count());
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
            &[],
        )
        .unwrap();
        let candidate = result
            .candidates
            .iter()
            .find(|candidate| candidate.value == "openrouter:")
            .unwrap();
        let (line, cursor) = apply_completion(
            input,
            input.chars().count(),
            &result.context,
            candidate,
            &[],
        )
        .unwrap();
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
            &[],
        )
        .unwrap();
        let file = Candidate {
            value: "src/main.rs".into(),
            description: "file".into(),
            kind: CandidateKind::File,
        };
        let (line, cursor) =
            apply_completion(input, input.chars().count(), &result.context, &file, &[]).unwrap();
        assert_eq!(line, "/export src/main.rs ");
        assert_eq!(cursor, line.chars().count());

        let directory = Candidate {
            value: "src/".into(),
            description: "directory".into(),
            kind: CandidateKind::Directory,
        };
        let (line, cursor) = apply_completion(
            input,
            input.chars().count(),
            &result.context,
            &directory,
            &[],
        )
        .unwrap();
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
            &[],
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
            &[],
        )
        .unwrap();
        let candidate = Candidate {
            value: "replacement".into(),
            description: String::new(),
            kind: CandidateKind::Slash,
        };
        let (line, cursor) = apply_completion(
            input,
            input.chars().count(),
            &result.context,
            &candidate,
            &[],
        )
        .unwrap();
        assert_eq!(line, "/model replacement");
        assert_eq!(cursor, line.chars().count());
    }

    #[test]
    fn provider_and_model_fuzzy_matching_is_ranked_and_qualified() {
        let lists = HashMap::from([(
            "opencode-go".into(),
            vec![
                model("gpt-5.6-luna", Some("GPT 5.6 Luna"), Some(400_000)),
                model("minimax-m3", Some("MiniMax"), None),
            ],
        )]);
        let prefix = candidates("/model open", &providers(), &lists, "opencode-go");
        assert_eq!(prefix[0].value, "openrouter:");
        assert!(
            prefix
                .iter()
                .any(|candidate| candidate.value == "opencode-go:")
        );
        // Ordered subsequences ignore punctuation (`g56l` → `gpt-5.6-luna`),
        // and the inserted value always identifies its provider.
        let filtered = candidates("/model g56l", &providers(), &lists, "opencode-go");
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].value, "opencode-go:gpt-5.6-luna");
        assert!(filtered[0].description.contains("GPT 5.6 Luna"));
        assert!(filtered[0].description.contains("400k ctx"));
    }

    #[test]
    fn unqualified_search_spans_cached_providers_and_keeps_them_distinct() {
        let lists = HashMap::from([
            (
                "opencode-go".into(),
                vec![model("claude-sonnet", Some("Claude Sonnet"), None)],
            ),
            (
                "openrouter".into(),
                vec![model(
                    "anthropic/claude-sonnet",
                    Some("Claude Sonnet"),
                    None,
                )],
            ),
        ]);
        let values = candidates("/model sonnet", &providers(), &lists, "opencode-go")
            .into_iter()
            .map(|candidate| candidate.value)
            .collect::<Vec<_>>();
        assert_eq!(values.len(), 2);
        assert!(values.contains(&"opencode-go:claude-sonnet".into()));
        assert!(values.contains(&"openrouter:anthropic/claude-sonnet".into()));
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
    fn help_parses_and_appears_in_command_candidates() {
        assert_eq!(parse_command("/help"), Ok(ParsedCommand::Help));
        assert_eq!(parse_command("/help now"), Err("usage: /help".into()));
        assert_eq!(parse_command("/HELP"), Ok(ParsedCommand::Help));

        let lists = HashMap::new();
        assert!(
            candidates("/", &providers(), &lists, "opencode-go")
                .iter()
                .any(|candidate| candidate.value == "/help")
        );
        assert!(
            candidates("/he", &providers(), &lists, "opencode-go")
                .iter()
                .any(|candidate| candidate.value == "/help")
        );
    }

    fn candidate(value: &str) -> Candidate {
        Candidate {
            value: value.into(),
            description: String::new(),
            kind: CandidateKind::Slash,
        }
    }

    #[test]
    fn common_prefix_extends_shared_command_and_model_text() {
        assert_eq!(common_prefix(&[]), "");
        let commands = vec![candidate("/model"), candidate("/new")];
        assert_eq!(common_prefix(&commands), "/");
        let providers = vec![candidate("opencode-go:"), candidate("openrouter:")];
        assert_eq!(common_prefix(&providers), "open");
        // Disjoint values share nothing.
        assert_eq!(common_prefix(&[candidate("abc"), candidate("xyz")]), "");
    }

    #[test]
    fn candidate_suffix_returns_the_untyped_token_remainder() {
        let lists = HashMap::new();

        // Provider: `/model open` completes to `openrouter:` → suffix.
        let input = "/model open";
        let cursor = input.chars().count();
        let result =
            candidates_at_cursor(input, cursor, &providers(), &lists, "opencode-go", &[], &[])
                .unwrap();
        let provider_candidate = result
            .candidates
            .iter()
            .find(|candidate| candidate.value == "opencode-go:")
            .unwrap();
        assert_eq!(
            candidate_suffix(input, cursor, &result.context, provider_candidate),
            "code-go:"
        );

        // No model typed yet: the full id is the suffix.
        let suffix_input = "/model ";
        let suffix_cursor = suffix_input.chars().count();
        let context = completion_context(suffix_input, suffix_cursor).unwrap();
        let model = candidate("gpt-5.6-luna");
        assert_eq!(
            candidate_suffix(suffix_input, suffix_cursor, &context, &model),
            "gpt-5.6-luna"
        );

        // Explicit `provider:` token keeps the prefix; the suffix starts
        // past the colon.
        let explicit = "/model openrouter:";
        let explicit_cursor = explicit.chars().count();
        let explicit_context = completion_context(explicit, explicit_cursor).unwrap();
        let explicit_model = candidate("openrouter:openai/gpt-5.6-luna");
        assert_eq!(
            candidate_suffix(
                explicit,
                explicit_cursor,
                &explicit_context,
                &explicit_model
            ),
            "openai/gpt-5.6-luna"
        );
    }

    #[test]
    fn parses_success_and_usage_errors() {
        assert_eq!(parse_command("/auth"), Err("unknown command: /auth".into()));
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
        assert_eq!(parse_command("/usage"), Ok(ParsedCommand::Usage));
        assert_eq!(parse_command("/USAGE"), Ok(ParsedCommand::Usage));
        assert_eq!(parse_command("/usage now"), Err("usage: /usage".into()));
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

    #[test]
    fn skill_commands_parse_and_shadow_safely() {
        let skills = skills();

        // `/skill <name>` requires exactly one known skill name.
        assert_eq!(
            parse_command_with_skills("/skill greeter", &skills),
            Ok(ParsedCommand::InvokeSkill {
                name: "greeter".into(),
                alias: false
            })
        );
        assert_eq!(
            parse_command_with_skills("/skill", &skills),
            Err("usage: /skill <name>".into())
        );
        assert_eq!(
            parse_command_with_skills("/skill greeter now", &skills),
            Err("usage: /skill <name>".into())
        );
        assert_eq!(
            parse_command_with_skills("/skill nope", &skills),
            Err("unknown skill: nope".into())
        );
        // Case-insensitive lookup returns the catalog name.
        assert_eq!(
            parse_command_with_skills("/skill GREETER", &skills),
            Ok(ParsedCommand::InvokeSkill {
                name: "greeter".into(),
                alias: false
            })
        );

        // `/skills` takes no arguments.
        assert_eq!(
            parse_command_with_skills("/skills", &skills),
            Ok(ParsedCommand::Skills)
        );
        assert_eq!(
            parse_command_with_skills("/skills now", &skills),
            Err("usage: /skills".into())
        );

        // Bare skill aliases parse to invocations…
        assert_eq!(
            parse_command_with_skills("/greeter", &skills),
            Ok(ParsedCommand::InvokeSkill {
                name: "greeter".into(),
                alias: true
            })
        );
        // …but never shadow static commands, including the skill commands.
        assert_eq!(
            parse_command_with_skills("/model", &skills),
            Ok(ParsedCommand::SetModel {
                provider: None,
                model: String::new()
            })
        );
        assert_eq!(
            parse_command_with_skills("/skill", &skills),
            Err("usage: /skill <name>".into())
        );
        assert_eq!(
            parse_command_with_skills("/skills", &skills),
            Ok(ParsedCommand::Skills)
        );
        // Unknown names still fall through to the static error.
        assert_eq!(
            parse_command_with_skills("/foo", &skills),
            Err("unknown command: /foo".into())
        );
        // Static commands still parse with an empty catalog.
        assert_eq!(
            parse_command_with_skills("/help", &[]),
            Ok(ParsedCommand::Help)
        );
    }

    #[test]
    fn skill_aliases_and_arguments_complete() {
        let skills = skills();

        // Command phase: static commands plus non-colliding skill aliases.
        let result = candidates_at_cursor(
            "/",
            1,
            &providers(),
            &HashMap::new(),
            "opencode-go",
            &[],
            &skills,
        )
        .unwrap();
        let values = result
            .candidates
            .iter()
            .map(|candidate| candidate.value.as_str())
            .collect::<Vec<_>>();
        assert!(values.contains(&"/skill") && values.contains(&"/skills"));
        assert!(values.contains(&"/greeter"));
        // Colliding names appear exactly once: the static command, with no
        // duplicate skill alias.
        for collided in ["/model", "/skill", "/skills"] {
            assert_eq!(
                values.iter().filter(|value| **value == collided).count(),
                1,
                "{collided} should appear exactly once"
            );
        }

        // `/skill ` completes skill names with descriptions.
        let result = candidates_at_cursor(
            "/skill ",
            "/skill ".chars().count(),
            &providers(),
            &HashMap::new(),
            "opencode-go",
            &[],
            &skills,
        )
        .unwrap();
        assert_eq!(
            result.context.target,
            CompletionTarget::Argument(ArgumentKind::Skill)
        );
        let greeter = result
            .candidates
            .iter()
            .find(|candidate| candidate.value == "greeter")
            .unwrap();
        assert_eq!(greeter.description, "Says hello");

        // Prefix filtering works for both spellings.
        let result = candidates_at_cursor(
            "/gree",
            5,
            &providers(),
            &HashMap::new(),
            "opencode-go",
            &[],
            &skills,
        )
        .unwrap();
        assert_eq!(result.candidates.len(), 1);
        assert_eq!(result.candidates[0].value, "/greeter");

        // Accepting a skill alias keeps the argument phase open (skill takes
        // an argument), so the cursor lands after the trailing space.
        let (line, cursor) =
            apply_completion("/gree", 5, &result.context, &result.candidates[0], &skills).unwrap();
        assert_eq!(line, "/greeter ");
        assert_eq!(cursor, line.chars().count());
    }
}
