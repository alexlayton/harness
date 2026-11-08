//! Inline completion: slash commands, model/session/skill arguments, and
//! debounced `@` file references, driving the ghost text and hint row.

use super::*;

/// An open completion list driving the fish-style ghost preview, the Tab
/// accept, and the one-row hint below the input.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Completion {
    pub(super) context: CompletionContext,
    pub(super) candidates: Vec<Candidate>,
    pub(super) selected: usize,
    pub(super) kind: CompletionKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CompletionKind {
    Slash,
    Model,
    Path,
}

/// A debounced path-completion scan result delivered back over the run-loop
/// channel so a large directory tree never blocks the UI thread. `at_prefix`
/// is set when the scan was for an `@` file reference rather than a command
/// argument.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct PathCompletionResult {
    pub(super) generation: u64,
    pub(super) context: CompletionContext,
    pub(super) candidates: Vec<Candidate>,
    pub(super) at_prefix: Option<AtPrefix>,
}

impl CrossTerm {
    // ------------------------------------------------------------------
    // Inline completion
    // ------------------------------------------------------------------

    /// Rebuild the completion list for the current draft and cursor. Cheap
    /// except for path completion, which spawns a debounced blocking scan and
    /// delivers results over `path_completion_rx`. Pure callers (tests) drive
    /// this directly; the run loop calls it after every input mutation.
    pub(super) fn refresh_completion(&mut self) {
        if commands::is_command_input(&self.input) {
            self.refresh_command_completion();
            return;
        }
        // Outside slash commands, an active `@` token completes file
        // references against a debounced workspace scan.
        let Some(prefix) = paths::extract_at_prefix(&self.input, self.cursor_char_col()) else {
            self.close_completion();
            return;
        };
        // Seed the request with the open list so the hint never blanks out
        // while the debounced scan for the refined query is in flight; the
        // fresh results replace them on arrival.
        let (initial, preferred) = self
            .completion
            .as_ref()
            .filter(|completion| completion.kind == CompletionKind::Path)
            .map(|completion| {
                let preferred = completion.candidates.get(completion.selected).cloned();
                (completion.candidates.clone(), preferred)
            })
            .unwrap_or_default();
        self.request_path_completion(
            CompletionContext {
                target: CompletionTarget::Argument(ArgumentKind::Path),
                token_start: prefix.token_start,
                token_end: prefix.token_end,
                query: prefix.query.clone(),
            },
            initial,
            preferred.map(|candidate| candidate.value),
            Some(prefix),
        );
    }

    /// [`Self::refresh_completion`] half that handles `/command` drafts:
    /// static candidates plus, for path arguments, a debounced scan.
    pub(super) fn refresh_command_completion(&mut self) {
        let old_value = self.completion.as_ref().and_then(|completion| {
            completion
                .candidates
                .get(completion.selected)
                .map(|candidate| candidate.value.clone())
        });
        let cursor_col = self.cursor_char_col();
        let Some(result) = commands::candidates_at_cursor(
            &self.input,
            cursor_col,
            &self.providers,
            &self.model_lists,
            &self.provider,
            &self.session_candidates,
            &self.skills,
        ) else {
            self.close_completion();
            return;
        };

        // Path arguments scan the filesystem on a debounced task. Do not
        // cancel first: `request_path_completion` keeps the current scan and
        // its results when the token context is unchanged (as `@` paths do),
        // so Tab after the results arrive accepts them instead of clearing
        // them and starting again.
        if matches!(
            result.context.target,
            CompletionTarget::Argument(ArgumentKind::Path | ArgumentKind::Session),
        ) {
            self.request_path_completion(result.context, result.candidates, old_value, None);
            return;
        }
        self.cancel_path_completion();
        // Session arguments use the path-completion scan above.
        let kind = match result.context.target {
            CompletionTarget::Argument(ArgumentKind::Model) => CompletionKind::Model,
            _ => CompletionKind::Slash,
        };
        self.set_completion(kind, result, old_value);
    }

    pub(super) fn set_completion(
        &mut self,
        kind: CompletionKind,
        result: CompletionResult,
        old_value: Option<String>,
    ) {
        if result.candidates.is_empty() {
            // Keep a model list open (empty) so Tab can request it; close
            // everything else.
            if kind == CompletionKind::Model {
                self.completion = Some(Completion {
                    context: result.context,
                    candidates: Vec::new(),
                    selected: 0,
                    kind,
                });
            } else {
                self.completion = None;
            }
            return;
        }
        let selected = old_value
            .as_deref()
            .and_then(|value| {
                result
                    .candidates
                    .iter()
                    .position(|candidate| candidate.value == value)
            })
            .unwrap_or(0)
            .min(result.candidates.len().saturating_sub(1));
        self.completion = Some(Completion {
            context: result.context,
            candidates: result.candidates,
            selected,
            kind,
        });
    }

    pub(super) fn cursor_char_col(&self) -> usize {
        self.input[..self.cursor].chars().count()
    }

    pub(super) fn close_completion(&mut self) {
        self.cancel_path_completion();
        self.completion = None;
        self.session_fetch = Fetch::Idle;
    }

    /// Apply a candidate to the draft, move the cursor, then recompute the
    /// list and fire any on-demand backend fetch (sessions / model lists).
    pub(super) fn accept_candidate(
        &mut self,
        candidate: &Candidate,
        context: &CompletionContext,
        input_tx: &mpsc::UnboundedSender<InputMessage>,
    ) -> Result<()> {
        self.apply_replacement(candidate, context);
        // Also cancels any in-flight path scan so a stale result cannot
        // reopen the popup over the accepted text.
        self.close_completion();
        // A None-argument command (e.g. `/new`) has no arguments to complete;
        // the refresh below would otherwise keep its own name listed.
        let argument_less = matches!(context.target, CompletionTarget::Command,)
            && commands::command_spec(&candidate.value)
                .is_some_and(|spec| spec.argument_kind == ArgumentKind::None);
        if argument_less {
            return Ok(());
        }
        self.request_backend(input_tx, context, candidate, true)?;
        self.refresh_completion();
        Ok(())
    }

    pub(super) fn apply_replacement(&mut self, candidate: &Candidate, context: &CompletionContext) {
        let cursor_col = self.cursor_char_col();
        let Some((replacement, new_cursor_col)) =
            commands::apply_completion(&self.input, cursor_col, context, candidate, &self.skills)
        else {
            return;
        };
        self.input = replacement;
        self.cursor = byte_index_at_char(&self.input, new_cursor_col);
    }

    /// Request backend data for a directly typed model provider prefix or a
    /// session/path command. Candidate acceptance already calls
    /// [`Self::request_backend`]; this covers the equivalent typed form.
    pub(super) fn request_typed_backend(
        &mut self,
        input_tx: &mpsc::UnboundedSender<InputMessage>,
        explicit: bool,
    ) -> Result<()> {
        let Some(context) = self
            .completion
            .as_ref()
            .map(|completion| completion.context.clone())
        else {
            return Ok(());
        };
        let CompletionTarget::Argument(kind) = context.target else {
            return Ok(());
        };
        if !matches!(kind, ArgumentKind::Model | ArgumentKind::Session) {
            return Ok(());
        }
        let start = context.token_start;
        let end = context.token_end.min(self.input.len());
        let value = self.input[start..end].to_owned();
        self.request_backend(
            input_tx,
            &context,
            &Candidate {
                value,
                description: String::new(),
                kind: CandidateKind::Slash,
            },
            explicit,
        )
    }

    /// Send anything the new draft now needs from the agent: a `ListSessions`
    /// for `/load` arguments or a `ListModels` when a `provider:` token was
    /// just completed and its model list is unknown. The dedupe sets keep
    /// repeated Tab presses from spamming the agent. Typing (`explicit` false)
    /// only fetches what was never requested; an explicit Tab also retries a
    /// failed fetch.
    pub(super) fn request_backend(
        &mut self,
        input_tx: &mpsc::UnboundedSender<InputMessage>,
        context: &CompletionContext,
        candidate: &Candidate,
        explicit: bool,
    ) -> Result<()> {
        let is_session = matches!(
            context.target,
            CompletionTarget::Argument(ArgumentKind::Session),
        );
        let session_may_fetch = match self.session_fetch {
            Fetch::Idle => true,
            // A list that arrived empty is only re-asked on an explicit Tab,
            // not on every keystroke.
            Fetch::Loaded | Fetch::Failed => explicit,
            Fetch::InFlight => false,
        };
        if is_session && self.session_candidates.is_empty() && session_may_fetch {
            self.session_fetch = Fetch::InFlight;
            input_tx
                .send(InputMessage::ListSessions)
                .map_err(|_| anyhow::anyhow!("agent input channel closed"))?;
        }
        // A bare `provider:` token requests the catalogue; so does a
        // partially typed `provider:model` token whose provider part is
        // known but uncached — the user is clearly addressing that
        // provider's catalogue even before Tab-accepting the prefix.
        let typed = candidate.value.as_str();
        let provider = typed
            .strip_suffix(':')
            .or_else(|| typed.split_once(':').map(|(provider, _)| provider))
            .filter(|name| {
                self.providers
                    .iter()
                    .any(|known| known.eq_ignore_ascii_case(name))
            })
            .map(str::to_owned);
        if let Some(provider) = provider
            && !self.model_lists.contains_key(&provider)
            && match self
                .model_fetch
                .get(provider.as_str())
                .copied()
                .unwrap_or_default()
            {
                Fetch::Idle => true,
                Fetch::Failed => explicit,
                Fetch::InFlight | Fetch::Loaded => false,
            }
        {
            self.model_fetch.insert(provider.clone(), Fetch::InFlight);
            input_tx
                .send(InputMessage::ListModels {
                    provider: provider.clone(),
                })
                .map_err(|_| anyhow::anyhow!("agent input channel closed"))?;
        }
        Ok(())
    }

    /// The Tab handler for completion.
    pub(super) fn handle_tab(
        &mut self,
        input_tx: &mpsc::UnboundedSender<InputMessage>,
    ) -> Result<()> {
        self.refresh_completion();
        let Some(completion) = self.completion.as_ref().cloned() else {
            return Ok(());
        };
        if completion.candidates.is_empty() {
            // Session/model lists often arrive a beat late; repeat Tab asks
            // the backend again. Use the typed token (not an empty one) so a
            // `provider:` model catalogue can be re-requested too.
            self.request_typed_backend(input_tx, true)?;
            return Ok(());
        }
        if completion.candidates.len() == 1 {
            let candidate = completion.candidates[0].clone();
            self.accept_candidate(&candidate, &completion.context, input_tx)?;
            return Ok(());
        }
        // File and model references rank fuzzily, so a shared prefix is often
        // meaningless (`luna` may rank `openrouter:openai/gpt-5.6-luna`
        // first). Tab accepts the highlighted candidate — the top-ranked
        // match on the first press — rather than becoming a no-op merely
        // because the typed query is not a literal prefix.
        if matches!(
            completion.kind,
            CompletionKind::Path | CompletionKind::Model
        ) {
            let mut selected = completion.selected;
            let token = &self.input[completion.context.token_start
                ..completion.context.token_end.min(self.input.len())];
            if completion.candidates[selected].value == token {
                selected = (selected + 1) % completion.candidates.len();
            }
            let candidate = completion.candidates[selected].clone();
            self.accept_candidate(&candidate, &completion.context, input_tx)?;
            return Ok(());
        }
        // Multiple candidates: extend the shared prefix, or cycle among them
        // once the token already matches the common prefix.
        let prefix = commands::common_prefix(&completion.candidates);
        if prefix.is_empty() {
            return Ok(());
        }
        let token = &self.input
            [completion.context.token_start..completion.context.token_end.min(self.input.len())];
        if token == prefix {
            let selected = (completion.selected + 1) % completion.candidates.len();
            let candidate = completion.candidates[selected].clone();
            self.accept_candidate(&candidate, &completion.context, input_tx)?;
        } else {
            let prefix_candidate = Candidate {
                value: prefix,
                description: String::new(),
                kind: CandidateKind::Slash,
            };
            // Extend to the common prefix and keep the list open.
            self.apply_replacement(&prefix_candidate, &completion.context);
            self.refresh_completion();
        }
        Ok(())
    }

    // Path completion helpers -------------------------------------------------

    pub(super) fn cancel_path_completion(&mut self) {
        if let Some(cancel) = self.path_completion_cancel.take() {
            cancel.cancel();
        }
        self.path_completion_query = None;
        self.path_completion_generation = self.path_completion_generation.wrapping_add(1);
        if self
            .completion
            .as_ref()
            .is_some_and(|completion| completion.kind == CompletionKind::Path)
        {
            self.completion = None;
        }
    }

    pub(super) fn request_path_completion(
        &mut self,
        context: CompletionContext,
        initial: Vec<Candidate>,
        preferred: Option<String>,
        at_prefix: Option<AtPrefix>,
    ) {
        let query = context.query.clone();
        if self.path_completion_query.as_deref() == Some(query.as_str())
            && self.completion.as_ref().is_some_and(|completion| {
                completion.kind == CompletionKind::Path && completion.context == context
            })
        {
            return;
        }
        self.cancel_path_completion();
        let generation = self.path_completion_generation;
        let cancel = CancellationToken::new();
        let scan_cancel = cancel.clone();
        let root = self.environment.cwd.clone();
        let sender = self.path_completion_tx.clone();
        let scan_query = query.clone();
        let selected = preferred
            .as_deref()
            .and_then(|value| initial.iter().position(|c| c.value == value))
            .unwrap_or(0)
            .min(initial.len().saturating_sub(1));
        self.path_completion_cancel = Some(cancel.clone());
        self.path_completion_query = Some(query.clone());
        self.completion = Some(Completion {
            context: context.clone(),
            candidates: initial,
            selected,
            kind: CompletionKind::Path,
        });
        tokio::spawn(async move {
            // Debounce so a burst of typing triggers one scan, not many.
            tokio::time::sleep(Duration::from_millis(200)).await;
            if cancel.is_cancelled() {
                return;
            }
            // `@` references walk the whole workspace; command arguments list
            // one directory.
            let at = at_prefix.is_some();
            let scan = move |root: &Path, query: &str, cancel: &CancellationToken| {
                if at {
                    paths::find_candidates(root, query, cancel)
                } else {
                    paths::find_path_candidates(root, query, cancel)
                }
            };
            let candidates =
                tokio::task::spawn_blocking(move || scan(&root, &scan_query, &scan_cancel))
                    .await
                    .unwrap_or_default();
            if cancel.is_cancelled() {
                return;
            }
            let _ = sender.send(PathCompletionResult {
                generation,
                context,
                candidates,
                at_prefix,
            });
        });
    }

    pub(super) fn apply_path_completion(&mut self, result: PathCompletionResult) {
        if result.generation != self.path_completion_generation {
            return;
        }
        let cursor_col = self.cursor_char_col();
        // The draft must still be on the same token the scan was started for.
        let context_now = match &result.at_prefix {
            Some(prefix) => paths::extract_at_prefix(&self.input, cursor_col)
                .filter(|now| now.token_start == prefix.token_start)
                .map(|now| now.into_context()),
            None => commands::completion_context(&self.input, cursor_col),
        };
        let Some(context_now) = context_now else {
            return;
        };
        if result.context != context_now {
            return;
        }
        self.path_completion_cancel = None;
        let old_value = self
            .completion
            .as_ref()
            .and_then(|completion| completion.candidates.get(completion.selected))
            .map(|candidate| candidate.value.as_str());
        // Merge static candidates (sessions for `/load`, providers/models for
        // `/model`) with the scanned filesystem results.
        let static_candidates = commands::candidates_at_cursor(
            &self.input,
            cursor_col,
            &self.providers,
            &self.model_lists,
            &self.provider,
            &self.session_candidates,
            &self.skills,
        )
        .map(|result| result.candidates)
        .unwrap_or_default();
        let merged = merge_candidates(static_candidates, result.candidates);
        if merged.is_empty() {
            self.completion = None;
            return;
        }
        let selected = old_value
            .and_then(|value| merged.iter().position(|c| c.value == value))
            .unwrap_or(0)
            .min(merged.len().saturating_sub(1));
        self.completion = Some(Completion {
            context: result.context,
            candidates: merged,
            selected,
            kind: CompletionKind::Path,
        });
    }

    /// The dim ghost suffix drawn after the cursor when the input matches a
    /// single untyped completion (or the shared prefix of several).
    pub(super) fn ghost_text(&self) -> String {
        let Some(completion) = self.completion.as_ref() else {
            return String::new();
        };
        // Ghost only previews at the very end of the draft.
        if self.cursor != self.input.len() {
            return String::new();
        }
        let cursor_col = self.cursor_char_col();
        if completion.candidates.is_empty() {
            return String::new();
        }
        // Path completions preview the highlighted candidate — exactly what
        // the next Tab inserts — as the untyped remainder of the token. A
        // fuzzy match (`agent.rs` → `crates/agent/src/agent.rs`) shares no
        // literal prefix, so there is nothing honest to dim inline; the hint
        // row below the input carries those suggestions instead.
        if completion.kind == CompletionKind::Path {
            let candidate = &completion.candidates[completion.selected];
            let token_end = completion.context.token_end.min(self.input.len());
            let typed = &self.input[completion.context.token_start..token_end];
            return candidate
                .value
                .strip_prefix(typed)
                .map(str::to_owned)
                .unwrap_or_default();
        }
        if completion.candidates.len() == 1 {
            return commands::candidate_suffix(
                &self.input,
                cursor_col,
                &completion.context,
                &completion.candidates[0],
            );
        }
        let prefix = commands::common_prefix(&completion.candidates);
        if prefix.is_empty() {
            return String::new();
        }
        // Already at the shared prefix: the list hint stands in for the ghost.
        if self.input
            [completion.context.token_start..completion.context.token_end.min(self.input.len())]
            == prefix
        {
            return String::new();
        }
        let prefix_candidate = Candidate {
            value: prefix,
            description: String::new(),
            kind: CandidateKind::Slash,
        };
        commands::candidate_suffix(
            &self.input,
            cursor_col,
            &completion.context,
            &prefix_candidate,
        )
    }

    /// One dim hint row below the input. It is fitted to the actual content
    /// width so long provider-qualified model IDs can never soft-wrap behind
    /// the region bookkeeping and leave apparent duplicate suggestion rows.
    pub(super) fn completion_hint(&self, width: usize) -> String {
        let Some(completion) = self.completion.as_ref() else {
            return String::new();
        };
        fit_completion_hint(&completion.candidates, completion.selected, width)
    }
}

/// Merge two candidate lists, deduplicating by value while preserving the
/// first list's order. Used to combine static session candidates with the
/// scanned path results for command arguments.
pub(super) fn merge_candidates(
    primary: Vec<Candidate>,
    secondary: Vec<Candidate>,
) -> Vec<Candidate> {
    let mut seen = HashSet::new();
    primary
        .into_iter()
        .chain(secondary)
        .filter(|candidate| seen.insert(candidate.value.clone()))
        .collect()
}
