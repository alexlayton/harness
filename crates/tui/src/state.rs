/// A complete tool result retained by the UI for collapsed and expanded
/// rendering.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolRecord {
    pub name: String,
    pub args: String,
    pub summary: String,
    pub ok: bool,
    pub duration_ms: u64,
    pub output: String,
    pub error: Option<String>,
    pub status: ToolStatus,
}

/// Stable identity used to preserve transcript positions while the terminal is
/// resized or the transcript is reflowed.
pub(crate) type EntryId = u64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolStatus {
    Running,
    Success,
    Failure,
}

impl ToolStatus {
    pub fn is_success(self) -> bool {
        matches!(self, Self::Success)
    }

    pub fn is_running(self) -> bool {
        matches!(self, Self::Running)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TranscriptEntry {
    User {
        id: EntryId,
        text: String,
    },
    Assistant {
        id: EntryId,
        markdown: String,
        reasoning: String,
        streaming: bool,
    },
    Tool {
        id: EntryId,
        record: ToolRecord,
        expanded: bool,
    },
    Notice {
        id: EntryId,
        text: String,
    },
    Error {
        id: EntryId,
        text: String,
    },
}

impl TranscriptEntry {
    pub fn id(&self) -> EntryId {
        match self {
            Self::User { id, .. }
            | Self::Assistant { id, .. }
            | Self::Tool { id, .. }
            | Self::Notice { id, .. }
            | Self::Error { id, .. } => *id,
        }
    }

    pub fn is_meaningful(&self) -> bool {
        matches!(
            self,
            Self::User { .. } | Self::Assistant { .. } | Self::Tool { .. }
        )
    }

    pub fn tool_record(&self) -> Option<&ToolRecord> {
        match self {
            Self::Tool { record, .. } => Some(record),
            _ => None,
        }
    }

    pub fn tool_record_mut(&mut self) -> Option<&mut ToolRecord> {
        match self {
            Self::Tool { record, .. } => Some(record),
            _ => None,
        }
    }
}

/// The outcome of a transcript-wide collapse/expand toggle. Collapsed when at
/// least one tool is collapsed, expanded when every tool is already expanded.
/// Pure so the toggle logic can be unit-tested without a TTY.
pub(crate) fn toggle_all_direction(entries: &[TranscriptEntry]) -> bool {
    entries.iter().any(|entry| {
        matches!(
            entry,
            TranscriptEntry::Tool {
                expanded: false,
                ..
            }
        )
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Focus {
    Prompt,
    Tool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toggle_all_direction_expands_when_any_tool_is_collapsed() {
        let tool = |expanded| TranscriptEntry::Tool {
            id: 1,
            record: ToolRecord {
                name: "bash".into(),
                args: "{}".into(),
                summary: "bash: echo hi".into(),
                ok: true,
                duration_ms: 1,
                output: "hi".into(),
                error: None,
                status: ToolStatus::Success,
            },
            expanded,
        };
        // All expanded → the next toggle collapses everything.
        assert!(!toggle_all_direction(&[tool(true), tool(true)]));
        // Any collapsed → the next toggle expands everything.
        assert!(toggle_all_direction(&[tool(true), tool(false)]));
        assert!(toggle_all_direction(&[tool(false), tool(false)]));
        // Non-tool entries do not affect the direction.
        assert!(!toggle_all_direction(&[
            TranscriptEntry::User {
                id: 2,
                text: "hi".into()
            },
            tool(true),
        ]));
    }
}
