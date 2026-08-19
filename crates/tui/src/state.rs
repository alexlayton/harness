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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Focus {
    Prompt,
    Tool,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a ToolRecord so tests can construct transcript fixtures concisely.
    fn tool(status: ToolStatus) -> TranscriptEntry {
        TranscriptEntry::Tool {
            id: 1,
            record: ToolRecord {
                name: "bash".into(),
                args: "{}".into(),
                summary: "bash".into(),
                ok: status.is_success(),
                duration_ms: 1,
                output: String::new(),
                error: None,
                status,
            },
            expanded: false,
        }
    }

    #[test]
    fn meaningful_entries_are_user_assistant_and_tool() {
        let user = TranscriptEntry::User {
            id: 2,
            text: "hi".into(),
        };
        let assistant = TranscriptEntry::Assistant {
            id: 3,
            markdown: "m".into(),
            reasoning: String::new(),
            streaming: false,
        };
        let notice = TranscriptEntry::Notice {
            id: 4,
            text: "n".into(),
        };
        let error = TranscriptEntry::Error {
            id: 5,
            text: "e".into(),
        };
        assert!(user.is_meaningful());
        assert!(assistant.is_meaningful());
        assert!(tool(ToolStatus::Running).is_meaningful());
        assert!(!notice.is_meaningful());
        assert!(!error.is_meaningful());
    }
}
