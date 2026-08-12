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
    Transcript,
    Tool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ScrollState {
    pub offset: usize,
    pub content_height: usize,
    pub viewport_height: usize,
    pub follow_latest: bool,
    pub new_content_below: bool,
}

impl Default for ScrollState {
    fn default() -> Self {
        Self {
            offset: 0,
            content_height: 0,
            viewport_height: 1,
            follow_latest: true,
            new_content_below: false,
        }
    }
}

impl ScrollState {
    pub fn max_offset(&self) -> usize {
        self.content_height.saturating_sub(self.viewport_height)
    }

    pub fn at_bottom(&self) -> bool {
        self.offset >= self.max_offset()
    }

    pub fn clamp(&mut self) {
        self.offset = self.offset.min(self.max_offset());
        if self.at_bottom() {
            self.follow_latest = true;
            self.new_content_below = false;
        }
    }

    pub fn scroll_by(&mut self, delta: isize) {
        let max = self.max_offset();
        self.offset = if delta.is_negative() {
            self.offset.saturating_sub(delta.unsigned_abs())
        } else {
            self.offset.saturating_add(delta as usize).min(max)
        };
        self.follow_latest = self.at_bottom();
        if self.follow_latest {
            self.new_content_below = false;
        }
    }

    pub fn page_size(&self) -> usize {
        self.viewport_height.saturating_sub(2).max(1)
    }

    pub fn go_bottom(&mut self) {
        self.offset = self.max_offset();
        self.follow_latest = true;
        self.new_content_below = false;
    }

    pub fn on_content_changed(&mut self, was_at_bottom: bool) {
        if was_at_bottom || self.follow_latest {
            self.offset = self.max_offset();
            self.follow_latest = true;
            self.new_content_below = false;
        } else {
            self.offset = self.offset.min(self.max_offset());
            self.new_content_below = self.offset < self.max_offset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scroll_state_follows_new_content_only_at_bottom() {
        let mut state = ScrollState {
            offset: 5,
            content_height: 10,
            viewport_height: 5,
            follow_latest: true,
            new_content_below: false,
        };
        state.on_content_changed(true);
        assert_eq!(state.offset, 5);
        assert!(state.follow_latest);

        state.offset = 1;
        state.follow_latest = false;
        state.on_content_changed(false);
        assert_eq!(state.offset, 1);
        assert!(state.new_content_below);
    }

    #[test]
    fn scroll_state_resumes_follow_at_bottom() {
        let mut state = ScrollState {
            offset: 0,
            content_height: 20,
            viewport_height: 5,
            follow_latest: false,
            new_content_below: true,
        };
        state.scroll_by(100);
        assert_eq!(state.offset, 15);
        assert!(state.follow_latest);
        assert!(!state.new_content_below);
    }
}
