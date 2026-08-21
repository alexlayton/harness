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
