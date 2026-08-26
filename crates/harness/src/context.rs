use std::path::Path;

/// Render project instructions for a workspace, or an empty string when
/// context-file injection is disabled by the host.
pub fn project_context_for(workspace_root: &Path, disabled: bool) -> String {
    if disabled {
        return String::new();
    }
    let files = tools::context_files::load_context_files(workspace_root);
    tools::context_files::format_context_files(&files)
}
