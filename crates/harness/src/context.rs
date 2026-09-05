use std::path::Path;

/// Context loaded once for a frontend startup. The rendered prompt block and
/// display paths come from the same file snapshot, so the TUI does not repeat
/// discovery merely to show which instructions were injected.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct ContextBundle {
    pub rendered: String,
    pub display_paths: Vec<String>,
}

/// Discover and render project instructions once for a workspace.
pub(crate) fn load_context_bundle(
    workspace_root: &Path,
    disabled: bool,
    include_display_paths: bool,
) -> ContextBundle {
    if disabled {
        return ContextBundle::default();
    }
    let files = tools::context_files::load_context_files(workspace_root);
    ContextBundle {
        rendered: tools::context_files::format_context_files(&files),
        display_paths: if include_display_paths {
            files
                .iter()
                .map(|file| tools::context_files::display_path(&file.path))
                .collect()
        } else {
            Vec::new()
        },
    }
}

/// Render project instructions for a workspace, or an empty string when
/// context-file injection is disabled by the host.
pub fn project_context_for(workspace_root: &Path, disabled: bool) -> String {
    load_context_bundle(workspace_root, disabled, false).rendered
}
