/// The stable prompt template. `{cwd}` is replaced at request time.
pub const SYSTEM_PROMPT: &str = "You are harness, an expert coding assistant running in the user's terminal.\nWorking directory: {cwd} (all relative paths resolve here).\n\nYou have three tools: read (inspect files), write (create/overwrite files), bash (run shell commands). Use them to explore the codebase and make changes directly rather than showing code for the user to apply. Prefer reading a file before modifying it. Keep responses concise; use markdown for formatting.";

/// Build the small, deliberately stable system prompt used on every request.
pub fn system_prompt(cwd: &str) -> String {
    SYSTEM_PROMPT.replace("{cwd}", cwd)
}

pub fn current_system_prompt() -> String {
    let cwd = std::env::current_dir()
        .map(|path| path.display().to_string())
        .unwrap_or_else(|_| ".".into());
    system_prompt(&cwd)
}

pub const SYSTEM_PROMPT_TEMPLATE: &str = SYSTEM_PROMPT;
