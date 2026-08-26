//! Human-facing standalone credential login.
use crate::config::{LoginArgs, LoginProvider, ProviderArg, select_provider_after_login};
use anyhow::{Result, bail};
use auth::{AuthEvent, CopilotAuth, OpenAiCodexAuth};
use std::process::ExitCode;
use tokio_util::sync::CancellationToken;

pub async fn run(args: &LoginArgs) -> Result<ExitCode> {
    if args.device_code && args.provider != LoginProvider::OpenAiCodex {
        bail!("--device-code is only valid for openai-codex");
    }
    let cancel = CancellationToken::new();
    let signal = cancel.clone();
    tokio::spawn(async move {
        let _ = tokio::signal::ctrl_c().await;
        signal.cancel();
    });
    let provider = match args.provider {
        LoginProvider::GithubCopilot => {
            let auth = CopilotAuth::from_default()?;
            auth.login_with_events(None, &cancel, render_event).await?;
            eprintln!("GitHub Copilot login complete.");
            ProviderArg::GithubCopilot
        }
        LoginProvider::OpenAiCodex => {
            let auth = OpenAiCodexAuth::from_default()?;
            if args.device_code {
                auth.login_device(&cancel, render_event).await?;
            } else {
                auth.login_browser(&cancel, |event| {
                    if let AuthEvent::Prompt { message } = &event {
                        eprintln!("Open this URL to sign in:\n{message}");
                        open_browser(message);
                    } else {
                        render_event(event);
                    }
                })
                .await?;
            }
            eprintln!("OpenAI Codex login complete.");
            ProviderArg::OpenAiCodex
        }
    };
    if select_provider_after_login(provider)? {
        eprintln!("Selected {provider} as the default provider.");
    }
    Ok(ExitCode::SUCCESS)
}
fn render_event(event: AuthEvent) {
    match event {
        AuthEvent::DeviceCode {
            verification_url,
            user_code,
            expires_in,
            ..
        } => {
            eprintln!("Open {verification_url}\nEnter code: {user_code}\nExpires in {expires_in}s");
            open_browser(&verification_url);
        }
        AuthEvent::Prompt { message } | AuthEvent::Progress { message } => eprintln!("{message}"),
        AuthEvent::Failed { message } => eprintln!("login failed: {message}"),
        AuthEvent::Started | AuthEvent::Finished => {}
    }
}
/// Opening is best-effort; the printed URL remains usable in terminals without
/// a desktop browser.
fn open_browser(url: &str) {
    #[cfg(target_os = "macos")]
    let command = "open";
    #[cfg(not(target_os = "macos"))]
    let command = "xdg-open";
    let _ = std::process::Command::new(command).arg(url).spawn();
}
