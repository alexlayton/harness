use super::*;
use crate::{Message, ReasoningPolicy, StreamEvent};
use auth::{AnthropicSubscriptionCredential, AuthStore};
use futures_util::StreamExt;
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

fn request() -> CompletionRequest {
    CompletionRequest {
        model: MODELS[0].into(),
        system: None,
        messages: vec![Message::user("hello 😀")],
        tools: vec![],
        max_tokens: None,
        temperature: None,
        reasoning: ReasoningPolicy::Auto,
        session_id: None,
    }
}

async fn fixture(
    status: u16,
    body: String,
) -> (String, tokio::task::JoinHandle<(String, Vec<u8>)>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut bytes = Vec::new();
        let (headers, data) = loop {
            let mut chunk = [0; 4096];
            let n = socket.read(&mut chunk).await.unwrap();
            assert!(n > 0);
            bytes.extend_from_slice(&chunk[..n]);
            if let Some(end) = bytes.windows(4).position(|w| w == b"\r\n\r\n") {
                let headers = String::from_utf8(bytes[..end].to_vec()).unwrap();
                let len: usize = headers
                    .lines()
                    .find_map(|l| {
                        l.to_ascii_lowercase()
                            .strip_prefix("content-length: ")
                            .map(str::to_owned)
                    })
                    .unwrap()
                    .parse()
                    .unwrap();
                if bytes.len() >= end + 4 + len {
                    break (headers, bytes[end + 4..end + 4 + len].to_vec());
                }
            }
        };
        socket.write_all(format!("HTTP/1.1 {status} Fixture\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).as_bytes()).await.unwrap();
        (headers, data)
    });
    (endpoint, task)
}

fn provider(store: AuthStore, endpoint: String) -> AnthropicSubscriptionProvider {
    store
        .save_anthropic_subscription(&AnthropicSubscriptionCredential::new(
            "fixture-secret",
            "fixture-refresh",
            u64::MAX,
            "account",
        ))
        .unwrap();
    AnthropicSubscriptionProvider::new(Arc::new(AnthropicSubscriptionAuth::new(store).unwrap()))
        .with_endpoint(endpoint)
}

#[tokio::test]
async fn catalog_reports_model_specific_context_limits_without_network() {
    let dir = tempfile::tempdir().unwrap();
    let p = provider(
        AuthStore::new(dir.path().join("auth.json")),
        "http://unused.invalid".into(),
    );
    let models = p.list_models().await.unwrap();
    let expected = [
        ("claude-opus-5-5", Some(1_000_000)),
        ("claude-sonnet-5-5", None),
        ("claude-haiku-5-5", None),
        ("claude-fable-5-1", Some(1_000_000)),
        ("claude-opus-5", Some(1_000_000)),
        ("claude-sonnet-5", Some(1_000_000)),
        ("claude-fable-5", Some(1_000_000)),
        ("claude-opus-4-8", Some(1_000_000)),
        ("claude-opus-4-7", Some(1_000_000)),
        ("claude-sonnet-4-6", Some(1_000_000)),
        ("claude-opus-4-6", Some(1_000_000)),
        ("claude-opus-4-5", Some(200_000)),
        ("claude-sonnet-4-5", Some(200_000)),
        ("claude-haiku-4-5", Some(200_000)),
    ];
    assert_eq!(models.len(), MODELS.len());
    for model in models {
        let (_, limit) = expected.iter().find(|(id, _)| *id == model.id).unwrap();
        assert_eq!(model.context_length, *limit, "{}", model.id);
    }
}

#[tokio::test]
async fn mock_posts_exact_agent_request_without_tools_and_parses_sse() {
    let body = [json!({"type":"message_start","message":{"usage":{"input_tokens":2}}}), json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}), json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}), json!({"type":"content_block_stop","index":0}), json!({"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}), json!({"type":"message_stop"})].iter().map(|v| format!("data: {v}\n\n")).collect::<String>();
    let (endpoint, task) = fixture(200, body).await;
    let dir = tempfile::tempdir().unwrap();
    let p = provider(AuthStore::new(dir.path().join("auth.json")), endpoint);
    let models = p.list_models().await.unwrap();
    assert_eq!(models.len(), MODELS.len());
    assert_eq!(models[0].context_length, Some(1_000_000));
    let mut stream = p.stream(&request()).await.unwrap();
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event.unwrap());
    }
    assert!(
        events
            .iter()
            .any(|e| matches!(e, StreamEvent::TextDelta(t) if t == "hi"))
    );
    assert!(events.iter().any(|e| matches!(e, StreamEvent::Done { .. })));
    let (headers, bytes) = task.await.unwrap();
    let h = headers.to_ascii_lowercase();
    assert!(h.starts_with("post /v1/messages http/1.1"));
    assert!(h.contains("authorization: bearer fixture-secret"));
    assert!(!h.contains("x-api-key"));
    assert!(h.contains("claude-code-20250219"));
    assert!(h.contains("fallback-credit-2026-06-01"));
    assert!(h.contains("effort-2025-11-24"));
    assert!(!h.contains("context-1m"));
    let body: Value = serde_json::from_slice(&bytes).unwrap();
    assert!(body.get("tools").is_none());
    assert_eq!(body["thinking"]["type"], "adaptive");
    assert!(!String::from_utf8(bytes).unwrap().contains("fixture-secret"));
}

#[tokio::test]
async fn mock_errors_redact_secrets_and_truncated_stream_is_not_success() {
    for (status, body) in [
        (401, "fixture-secret rejected"),
        (
            200,
            "data: {\"type\":\"error\",\"error\":{\"message\":\"fixture-secret failure\"}}\n\n",
        ),
        (200, "data: {\"type\":\"message_start\",\"message\":{}}\n\n"),
    ] {
        let (endpoint, task) = fixture(status, body.into()).await;
        let dir = tempfile::tempdir().unwrap();
        let p = provider(AuthStore::new(dir.path().join("auth.json")), endpoint);
        let error = match p.stream(&request()).await {
            Err(e) => e,
            Ok(mut s) => {
                let mut error = None;
                while let Some(event) = s.next().await {
                    if let Err(e) = event {
                        error = Some(e);
                        break;
                    }
                }
                error.expect("must not turn a truncated/error stream into success")
            }
        };
        assert!(!format!("{error:?} {error}").contains("fixture-secret"));
        task.await.unwrap();
    }
}
