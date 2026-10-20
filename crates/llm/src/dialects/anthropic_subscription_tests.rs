use super::*;
use crate::sse::SseEvent;
use crate::{Message, ToolCall, ToolDefinition};

fn request() -> CompletionRequest {
    CompletionRequest {
        model: "claude-sonnet-4-6".into(),
        system: Some("workspace instructions".into()),
        messages: vec![Message::user("hello world")],
        tools: vec![ToolDefinition {
            name: "_read".into(),
            description: "Read it".into(),
            parameters: json!({"type":"object","properties":{"path":{"type":"string","enum":["é","漢字"]}},"required":["path"],"additionalProperties":false,"$defs":{"nested":{"type":"integer"}}}),
        }],
        max_tokens: None,
        temperature: None,
        reasoning: ReasoningPolicy::Auto,
        session_id: None,
    }
}
fn feed(p: &mut SubscriptionParser, v: Value) -> Result<Vec<StreamEvent>, LlmError> {
    p.parse_event(&SseEvent {
        event: None,
        data: v.to_string(),
    })
}
fn start(p: &mut SubscriptionParser) {
    feed(p, json!({"type":"message_start","message":{"usage":{"input_tokens":3,"cache_read_input_tokens":4}}})).unwrap();
}
fn block(p: &mut SubscriptionParser, index: u64, b: Value) {
    feed(
        p,
        json!({"type":"content_block_start","index":index,"content_block":b}),
    )
    .unwrap();
    feed(p, json!({"type":"content_block_stop","index":index})).unwrap();
}

#[test]
fn billing_matches_upstream_utf16_vectors() {
    // Generated with /root/code/pi-sub-anthropic/fingerprint.ts createClaudeBillingHeader.
    for (text, suffix) in [
        ("", "d7b"),
        ("hello world", "e80"),
        ("abcd😀x🚀abcdefghijklmnop", "98f"),
        ("é漢字abc🙂defghijklmnopqrstuvwxyz", "10a"),
    ] {
        assert_eq!(
            billing(text),
            format!(
                "x-anthropic-billing-header: cc_version=2.1.280.{suffix}; cc_entrypoint=claude-desktop; cch=00000;"
            )
        );
    }
}

#[test]
fn attestation_covers_exact_unicode_bytes_not_reserialized_json() {
    let mut req = request();
    req.messages = vec![Message::user(
        "abcd😀x🚀abcdefghijklmnop cch=00000 \"quote\"\n漢字",
    )];
    let bytes = serialize(&req).unwrap();
    let body: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        body["messages"][2]["content"][0]["text"],
        "abcd😀x🚀abcdefghijklmnop cch=00000 \"quote\"\n漢字"
    );
    let offset = bytes.windows(4).position(|w| w == b"cch=").unwrap() + 4;
    let actual = std::str::from_utf8(&bytes[offset..offset + 5]).unwrap();
    let mut placeholder = bytes.clone();
    placeholder[offset..offset + 5].copy_from_slice(b"00000");
    assert_eq!(
        actual,
        format!(
            "{:05x}",
            xxhash_rust::xxh64::xxh64(&placeholder, SEED) & 0xfffff
        )
    );
    assert!(
        String::from_utf8(bytes)
            .unwrap()
            .starts_with("{\"system\":[{\"type\":\"text\",\"text\":\"x-anthropic-billing-header:")
    );
}

#[test]
fn schemas_names_ids_and_cache_boundaries_are_preserved() {
    let mut req = request();
    req.messages
        .push(Message::assistant(vec![Content::ToolCall(ToolCall {
            id: "a/b".into(),
            name: "_read".into(),
            arguments: json!({"path":"é"}),
        })]));
    req.messages.push(Message::tool_result("a/b", "ok", false));
    let body: Value = serde_json::from_slice(&serialize(&req).unwrap()).unwrap();
    assert_eq!(body["tools"][0]["name"], "__read");
    assert_eq!(body["tools"][0]["input_schema"], req.tools[0].parameters);
    assert_eq!(body["messages"][3]["content"][0]["name"], "__read");
    assert_eq!(
        body["messages"][3]["content"][0]["id"],
        body["messages"][4]["content"][0]["tool_use_id"]
    );
    assert_ne!(normalize_id("a/b"), normalize_id("a?b"));
    assert!(
        body.get("system").unwrap().as_array().unwrap()[1]
            .get("cache_control")
            .is_some()
    );
    assert!(body["tools"][0].get("cache_control").is_some());
    assert!(
        body["messages"][4]["content"][0]
            .get("cache_control")
            .is_some()
    );
    assert!(
        body["metadata"]["user_id"]
            .as_str()
            .unwrap()
            .starts_with("user_")
    );
}

#[test]
fn signed_and_redacted_thinking_roundtrip_without_display_duplication() {
    let mut req = request();
    let mut p = SubscriptionParser::new(&req);
    start(&mut p);
    feed(&mut p, json!({"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}})).unwrap();
    feed(&mut p, json!({"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"private"}})).unwrap();
    feed(&mut p, json!({"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig+/="}})).unwrap();
    feed(&mut p, json!({"type":"content_block_stop","index":0})).unwrap();
    block(
        &mut p,
        1,
        json!({"type":"redacted_thinking","data":"opaque+/="}),
    );
    block(&mut p, 2, json!({"type":"text","text":"answer"}));
    feed(&mut p, json!({"type":"content_block_start","index":3,"content_block":{"type":"tool_use","id":"call-1","name":"__read","input":{}}})).unwrap();
    feed(&mut p, json!({"type":"content_block_delta","index":3,"delta":{"type":"input_json_delta","partial_json":"{\"path\":\"é\"}"}})).unwrap();
    assert!(
        feed(&mut p, json!({"type":"content_block_stop","index":3}))
            .unwrap()
            .is_empty()
    );
    feed(&mut p, json!({"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":9}})).unwrap();
    let events = feed(&mut p, json!({"type":"message_stop"})).unwrap();
    let data = events
        .iter()
        .find_map(|e| {
            if let StreamEvent::OpaqueState { data, .. } = e {
                Some(data.clone())
            } else {
                None
            }
        })
        .unwrap();
    assert!(events.iter().any(|e| matches!(e, StreamEvent::ToolCallComplete(c) if c.name == "_read" && c.arguments == json!({"path":"é"}))));
    assert!(events.iter().any(|e| matches!(e, StreamEvent::Done { usage: Some(u), .. } if u.input_tokens == 7 && u.output_tokens == 9)));
    req.messages.push(Message::assistant(vec![
        Content::Reasoning("private".into()),
        Content::Text("answer".into()),
        Content::Opaque {
            provider: TAG.into(),
            data,
        },
    ]));
    req.messages
        .push(Message::tool_result("call-1", "ok", false));
    let body: Value = serde_json::from_slice(&serialize(&req).unwrap()).unwrap();
    let blocks = body["messages"][3]["content"].as_array().unwrap();
    assert_eq!(blocks.len(), 4);
    assert_eq!(
        blocks[0],
        json!({"type":"thinking","thinking":"private","signature":"sig+/="})
    );
    assert_eq!(
        blocks[1],
        json!({"type":"redacted_thinking","data":"opaque+/="})
    );
    assert_eq!(blocks[3]["name"], "__read");
    assert_eq!(
        blocks[3]["id"],
        body["messages"][4]["content"][0]["tool_use_id"]
    );
    assert!(p.finish().unwrap().is_empty());
}

#[test]
fn malformed_and_truncated_streams_never_release_tools() {
    let req = request();
    for payload in [
        json!({}),
        json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"x"}}),
        json!({"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"c","name":"unregistered","input":{}}}),
        json!({"type":"content_block_stop","index":0}),
    ] {
        let mut p = SubscriptionParser::new(&req);
        start(&mut p);
        assert!(feed(&mut p, payload).is_err());
    }
    for args in ["{bad", "[]", "null"] {
        let mut p = SubscriptionParser::new(&req);
        start(&mut p);
        feed(&mut p, json!({"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"c","name":"__read","input":{}}})).unwrap();
        feed(&mut p, json!({"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":args}})).unwrap();
        let stop = feed(&mut p, json!({"type":"content_block_stop","index":0}));
        if let Ok(events) = stop {
            assert!(events.is_empty());
            assert!(feed(&mut p, json!({"type":"message_stop"})).is_err());
        }
    }
    let mut p = SubscriptionParser::new(&req);
    start(&mut p);
    block(
        &mut p,
        0,
        json!({"type":"tool_use","id":"c","name":"__read","input":{}}),
    );
    assert!(p.finish().is_err());
    let mut p = SubscriptionParser::new(&req);
    start(&mut p);
    feed(
        &mut p,
        json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}),
    )
    .unwrap();
    assert!(feed(&mut p, json!({"type":"message_stop"})).is_err());
}

#[test]
fn invalid_block_diagnostics_identify_fields_without_exposing_payloads() {
    for (block, expected) in [
        (
            json!({"type":"tool_use","id":"c","name":"private-name","input":{}}),
            "tool block name is not registered",
        ),
        (
            json!({"type":"tool_use","id":"c","name":"__read","input":"private-input"}),
            "tool block input must be an object",
        ),
        (
            json!({"type":"tool_use","id":"","name":"__read","input":{}}),
            "tool block requires a nonempty id",
        ),
        (
            json!({"type":"text","text":null}),
            "text block requires string text",
        ),
        (
            json!({"type":"thinking","thinking":null}),
            "thinking block requires string thinking",
        ),
        (
            json!({"type":"redacted_thinking","data":""}),
            "redacted thinking block requires nonempty data",
        ),
        (
            json!({"type":"private-unknown-type","data":"private-data"}),
            "unsupported or missing block type",
        ),
    ] {
        let mut parser = SubscriptionParser::new(&request());
        start(&mut parser);
        let error = feed(
            &mut parser,
            json!({"type":"content_block_start","index":3,"content_block":block}),
        )
        .unwrap_err();
        assert_eq!(
            error.to_string(),
            format!("parse: Anthropic subscription: invalid content block at index 3: {expected}")
        );
        assert!(!error.to_string().contains("private-"));
    }
}

#[test]
fn thinking_validation_and_unsigned_state() {
    let mut req = request();
    req.reasoning = ReasoningPolicy::Off;
    assert!(serialize(&req).is_err());
    req.reasoning = ReasoningPolicy::Auto;
    req.temperature = Some(0.5);
    assert!(serialize(&req).is_err());
    req.temperature = None;
    req.max_tokens = Some(0);
    assert!(serialize(&req).is_err());
    let mut p = SubscriptionParser::new(&request());
    start(&mut p);
    block(&mut p, 0, json!({"type":"thinking","thinking":"unsigned"}));
    assert!(
        !feed(&mut p, json!({"type":"message_stop"}))
            .unwrap()
            .iter()
            .any(|e| matches!(e, StreamEvent::OpaqueState { .. }))
    );
}
