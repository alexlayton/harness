use bytes::Bytes;
use futures_core::Stream;
use reqwest::Response;
use std::collections::VecDeque;
use std::pin::Pin;
use std::task::{Context, Poll};

use crate::LlmError;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SseEvent {
    pub event: Option<String>,
    pub data: String,
}

/// Incremental SSE decoder.  It is deliberately small, but implements the
/// parts of the SSE wire format used by all three providers: comments, event
/// names, multiple data lines, CRLF, and dispatch on an empty line.
#[derive(Debug, Default)]
pub struct SseParser {
    line: Vec<u8>,
    event: Option<String>,
    data: String,
    has_data: bool,
}

impl SseParser {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push_bytes(&mut self, bytes: &[u8]) -> Vec<SseEvent> {
        let mut events = Vec::new();
        for byte in bytes {
            if *byte == b'\n' {
                let mut line = std::mem::take(&mut self.line);
                if line.last() == Some(&b'\r') {
                    line.pop();
                }
                self.process_line(&line, &mut events);
            } else {
                self.line.push(*byte);
            }
        }
        events
    }

    pub fn finish(&mut self) -> Vec<SseEvent> {
        let mut events = Vec::new();
        if !self.line.is_empty() {
            let line = std::mem::take(&mut self.line);
            self.process_line(&line, &mut events);
        }
        // The SSE protocol normally ends with a blank line, but accepting a
        // final unterminated event is useful for proxies and fixture tests.
        self.dispatch(&mut events);
        events
    }

    fn process_line(&mut self, line: &[u8], events: &mut Vec<SseEvent>) {
        if line.is_empty() {
            self.dispatch(events);
            return;
        }

        let line = String::from_utf8_lossy(line);
        if line.starts_with(':') {
            return; // comment/keep-alive
        }

        let (field, value) = match line.split_once(':') {
            Some((field, value)) => {
                let value = value.strip_prefix(' ').unwrap_or(value);
                (field, value)
            }
            None => (line.as_ref(), ""),
        };

        match field {
            "event" => self.event = Some(value.to_owned()),
            "data" => {
                if self.has_data {
                    self.data.push('\n');
                }
                self.data.push_str(value);
                self.has_data = true;
            }
            // id and retry are intentionally ignored.  None of the providers
            // needs reconnection based on an SSE id in v1.
            _ => {}
        }
    }

    fn dispatch(&mut self, events: &mut Vec<SseEvent>) {
        if !self.has_data && self.event.is_none() {
            return;
        }
        events.push(SseEvent {
            event: self.event.take(),
            data: std::mem::take(&mut self.data),
        });
        self.has_data = false;
    }
}

pub fn parse_events(input: &str) -> Vec<SseEvent> {
    let mut parser = SseParser::new();
    let mut result = parser.push_bytes(input.as_bytes());
    result.extend(parser.finish());
    result
}

/// Alias with a descriptive name for callers that have a complete fixture.
pub fn parse_sse(input: &str) -> Vec<SseEvent> {
    parse_events(input)
}

/// A stream adapter over reqwest's response body.  Keeping the decoder here
/// means dialect parsers can be pure functions over one complete SSE payload.
pub struct SseStream {
    inner: Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>,
    parser: SseParser,
    pending: VecDeque<SseEvent>,
    finished: bool,
}

impl SseStream {
    pub fn new(response: Response) -> Self {
        Self {
            inner: Box::pin(response.bytes_stream()),
            parser: SseParser::new(),
            pending: VecDeque::new(),
            finished: false,
        }
    }
}

impl Stream for SseStream {
    type Item = Result<SseEvent, LlmError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if let Some(event) = self.pending.pop_front() {
            return Poll::Ready(Some(Ok(event)));
        }
        if self.finished {
            return Poll::Ready(None);
        }

        loop {
            match self.inner.as_mut().poll_next(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Some(Ok(bytes))) => {
                    let parsed = self.parser.push_bytes(bytes.as_ref());
                    self.pending.extend(parsed);
                    if let Some(event) = self.pending.pop_front() {
                        return Poll::Ready(Some(Ok(event)));
                    }
                }
                Poll::Ready(Some(Err(error))) => {
                    self.finished = true;
                    return Poll::Ready(Some(Err(LlmError::Network(error))));
                }
                Poll::Ready(None) => {
                    self.finished = true;
                    let parsed = self.parser.finish();
                    self.pending.extend(parsed);
                    if let Some(event) = self.pending.pop_front() {
                        return Poll::Ready(Some(Ok(event)));
                    }
                    return Poll::Ready(None);
                }
            }
        }
    }
}

pub fn stream_response(response: Response) -> SseStream {
    SseStream::new(response)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_comments_crlf_and_multiline_data() {
        let input = ": keep alive\r\n\nevent: message\r\ndata: {\"a\":\ndata: 1}\r\n\r\n";
        assert_eq!(
            parse_events(input),
            vec![SseEvent {
                event: Some("message".into()),
                data: "{\"a\":\n1}".into()
            }]
        );
    }

    #[test]
    fn parses_done_and_chunk_boundaries() {
        let mut parser = SseParser::new();
        assert!(parser.push_bytes(b"data: [DO").is_empty());
        assert_eq!(
            parser.push_bytes(b"NE]\n\n"),
            vec![SseEvent {
                event: None,
                data: "[DONE]".into()
            }]
        );
    }
}
