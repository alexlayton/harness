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
///
/// Records with no `data` lines are never dispatched, even when they carry
/// an `event` name: an event-only frame (`event: ping`) is a keep-alive,
/// not a payload, and leaking its name into the next data frame would
/// corrupt dialect parsing.  Incremental parsing returns `Result` so
/// over-limit input fails fast as `LlmError::Stream` instead of growing
/// without bound across chunk boundaries.
#[derive(Debug, Default)]
pub struct SseParser {
    line: Vec<u8>,
    event: Option<String>,
    data: String,
    has_data: bool,
}

/// Maximum bytes for one SSE line; generous (providers send small JSON
/// lines) but bounded so a malicious endpoint cannot grow one line forever.
pub const MAX_LINE_BYTES: usize = 1024 * 1024;
/// Maximum accumulated bytes for one event's `data` buffer across chunks.
pub const MAX_EVENT_BYTES: usize = 8 * 1024 * 1024;

impl SseParser {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push_bytes(&mut self, bytes: &[u8]) -> Result<Vec<SseEvent>, LlmError> {
        let mut events = Vec::new();
        for byte in bytes {
            if *byte == b'\n' {
                let mut line = std::mem::take(&mut self.line);
                if line.last() == Some(&b'\r') {
                    line.pop();
                }
                self.process_line(&line, &mut events)?;
            } else {
                self.line.push(*byte);
                if self.line.len() > MAX_LINE_BYTES {
                    return Err(LlmError::Stream("SSE line exceeds size limit".into()));
                }
            }
        }
        Ok(events)
    }

    pub fn finish(&mut self) -> Result<Vec<SseEvent>, LlmError> {
        let mut events = Vec::new();
        if !self.line.is_empty() {
            let line = std::mem::take(&mut self.line);
            self.process_line(&line, &mut events)?;
        }
        // The SSE protocol normally ends with a blank line, but accepting a
        // final unterminated event is useful for proxies and fixture tests.
        self.dispatch(&mut events);
        Ok(events)
    }

    fn process_line(&mut self, line: &[u8], events: &mut Vec<SseEvent>) -> Result<(), LlmError> {
        if line.is_empty() {
            self.dispatch(events);
            return Ok(());
        }

        let line = String::from_utf8_lossy(line);
        if line.starts_with(':') {
            return Ok(()); // comment/keep-alive
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
                if self.data.len() > MAX_EVENT_BYTES {
                    return Err(LlmError::Stream("SSE event exceeds size limit".into()));
                }
            }
            // id and retry are intentionally ignored.  None of the providers
            // needs reconnection based on an SSE id in v1.
            _ => {}
        }
        Ok(())
    }

    fn dispatch(&mut self, events: &mut Vec<SseEvent>) {
        // Event-only frames carry no data and dispatch nothing; clear the
        // stored event name so it cannot leak into the next data frame.
        if !self.has_data {
            self.event = None;
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
    // Test/fixture helper: inputs are small and valid, so size violations
    // (which return `Err` incrementally) are dropped here.  Streaming and
    // dialect paths propagate the `Result` from `push_bytes`/`finish`.
    let mut parser = SseParser::new();
    let mut result = parser.push_bytes(input.as_bytes()).unwrap_or_default();
    result.extend(parser.finish().unwrap_or_default());
    result
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
                    let parsed = match self.parser.push_bytes(bytes.as_ref()) {
                        Ok(parsed) => parsed,
                        Err(error) => {
                            self.finished = true;
                            return Poll::Ready(Some(Err(error)));
                        }
                    };
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
                    let parsed = match self.parser.finish() {
                        Ok(parsed) => parsed,
                        Err(error) => return Poll::Ready(Some(Err(error))),
                    };
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
        assert!(parser.push_bytes(b"data: [DO").unwrap().is_empty());
        assert_eq!(
            parser.push_bytes(b"NE]\n\n").unwrap(),
            vec![SseEvent {
                event: None,
                data: "[DONE]".into()
            }]
        );
    }

    #[test]
    fn event_only_frames_emit_nothing_and_do_not_leak() {
        // `event: ping` alone is a keep-alive, not a payload.
        let mut parser = SseParser::new();
        assert!(parser.push_bytes(b"event: ping\n\n").unwrap().is_empty());
        // The next data frame must not inherit the stale event name.
        assert_eq!(
            parser.push_bytes(b"data: hello\n\n").unwrap(),
            vec![SseEvent {
                event: None,
                data: "hello".into()
            }]
        );
        // Same across `finish`: a trailing event-only fragment emits nothing.
        let mut parser = SseParser::new();
        assert!(parser.push_bytes(b"event: ping").unwrap().is_empty());
        assert!(parser.finish().unwrap().is_empty());
    }

    #[test]
    fn exactly_at_limit_succeeds_over_limit_fails_across_chunks() {
        // Exactly-at-limit line succeeds.
        let mut parser = SseParser::new();
        let mut line = vec![b'x'; MAX_LINE_BYTES - "data: ".len()];
        line.extend_from_slice(b"\n\n");
        let mut input = b"data: ".to_vec();
        input.extend_from_slice(&line);
        assert_eq!(parser.push_bytes(&input).unwrap().len(), 1);
        // One byte over the line limit fails, even split across chunks.
        let mut parser = SseParser::new();
        let over = vec![b'y'; MAX_LINE_BYTES + 1];
        let half = over.len() / 2;
        assert!(parser.push_bytes(&over[..half]).unwrap().is_empty());
        assert!(parser.push_bytes(&over[half..]).is_err());
        // Over-limit accumulated event data fails before unbounded growth.
        let mut parser = SseParser::new();
        let chunk = format!("data: {}\n", "z".repeat(1024));
        let mut result = Ok(Vec::new());
        for _ in 0..(MAX_EVENT_BYTES / 1024 + 2) {
            result = parser.push_bytes(chunk.as_bytes());
            if result.is_err() {
                break;
            }
        }
        assert!(result.is_err());
    }

    #[test]
    fn crlf_and_multiline_data_still_work() {
        let mut parser = SseParser::new();
        let events = parser.push_bytes(b"data: a\r\ndata: b\r\n\r\n").unwrap();
        assert_eq!(
            events,
            vec![SseEvent {
                event: None,
                data: "a\nb".into()
            }]
        );
    }
}
