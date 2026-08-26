use super::InputMessage;
use super::{Agent, AgentEvent, TurnError, send};
use futures_util::FutureExt;
use futures_util::stream::{FuturesUnordered, StreamExt};
use llm::{Content, Message, Role, ToolCall};
use std::future::Future;
use std::pin::Pin;
use std::time::Instant;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tools::{Concurrency, ToolOutput, ToolRegistry, call_summary};

/// Upper bound on read-only tool calls running at once.  Read tools are
/// cheap, but the shared `FileSearchIndex` already gates its own concurrency
/// and a runaway batch would thrash the disk; 8 keeps latency wins while
/// staying polite.
const MAX_CONCURRENT_READ_ONLY_TOOLS: usize = 8;

/// Upper bound on concurrent `Parallel` tool calls (subagents) and the
/// per-subagent turn budget.  Each parallel slot is an entire nested agent
/// loop, so this is deliberately separate from (and smaller than) the
/// read-only cap.
#[derive(Clone, Copy, Debug)]
pub struct SubagentLimits {
    pub max_concurrent: usize,
}

impl Default for SubagentLimits {
    fn default() -> Self {
        Self { max_concurrent: 4 }
    }
}

/// One scheduling unit of a turn's tool calls.  Read-only calls group into a
/// batch that runs concurrently; adjacent `Parallel` calls of the same
/// fan-out tool form their own concurrent batch; every exclusive call forms
/// a singleton batch, preserving program order around it.  A batch of one is
/// executed exactly like the historical serial path.
pub(crate) struct ToolBatch {
    pub(crate) calls: Vec<ToolCall>,
    /// Which concurrency class this batch runs under; only [`Concurrency::ReadOnly`]
    /// and [`Concurrency::Parallel`] batches ever hold more than one call.
    pub(crate) class: Concurrency,
}

impl ToolBatch {
    /// Whether this batch may launch more than one call at once.
    pub(crate) fn concurrent(&self) -> bool {
        matches!(self.class, Concurrency::ReadOnly | Concurrency::Parallel)
    }
}

/// An in-flight tool execution carrying its slot index so results can be
/// recorded in original call order rather than completion order.  Borrows
/// the registry only for the batch's lifetime.
type ToolRun<'a> = Pin<Box<dyn Future<Output = (usize, ToolOutput, Instant)> + Send + 'a>>;

/// Partition a turn's calls into batches without reordering anything: a
/// maximal run of read-only calls becomes one batch, a maximal run of
/// `Parallel` calls *of one tool* becomes its own batch, and every exclusive
/// call is a singleton.  A read is never hoisted above a write because the
/// model may intend the read to observe that write's effect; parallel
/// fan-out tools are likewise never merged across an intervening call.
pub(crate) fn plan_tool_batches(calls: Vec<ToolCall>, registry: &ToolRegistry) -> Vec<ToolBatch> {
    let mut batches: Vec<ToolBatch> = Vec::new();
    for call in calls {
        let class = registry.concurrency(&call.name, &call.arguments);
        match batches.last_mut() {
            // Read-only calls all share one class, so any maximal run merges.
            Some(batch)
                if batch.class == Concurrency::ReadOnly && class == Concurrency::ReadOnly =>
            {
                batch.calls.push(call);
            }
            // Parallel calls merge only with the same fan-out tool so two
            // different parallelizable tools cannot interleave their slots.
            Some(batch)
                if batch.class == Concurrency::Parallel
                    && class == Concurrency::Parallel
                    && batch.calls.iter().all(|prior| prior.name == call.name) =>
            {
                batch.calls.push(call);
            }
            _ => batches.push(ToolBatch {
                calls: vec![call],
                class,
            }),
        }
    }
    batches
}

impl Agent {
    /// Dispatch a turn's tool calls in program order, running provably
    /// read-only batches concurrently (bounded by
    /// [`MAX_CONCURRENT_READ_ONLY_TOOLS`]) and exclusive calls alone.
    ///
    /// Event contract: `ToolCallStarted` fires only when an execution future
    /// actually launches (never for a merely requested or queued call), and
    /// `ToolCallFinished` fires exactly once per started call as its future
    /// resolves — so live finish order may be completion order. History and
    /// the durable session remain in original call order.
    ///
    /// Cancellation is uniform and exactly-once: the in-flight batch shares
    /// a child token of the turn's `cancel`, so an interrupt kills the whole
    /// batch while leaving the turn token untouched for the next phase.
    /// Calls that completed keep their real events; launched-but-unfinished
    /// calls get one synthetic cancelled finish; calls never launched get a
    /// synthetic start + cancelled finish pair so every frontend sees a
    /// balanced lifecycle; all of them still receive failed tool results in
    /// parent history so provider history stays valid.
    pub(crate) async fn dispatch_tool_batches(
        &mut self,
        tool_calls: Vec<ToolCall>,
        events: &mpsc::UnboundedSender<AgentEvent>,
        input: &mut mpsc::UnboundedReceiver<InputMessage>,
        cancel: &CancellationToken,
    ) -> Result<(), TurnError> {
        for batch in plan_tool_batches(tool_calls, &self.tools) {
            let limit = if batch.concurrent() {
                match batch.class {
                    Concurrency::ReadOnly => MAX_CONCURRENT_READ_ONLY_TOOLS,
                    Concurrency::Parallel => self.subagent_limits.max_concurrent,
                    Concurrency::Exclusive => 1,
                }
            } else {
                1
            };
            if batch.calls.len() > 1 {
                tracing::debug!(
                    count = batch.calls.len(),
                    class = ?batch.class,
                    launch_limit = limit,
                    "running concurrent tool batch"
                );
            }
            // The batch shares a child token so an interrupt kills exactly
            // this phase; the turn token stays clean for the next batch.
            let batch_cancel = cancel.child_token();

            let mut futures: FuturesUnordered<ToolRun<'_>> = FuturesUnordered::new();
            let mut next_launch = 0usize;
            let mut starts: Vec<Option<Instant>> = (0..batch.calls.len()).map(|_| None).collect();
            let registry = &self.tools;
            while next_launch < batch.calls.len().min(limit) {
                let (future, started) = launch_call(
                    registry,
                    &batch.calls[next_launch],
                    next_launch,
                    batch_cancel.clone(),
                    events,
                );
                starts[next_launch] = Some(started);
                futures.push(future);
                next_launch += 1;
            }

            let mut slots: Vec<Option<(ToolOutput, Instant)>> =
                (0..batch.calls.len()).map(|_| None).collect();
            let mut finished = 0usize;
            let mut drain_before_cancel = false;
            loop {
                tokio::select! {
                    item = futures.next() => match item {
                        Some((index, result, started)) => {
                            // Live finish event: emitted in completion order,
                            // keyed by call id. Durable results below stay in
                            // original call order regardless.
                            send_finished(events, &batch.calls[index], &result, started);
                            slots[index] = Some((result, started));
                            finished += 1;
                            if finished == slots.len() {
                                break;
                            }
                            // Refill the freed slot from the remaining
                            // calls until the batch is exhausted.
                            if next_launch < batch.calls.len() {
                                let registry = &self.tools;
                                let (future, started) = launch_call(
                                    registry,
                                    &batch.calls[next_launch],
                                    next_launch,
                                    batch_cancel.clone(),
                                    events,
                                );
                                starts[next_launch] = Some(started);
                                futures.push(future);
                                next_launch += 1;
                            }
                        }
                        None => break,
                    },
                    message = input.recv(), if self.input_open => match message {
                        Some(InputMessage::Interrupt) => {
                            // First harvest completions that won the race; only
                            // then signal cancellation to unresolved calls.
                            drain_before_cancel = true;
                            break;
                        }
                        Some(message) => self.queued.push_back(message),
                        None => self.input_open = false,
                    },
                    _ = cancel.cancelled() => {
                        batch_cancel.cancel();
                        break;
                    }
                }
            }
            // An explicit interrupt can race a mutating call's final ready
            // result. Drain only completions available before broadcasting
            // cancellation; application shutdown has already cancelled the
            // parent token, so polling cancellation-aware tools there would
            // run their cancelled branch rather than harvest prior work.
            if drain_before_cancel {
                while let Some(Some((index, result, started))) = futures.next().now_or_never() {
                    send_finished(events, &batch.calls[index], &result, started);
                    slots[index] = Some((result, started));
                }
                batch_cancel.cancel();
            }
            drop(futures);

            // Drain whatever completed, then synthesize "cancelled" results
            // for the rest — mirroring the historical single-call behavior —
            // and end the turn. Finished slots already had their finish event
            // sent at resolution time, so they must not get a second one;
            // every call still gets its history entry and durable result.
            let mut cancelled = false;
            for (index, call) in batch.calls.iter().enumerate() {
                let mut was_cancelled = false;
                let (result, started) = match slots[index].take() {
                    Some((result, started)) => (result, started),
                    None => {
                        was_cancelled = true;
                        cancelled = true;
                        // A call never launched because the batch was
                        // interrupted still needs a balanced UI lifecycle:
                        // synthetic start now, cancelled finish below.
                        // Launched-but-unfinished calls already had their
                        // start; all of them get one cancelled finish here
                        // plus a failed durable result either way, so both
                        // frontends and provider history stay valid.
                        let started = match starts[index] {
                            Some(started) => started,
                            None => {
                                send_started(events, call);
                                Instant::now()
                            }
                        };
                        let cancellation =
                            if starts[index].is_some() && batch.class != Concurrency::ReadOnly {
                                "cancelled; execution status unknown"
                            } else {
                                "cancelled"
                            };
                        (
                            ToolOutput {
                                content: cancellation.to_owned(),
                                is_error: true,
                                summary: call_summary(&call.name, &call.arguments),
                            },
                            started,
                        )
                    }
                };
                if was_cancelled {
                    // Historical interrupt events carried an empty output
                    // field with the error carrying the reason.
                    send(
                        events,
                        AgentEvent::ToolCallFinished {
                            call_id: call.id.clone(),
                            name: call.name.clone(),
                            summary: result.summary.clone(),
                            ok: false,
                            duration_ms: started.elapsed().as_millis() as u64,
                            output: String::new(),
                            error: Some(result.content.clone()),
                        },
                    );
                }
                self.persist_tool_result(call, &result.content, result.is_error, events)?;
                self.history.push(Message {
                    role: Role::Tool,
                    content: vec![Content::ToolResult {
                        tool_call_id: call.id.clone(),
                        content: result.content.clone(),
                        is_error: result.is_error,
                    }],
                });
            }
            if cancelled {
                self.persist_cancelled("tool execution interrupted", events);
                send(events, AgentEvent::TurnFinished);
                if self.cancel.is_cancelled() {
                    return Err(TurnError::Shutdown);
                }
                return Ok(());
            }
        }
        Ok(())
    }
}

/// Launch one call's execution future: announce it (`ToolCallStarted` with
/// its original call id) immediately before pushing, stamping the true start
/// instant used later for the duration in `ToolCallFinished`. Each future
/// carries its own slot index; results land in that slot, never in
/// completion order.
fn launch_call<'a>(
    registry: &'a ToolRegistry,
    call: &'a ToolCall,
    index: usize,
    cancel: CancellationToken,
    events: &mpsc::UnboundedSender<AgentEvent>,
) -> (ToolRun<'a>, Instant) {
    send_started(events, call);
    let started = Instant::now();
    let name = call.name.clone();
    let arguments = call.arguments.clone();
    (
        Box::pin(async move {
            let result = registry.execute(&name, arguments, cancel).await;
            (index, result, started)
        }),
        started,
    )
}

fn send_started(events: &mpsc::UnboundedSender<AgentEvent>, call: &ToolCall) {
    send(
        events,
        AgentEvent::ToolCallStarted {
            call_id: call.id.clone(),
            name: call.name.clone(),
            summary: call_summary(&call.name, &call.arguments),
        },
    );
}

fn send_finished(
    events: &mpsc::UnboundedSender<AgentEvent>,
    call: &ToolCall,
    result: &ToolOutput,
    started: Instant,
) {
    send(
        events,
        AgentEvent::ToolCallFinished {
            call_id: call.id.clone(),
            name: call.name.clone(),
            summary: result.summary.clone(),
            ok: !result.is_error,
            duration_ms: started.elapsed().as_millis() as u64,
            output: result.content.clone(),
            error: result.is_error.then(|| result.content.clone()),
        },
    );
}
