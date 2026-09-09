use super::InputMessage;
use super::{Agent, AgentEvent, TurnError, send};
use futures_util::FutureExt;
use futures_util::stream::{FuturesUnordered, StreamExt};
use futures_util::task::noop_waker_ref;
use llm::{Content, Message, Role, ToolCall};
use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Instant;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tools::{Concurrency, ToolOutput, ToolRegistry, call_summary};

/// Upper bound on read-only tool calls running at once.  Read tools are
/// cheap, but the shared `FileSearchIndex` already gates its own concurrency
/// and a runaway batch would thrash the disk; 8 keeps latency wins while
/// staying polite.
pub(crate) const MAX_CONCURRENT_READ_ONLY_TOOLS: usize = 8;

/// Upper bound on concurrent potentially mutating fan-out calls.
pub(crate) const MAX_CONCURRENT_PARALLEL_TOOLS: usize = 4;

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

/// The reason a shared tool batch stopped admitting work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DispatchCancellation {
    /// The user interrupted the current turn or a child run.
    Explicit,
    /// The owning agent is shutting down.
    Shutdown,
}

/// State of one call returned by the shared batch executor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CallState {
    /// The tool future produced a real result.
    Completed,
    /// The call was admitted or not admitted, then cancelled before a result.
    Cancelled { launched: bool },
}

/// One result in original provider call order. Completion hooks observe live
/// completion order, while callers consume this vector in durable order.
pub(crate) struct CallOutcome {
    pub(crate) output: ToolOutput,
}

/// Result of executing one planned batch.
pub(crate) struct BatchOutcome {
    pub(crate) outcomes: Vec<CallOutcome>,
    pub(crate) cancellation: Option<DispatchCancellation>,
}

/// UI-only hooks for the shared executor. Hooks never persist or mutate an
/// [`Agent`], which keeps the registry borrow independent of caller state.
pub(crate) trait ToolDispatchHooks {
    fn started(&mut self, call: &ToolCall, started: Instant, synthetic: bool);

    fn finished(
        &mut self,
        call: &ToolCall,
        output: &ToolOutput,
        started: Instant,
        state: CallState,
    );
}

/// Parent hook translating shared lifecycle callbacks into agent events.
struct AgentToolDispatchHooks<'a> {
    events: &'a mpsc::UnboundedSender<AgentEvent>,
}

impl<'a> AgentToolDispatchHooks<'a> {
    fn new(events: &'a mpsc::UnboundedSender<AgentEvent>) -> Self {
        Self { events }
    }
}

impl ToolDispatchHooks for AgentToolDispatchHooks<'_> {
    fn started(&mut self, call: &ToolCall, _started: Instant, _synthetic: bool) {
        send_started(self.events, call);
    }

    fn finished(
        &mut self,
        call: &ToolCall,
        output: &ToolOutput,
        started: Instant,
        state: CallState,
    ) {
        match state {
            CallState::Completed => send_finished(self.events, call, output, started),
            CallState::Cancelled { .. } => send(
                self.events,
                AgentEvent::ToolCallFinished {
                    call_id: call.id.clone(),
                    name: call.name.clone(),
                    summary: output.summary.clone(),
                    ok: false,
                    duration_ms: started.elapsed().as_millis() as u64,
                    // A cancelled call has no result to display, but its
                    // execution status is useful in the error field.
                    output: String::new(),
                    error: Some(output.content.clone()),
                },
            ),
        }
    }
}

/// Child runs have no frontend lifecycle, but use the same scheduling and
/// cancellation implementation as the parent.
pub(crate) struct NoopToolDispatchHooks;

impl ToolDispatchHooks for NoopToolDispatchHooks {
    fn started(&mut self, _call: &ToolCall, _started: Instant, _synthetic: bool) {}

    fn finished(
        &mut self,
        _call: &ToolCall,
        _output: &ToolOutput,
        _started: Instant,
        _state: CallState,
    ) {
    }
}

/// A cancellation-only control future used by child runs.
pub(crate) struct CancellationControl<'a> {
    future: Pin<Box<dyn Future<Output = ()> + Send + 'a>>,
    reason: DispatchCancellation,
}

impl<'a> CancellationControl<'a> {
    pub(crate) fn new(token: &'a CancellationToken, reason: DispatchCancellation) -> Self {
        Self {
            future: Box::pin(token.cancelled()),
            reason,
        }
    }
}

impl Future for CancellationControl<'_> {
    type Output = DispatchCancellation;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        if this.future.as_mut().poll(cx).is_ready() {
            Poll::Ready(this.reason)
        } else {
            Poll::Pending
        }
    }
}

/// Parent control also drains ordinary messages into the agent queue while
/// waiting for an interrupt or application shutdown. This keeps normal input
/// from competing with tool completions in the scheduler.
pub(crate) struct ParentDispatchControl<'a> {
    input: &'a mut mpsc::UnboundedReceiver<InputMessage>,
    queued: &'a mut VecDeque<InputMessage>,
    input_open: &'a mut bool,
    application: Pin<Box<dyn Future<Output = ()> + Send + 'a>>,
    turn: Pin<Box<dyn Future<Output = ()> + Send + 'a>>,
}

impl<'a> ParentDispatchControl<'a> {
    pub(crate) fn new(
        input: &'a mut mpsc::UnboundedReceiver<InputMessage>,
        queued: &'a mut VecDeque<InputMessage>,
        input_open: &'a mut bool,
        application: &'a CancellationToken,
        turn: &'a CancellationToken,
    ) -> Self {
        Self {
            input,
            queued,
            input_open,
            application: Box::pin(application.cancelled()),
            turn: Box::pin(turn.cancelled()),
        }
    }
}

impl Future for ParentDispatchControl<'_> {
    type Output = DispatchCancellation;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        if this.application.as_mut().poll(cx).is_ready() {
            return Poll::Ready(DispatchCancellation::Shutdown);
        }
        if this.turn.as_mut().poll(cx).is_ready() {
            return Poll::Ready(DispatchCancellation::Explicit);
        }
        if *this.input_open {
            loop {
                match Pin::new(&mut *this.input).poll_recv(cx) {
                    Poll::Ready(Some(InputMessage::Interrupt)) => {
                        return Poll::Ready(DispatchCancellation::Explicit);
                    }
                    Poll::Ready(Some(message)) => this.queued.push_back(message),
                    Poll::Ready(None) => {
                        *this.input_open = false;
                        break;
                    }
                    Poll::Pending => break,
                }
            }
        }
        Poll::Pending
    }
}

/// Poll a control future once without waiting. This closes the race where a
/// tool result and an interrupt become ready together: no replacement call is
/// admitted after the interrupt is observable.
fn poll_control_now<C>(control: &mut C) -> Option<DispatchCancellation>
where
    C: Future<Output = DispatchCancellation> + Unpin,
{
    let mut context = Context::from_waker(noop_waker_ref());
    match Pin::new(control).poll(&mut context) {
        Poll::Ready(reason) => Some(reason),
        Poll::Pending => None,
    }
}

/// One in-flight tool execution carrying its slot index so results can be
/// recorded in original provider call order rather than completion order.
type ToolRun<'a> = Pin<Box<dyn Future<Output = (usize, ToolOutput, Instant)> + Send + 'a>>;

/// Execute one planned batch with bounded admission and shared cancellation
/// semantics. Results are returned in `batch.calls` order; hooks are called
/// as work completes, so a frontend can render real-time completion order.
pub(crate) async fn execute_tool_batch<C, H>(
    registry: &ToolRegistry,
    batch: &ToolBatch,
    launch_limit: usize,
    control: &mut C,
    hooks: &mut H,
) -> BatchOutcome
where
    C: Future<Output = DispatchCancellation> + Unpin,
    H: ToolDispatchHooks,
{
    let launch_limit = launch_limit.max(1);
    let batch_cancel = CancellationToken::new();
    let mut futures: FuturesUnordered<ToolRun<'_>> = FuturesUnordered::new();
    let mut starts: Vec<Option<Instant>> = (0..batch.calls.len()).map(|_| None).collect();
    let mut slots: Vec<Option<CallOutcome>> = (0..batch.calls.len()).map(|_| None).collect();
    let mut next_launch = 0usize;
    let mut finished = 0usize;
    let mut cancellation = poll_control_now(control);

    while cancellation.is_none() && next_launch < batch.calls.len().min(launch_limit) {
        let call = &batch.calls[next_launch];
        let started = Instant::now();
        starts[next_launch] = Some(started);
        hooks.started(call, started, false);
        let name = call.name.clone();
        let arguments = call.arguments.clone();
        let index = next_launch;
        let cancel = batch_cancel.clone();
        futures.push(Box::pin(async move {
            let output = registry.execute(&name, arguments, cancel).await;
            (index, output, started)
        }));
        next_launch += 1;
    }

    while cancellation.is_none() && finished < batch.calls.len() {
        tokio::select! {
            biased;
            reason = &mut *control => {
                cancellation = Some(reason);
            }
            item = futures.next() => match item {
                Some((index, output, started)) => {
                    hooks.finished(&batch.calls[index], &output, started, CallState::Completed);
                    slots[index] = Some(CallOutcome { output });
                    finished += 1;
                    if finished == batch.calls.len() {
                        break;
                    }
                    // Poll the control before refilling a freed slot. A
                    // ready interrupt wins over a simultaneous completion.
                    if let Some(reason) = poll_control_now(control) {
                        cancellation = Some(reason);
                        break;
                    }
                    if next_launch < batch.calls.len() {
                        let call = &batch.calls[next_launch];
                        let started = Instant::now();
                        starts[next_launch] = Some(started);
                        hooks.started(call, started, false);
                        let name = call.name.clone();
                        let arguments = call.arguments.clone();
                        let index = next_launch;
                        let cancel = batch_cancel.clone();
                        futures.push(Box::pin(async move {
                            let output = registry.execute(&name, arguments, cancel).await;
                            (index, output, started)
                        }));
                        next_launch += 1;
                    }
                }
                None => break,
            }
        }
    }

    if cancellation.is_some() {
        // Harvest work that completed before cancellation was broadcast. The
        // local token is cancelled only after this non-blocking drain, so a
        // ready mutation result is never relabelled as unknown cancellation.
        while let Some(Some((index, output, started))) = futures.next().now_or_never() {
            hooks.finished(&batch.calls[index], &output, started, CallState::Completed);
            slots[index] = Some(CallOutcome { output });
        }
        batch_cancel.cancel();
    }
    drop(futures);

    let mut outcomes = Vec::with_capacity(batch.calls.len());
    for (index, call) in batch.calls.iter().enumerate() {
        let outcome = slots[index].take().unwrap_or_else(|| {
            let started = starts[index].unwrap_or_else(|| {
                let started = Instant::now();
                hooks.started(call, started, true);
                started
            });
            let launched = starts[index].is_some();
            let content = if launched && batch.class != Concurrency::ReadOnly {
                "cancelled; execution status unknown"
            } else {
                "cancelled"
            };
            let output = ToolOutput {
                content: content.to_owned(),
                is_error: true,
                summary: call_summary(&call.name, &call.arguments),
            };
            let state = CallState::Cancelled { launched };
            hooks.finished(call, &output, started, state);
            CallOutcome { output }
        });
        outcomes.push(outcome);
    }

    BatchOutcome {
        outcomes,
        cancellation,
    }
}

impl Agent {
    /// Dispatch a turn's tool calls in program order, running provably
    /// read-only batches concurrently (bounded by
    /// [`MAX_CONCURRENT_READ_ONLY_TOOLS`]) and exclusive calls alone.
    ///
    /// Event contract: `ToolCallStarted` fires when a call is admitted, and
    /// once more as a synthetic lifecycle marker for a never-admitted call
    /// cancelled before the batch could start it. `ToolCallFinished` fires
    /// exactly once for every such lifecycle, while real finishes can arrive
    /// in completion order. History and the durable session remain in
    /// original call order.
    ///
    /// Cancellation is uniform and exactly-once: the executor gives each
    /// batch a private tool token, drains ready results, then cancels the
    /// unresolved work before synthesizing its durable outcomes.
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

            let mut control = ParentDispatchControl::new(
                input,
                &mut self.queued,
                &mut self.input_open,
                &self.cancel,
                cancel,
            );
            let mut hooks = AgentToolDispatchHooks::new(events);
            let outcome =
                execute_tool_batch(&self.tools, &batch, limit, &mut control, &mut hooks).await;
            drop(control);

            for (call, call_outcome) in batch.calls.iter().zip(outcome.outcomes) {
                self.persist_tool_result(
                    call,
                    &call_outcome.output.content,
                    call_outcome.output.is_error,
                    events,
                )?;
                self.history.push(Message {
                    role: Role::Tool,
                    content: vec![Content::ToolResult {
                        tool_call_id: call.id.clone(),
                        content: call_outcome.output.content,
                        is_error: call_outcome.output.is_error,
                    }],
                });
            }

            if let Some(reason) = outcome.cancellation {
                // Mark the turn token so the enclosing turn returns rather
                // than issuing a provider request after an interrupt.
                if reason == DispatchCancellation::Explicit {
                    cancel.cancel();
                }
                self.persist_cancelled(
                    match reason {
                        DispatchCancellation::Explicit => "tool execution interrupted",
                        DispatchCancellation::Shutdown => "application shutdown",
                    },
                    events,
                )?;
                if reason == DispatchCancellation::Shutdown {
                    return Err(TurnError::Shutdown);
                }
                return Ok(());
            }
        }
        Ok(())
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use serde_json::{Value, json};
    use std::sync::Arc;
    use std::time::Duration;
    use tokio::sync::Notify;
    use tools::{Tool, ToolSpec};

    struct TestTool {
        class: Concurrency,
        started: Option<Arc<Notify>>,
    }

    #[async_trait]
    impl Tool for TestTool {
        fn spec(&self) -> ToolSpec {
            ToolSpec {
                definition: llm::ToolDefinition {
                    name: "test".into(),
                    description: "test tool".into(),
                    parameters: json!({"type": "object"}),
                },
                prompt: tools::ToolPrompt::default(),
            }
        }

        fn concurrency(&self, _args: &Value) -> Concurrency {
            self.class
        }

        async fn execute(&self, args: Value, _cancel: CancellationToken) -> ToolOutput {
            if let Some(started) = &self.started {
                started.notify_one();
                std::future::pending::<()>().await;
            }
            tokio::time::sleep(Duration::from_millis(
                args.get("delay_ms").and_then(Value::as_u64).unwrap_or(0),
            ))
            .await;
            ToolOutput {
                content: args["label"].as_str().unwrap_or_default().to_owned(),
                is_error: false,
                summary: "test".into(),
            }
        }
    }

    fn call(id: &str, label: &str, delay_ms: u64) -> ToolCall {
        ToolCall {
            id: id.into(),
            name: "test".into(),
            arguments: json!({"label": label, "delay_ms": delay_ms}),
        }
    }

    #[tokio::test]
    async fn shared_executor_keeps_original_result_order() {
        let registry = ToolRegistry::try_new(vec![Box::new(TestTool {
            class: Concurrency::Parallel,
            started: None,
        })])
        .unwrap();
        let batch = ToolBatch {
            calls: vec![call("first", "first", 30), call("second", "second", 1)],
            class: Concurrency::Parallel,
        };
        let cancel = CancellationToken::new();
        let mut control = CancellationControl::new(&cancel, DispatchCancellation::Explicit);
        let mut hooks = NoopToolDispatchHooks;
        let outcome = execute_tool_batch(&registry, &batch, 2, &mut control, &mut hooks).await;

        assert_eq!(outcome.cancellation, None);
        let labels: Vec<&str> = outcome
            .outcomes
            .iter()
            .map(|outcome| outcome.output.content.as_str())
            .collect();
        assert_eq!(labels, vec!["first", "second"]);
    }

    #[tokio::test]
    async fn shared_executor_retains_ready_result_racing_cancellation() {
        // A call that completed before cancellation was observed keeps its
        // real result: the ready-drain after cancellation harvests it
        // instead of relabelling it as unknown cancellation. This is the
        // shared-executor path both parent and child (subagent) dispatch
        // flow through, so it pins the behavior for both.
        let finished_first = Arc::new(Notify::new());
        let release_second = Arc::new(Notify::new());
        struct GatedTool {
            finished_first: Arc<Notify>,
            release_second: Arc<Notify>,
        }
        #[async_trait]
        impl Tool for GatedTool {
            fn spec(&self) -> ToolSpec {
                ToolSpec {
                    definition: llm::ToolDefinition {
                        name: "gated".into(),
                        description: "gated tool".into(),
                        parameters: json!({"type": "object"}),
                    },
                    prompt: tools::ToolPrompt::default(),
                }
            }
            fn concurrency(&self, _args: &Value) -> Concurrency {
                Concurrency::Parallel
            }
            async fn execute(&self, args: Value, _cancel: CancellationToken) -> ToolOutput {
                if args["label"] == "first" {
                    self.finished_first.notify_one();
                    // Stay in-flight (but already completed-from-the-test's
                    // view) until the test releases us — then report the
                    // real result while cancellation is already pending.
                    self.release_second.notified().await;
                    ToolOutput {
                        content: "first".into(),
                        is_error: false,
                        summary: "gated".into(),
                    }
                } else {
                    // Park until the batch token is cancelled; the drop
                    // ends this future without a result (unresolved).
                    std::future::pending::<()>().await;
                    unreachable!("cancelled futures are dropped before returning")
                }
            }
        }
        let registry = ToolRegistry::try_new(vec![Box::new(GatedTool {
            finished_first: finished_first.clone(),
            release_second: release_second.clone(),
        })])
        .unwrap();
        let batch = ToolBatch {
            calls: vec![
                ToolCall {
                    id: "first".into(),
                    name: "gated".into(),
                    arguments: json!({"label": "first"}),
                },
                ToolCall {
                    id: "second".into(),
                    name: "gated".into(),
                    arguments: json!({"label": "second"}),
                },
            ],
            class: Concurrency::Parallel,
        };
        let cancel = CancellationToken::new();
        let mut control = CancellationControl::new(&cancel, DispatchCancellation::Explicit);
        let mut hooks = NoopToolDispatchHooks;
        let mut execution = Box::pin(execute_tool_batch(
            &registry,
            &batch,
            2,
            &mut control,
            &mut hooks,
        ));
        // Both calls are in flight. Cancel while the first is parked just
        // before reporting: the `select!` observes cancellation, then the
        // ready-drain must still harvest the first call's real result once
        // the test releases it.
        tokio::select! {
            _ = finished_first.notified() => cancel.cancel(),
            _ = &mut execution => panic!("batch completed before cancellation"),
        }
        release_second.notify_one();
        let outcome = execution.await;

        assert_eq!(outcome.cancellation, Some(DispatchCancellation::Explicit));
        assert_eq!(outcome.outcomes[0].output.content, "first");
        assert!(!outcome.outcomes[0].output.is_error);
        assert_eq!(
            outcome.outcomes[1].output.content,
            "cancelled; execution status unknown"
        );
    }

    #[tokio::test]
    async fn shared_executor_distinguishes_launched_and_queued_cancellation() {
        let started = Arc::new(Notify::new());
        let registry = ToolRegistry::try_new(vec![Box::new(TestTool {
            class: Concurrency::Exclusive,
            started: Some(started.clone()),
        })])
        .unwrap();
        let batch = ToolBatch {
            calls: vec![call("first", "first", 0), call("second", "second", 0)],
            class: Concurrency::Exclusive,
        };
        let cancel = CancellationToken::new();
        let mut control = CancellationControl::new(&cancel, DispatchCancellation::Explicit);
        let mut hooks = NoopToolDispatchHooks;
        let mut execution = Box::pin(execute_tool_batch(
            &registry,
            &batch,
            1,
            &mut control,
            &mut hooks,
        ));
        tokio::select! {
            outcome = &mut execution => panic!("batch completed unexpectedly: {}", outcome.outcomes.len()),
            _ = started.notified() => cancel.cancel(),
        }
        let outcome = execution.await;

        assert_eq!(outcome.cancellation, Some(DispatchCancellation::Explicit));
        assert_eq!(
            outcome.outcomes[0].output.content,
            "cancelled; execution status unknown"
        );
        assert_eq!(outcome.outcomes[1].output.content, "cancelled");
    }
}
