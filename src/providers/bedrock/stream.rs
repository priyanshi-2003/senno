use std::collections::BTreeMap;
use std::mem;
use std::time::Duration;

use aws_sdk_bedrockruntime::primitives::event_stream::EventReceiver;
use aws_sdk_bedrockruntime::types::{
    ContentBlockDelta, ContentBlockStart, ConverseStreamOutput, StopReason,
    error::ConverseStreamOutputError,
};
use futures::{Stream, StreamExt, stream};
use serde_json::{Map, Value};
use tokio::time;

use crate::error::Error;
use crate::provider::LlmStream;
use crate::types::{StreamChunk, Usage};

use super::converse::to_usage;
use super::error::{PROVIDER, from_sdk};

type Receiver = EventReceiver<ConverseStreamOutput, ConverseStreamOutputError>;

#[derive(Default)]
struct PendingTool {
    id: String,
    name: String,
    input: String,
    closed: bool,
}

#[derive(Default)]
pub(super) struct StreamState {
    tools: BTreeMap<i32, PendingTool>,
    stop_reason: Option<String>,
    done: bool,
}

impl StreamState {
    pub(super) fn on_event(&mut self, event: ConverseStreamOutput) -> Vec<StreamChunk> {
        match event {
            ConverseStreamOutput::ContentBlockStart(start) => {
                if let Some(ContentBlockStart::ToolUse(tool)) = start.start() {
                    self.tools.insert(
                        start.content_block_index(),
                        PendingTool {
                            id: tool.tool_use_id().to_string(),
                            name: tool.name().to_string(),
                            ..PendingTool::default()
                        },
                    );
                }
                Vec::new()
            }
            ConverseStreamOutput::ContentBlockDelta(delta) => match delta.delta() {
                Some(ContentBlockDelta::Text(text)) if !text.is_empty() => {
                    vec![StreamChunk::Text(text.clone())]
                }
                Some(ContentBlockDelta::ToolUse(fragment)) => {
                    if let Some(tool) = self.tools.get_mut(&delta.content_block_index()) {
                        tool.input.push_str(fragment.input());
                    }
                    Vec::new()
                }
                _ => Vec::new(),
            },
            ConverseStreamOutput::ContentBlockStop(stop) => {
                if let Some(tool) = self.tools.get_mut(&stop.content_block_index()) {
                    tool.closed = true;
                }
                Vec::new()
            }
            ConverseStreamOutput::MessageStop(stop) => {
                self.stop_reason = Some(stop.stop_reason().as_str().to_string());
                self.release_tools()
            }
            ConverseStreamOutput::Metadata(meta) => self.emit_done(meta.usage().map(to_usage)),
            _ => Vec::new(),
        }
    }

    pub(super) fn finish(&mut self) -> Result<Vec<StreamChunk>, Error> {
        if self.stop_reason.is_none() {
            return Err(Error::provider(
                PROVIDER,
                format!(
                    "ConverseStream ended before messageStop; dropped {} unfinished tool call(s)",
                    self.tools.len()
                ),
            ));
        }
        let mut chunks = self.release_tools();
        chunks.extend(self.emit_done(None));
        Ok(chunks)
    }

    fn release_tools(&mut self) -> Vec<StreamChunk> {
        let cut_off = self.stop_reason.as_deref() == Some(StopReason::MaxTokens.as_str());
        mem::take(&mut self.tools)
            .into_values()
            .filter_map(|tool| finish_tool(tool, cut_off))
            .collect()
    }

    fn emit_done(&mut self, usage: Option<Usage>) -> Vec<StreamChunk> {
        if self.done {
            return Vec::new();
        }
        self.done = true;
        vec![StreamChunk::Done {
            finish_reason: self
                .stop_reason
                .clone()
                .unwrap_or_else(|| "end_turn".into()),
            usage,
        }]
    }
}

fn finish_tool(tool: PendingTool, cut_off: bool) -> Option<StreamChunk> {
    let parsed = parse_tool_input(&tool.input);
    let arguments = match parsed {
        Some(arguments) if tool.closed || !cut_off => arguments,
        _ if cut_off => {
            tracing::warn!(tool = %tool.name, "dropped a bedrock tool call cut off by max_tokens");
            return None;
        }
        _ => {
            tracing::warn!(tool = %tool.name, "bedrock tool input was not a JSON object; sending {{}}");
            Value::Object(Map::new())
        }
    };
    Some(StreamChunk::ToolCall {
        id: tool.id,
        name: tool.name,
        arguments,
        thought_signature: None,
    })
}

fn parse_tool_input(input: &str) -> Option<Value> {
    if input.trim().is_empty() {
        return Some(Value::Object(Map::new()));
    }
    serde_json::from_str::<Value>(input)
        .ok()
        .filter(Value::is_object)
}

pub(super) fn into_chunks(receiver: Receiver, idle_timeout: Duration) -> LlmStream {
    let events = stream::unfold(receiver, |mut receiver| async move {
        match receiver.recv().await {
            Ok(Some(event)) => Some((Ok(event), receiver)),
            Ok(None) => None,
            Err(err) => Some((Err(from_sdk("ConverseStream", err)), receiver)),
        }
    });
    drive_events(events, idle_timeout)
}

fn drive_events<S>(events: S, idle_timeout: Duration) -> LlmStream
where
    S: Stream<Item = Result<ConverseStreamOutput, Error>> + Send + 'static,
{
    Box::pin(async_stream::stream! {
        let mut events = Box::pin(events);
        let mut state = StreamState::default();
        loop {
            let Ok(next) = time::timeout(idle_timeout, events.next()).await else {
                yield Err(Error::provider(
                    PROVIDER,
                    format!(
                        "ConverseStream sent nothing for {} ms; dropped the unfinished reply",
                        idle_timeout.as_millis()
                    ),
                ));
                break;
            };
            match next {
                Some(Ok(event)) => {
                    for chunk in state.on_event(event) {
                        yield Ok(chunk);
                    }
                }
                Some(Err(err)) => {
                    yield Err(err);
                    break;
                }
                None => {
                    match state.finish() {
                        Ok(chunks) => {
                            for chunk in chunks {
                                yield Ok(chunk);
                            }
                        }
                        Err(err) => yield Err(err),
                    }
                    break;
                }
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use aws_sdk_bedrockruntime::types::{
        ContentBlockDeltaEvent, ContentBlockStartEvent, ContentBlockStopEvent,
        ConverseStreamMetadataEvent, MessageStopEvent, StopReason, TokenUsage, ToolUseBlockDelta,
        ToolUseBlockStart,
    };

    fn text(index: i32, s: &str) -> ConverseStreamOutput {
        ConverseStreamOutput::ContentBlockDelta(
            ContentBlockDeltaEvent::builder()
                .content_block_index(index)
                .delta(ContentBlockDelta::Text(s.into()))
                .build()
                .unwrap(),
        )
    }

    fn tool_start(index: i32, id: &str, name: &str) -> ConverseStreamOutput {
        ConverseStreamOutput::ContentBlockStart(
            ContentBlockStartEvent::builder()
                .content_block_index(index)
                .start(ContentBlockStart::ToolUse(
                    ToolUseBlockStart::builder()
                        .tool_use_id(id)
                        .name(name)
                        .build()
                        .unwrap(),
                ))
                .build()
                .unwrap(),
        )
    }

    fn tool_delta(index: i32, json: &str) -> ConverseStreamOutput {
        ConverseStreamOutput::ContentBlockDelta(
            ContentBlockDeltaEvent::builder()
                .content_block_index(index)
                .delta(ContentBlockDelta::ToolUse(
                    ToolUseBlockDelta::builder().input(json).build().unwrap(),
                ))
                .build()
                .unwrap(),
        )
    }

    fn stop(index: i32) -> ConverseStreamOutput {
        ConverseStreamOutput::ContentBlockStop(
            ContentBlockStopEvent::builder()
                .content_block_index(index)
                .build()
                .unwrap(),
        )
    }

    fn message_stop(reason: StopReason) -> ConverseStreamOutput {
        ConverseStreamOutput::MessageStop(
            MessageStopEvent::builder()
                .stop_reason(reason)
                .build()
                .unwrap(),
        )
    }

    fn metadata(input: i32, output: i32) -> ConverseStreamOutput {
        ConverseStreamOutput::Metadata(
            ConverseStreamMetadataEvent::builder()
                .usage(
                    TokenUsage::builder()
                        .input_tokens(input)
                        .output_tokens(output)
                        .total_tokens(input + output)
                        .build()
                        .unwrap(),
                )
                .build(),
        )
    }

    fn run_raw(events: Vec<ConverseStreamOutput>) -> Result<Vec<StreamChunk>, Error> {
        let mut state = StreamState::default();
        let mut chunks: Vec<StreamChunk> =
            events.into_iter().flat_map(|e| state.on_event(e)).collect();
        chunks.extend(state.finish()?);
        Ok(chunks)
    }

    fn run(events: Vec<ConverseStreamOutput>) -> Vec<StreamChunk> {
        run_raw(events).expect("stream should finish cleanly")
    }

    #[test]
    fn text_then_tool_call_then_done_with_usage() {
        let chunks = run(vec![
            text(0, "Looking"),
            text(0, " now."),
            stop(0),
            tool_start(1, "tool-1", "search_catalog"),
            tool_delta(1, r#"{"catalog":{"query":"#),
            tool_delta(1, r#""red shoes"}}"#),
            stop(1),
            message_stop(StopReason::ToolUse),
            metadata(1200, 80),
        ]);

        assert!(matches!(&chunks[0], StreamChunk::Text(t) if t == "Looking"));
        assert!(matches!(&chunks[1], StreamChunk::Text(t) if t == " now."));
        match &chunks[2] {
            StreamChunk::ToolCall {
                id,
                name,
                arguments,
                ..
            } => {
                assert_eq!(id, "tool-1");
                assert_eq!(name, "search_catalog");
                assert_eq!(
                    arguments,
                    &serde_json::json!({"catalog": {"query": "red shoes"}})
                );
            }
            other => panic!("expected tool call, got {other:?}"),
        }
        match &chunks[3] {
            StreamChunk::Done {
                finish_reason,
                usage,
            } => {
                assert_eq!(finish_reason, "tool_use");
                let usage = usage.as_ref().expect("usage");
                assert_eq!(usage.input_tokens, Some(1200));
                assert_eq!(usage.output_tokens, Some(80));
            }
            other => panic!("expected done, got {other:?}"),
        }
        assert_eq!(chunks.len(), 4, "exactly one Done");
    }

    #[test]
    fn tool_with_no_input_gets_empty_object() {
        let chunks = run(vec![
            tool_start(0, "t", "get_cart"),
            stop(0),
            message_stop(StopReason::ToolUse),
        ]);
        assert!(matches!(&chunks[0], StreamChunk::ToolCall { arguments, .. }
            if arguments == &serde_json::json!({})));
    }

    #[test]
    fn stream_without_metadata_still_ends_with_done() {
        let chunks = run(vec![text(0, "hi"), message_stop(StopReason::EndTurn)]);
        assert!(
            matches!(chunks.last(), Some(StreamChunk::Done { finish_reason, usage: None })
            if finish_reason == "end_turn")
        );
    }

    #[test]
    fn cut_off_stream_is_a_retryable_error_without_tool_calls() {
        let mut state = StreamState::default();
        let mut chunks = Vec::new();
        for event in [
            text(0, "Searching"),
            tool_start(1, "t", "search"),
            tool_delta(1, r#"{"q":"#),
        ] {
            chunks.extend(state.on_event(event));
        }
        let err = state.finish().unwrap_err();
        assert!(err.is_retryable(), "{err}");
        assert!(
            !chunks
                .iter()
                .any(|c| matches!(c, StreamChunk::ToolCall { .. } | StreamChunk::Done { .. }))
        );
    }

    #[test]
    fn unclosed_tool_block_after_message_stop_is_flushed() {
        let chunks = run(vec![
            tool_start(0, "t", "search"),
            tool_delta(0, r#"{"q":1}"#),
            message_stop(StopReason::ToolUse),
        ]);
        assert!(matches!(&chunks[0], StreamChunk::ToolCall { name, .. } if name == "search"));
        assert!(matches!(&chunks[1], StreamChunk::Done { .. }));
    }

    #[test]
    fn max_tokens_drops_a_cut_off_tool_call() {
        let chunks = run(vec![
            text(0, "Let me search"),
            tool_start(1, "t", "search"),
            tool_delta(1, r#"{"q":"red sh"#),
            message_stop(StopReason::MaxTokens),
        ]);
        assert!(
            !chunks
                .iter()
                .any(|c| matches!(c, StreamChunk::ToolCall { .. }))
        );
        assert!(
            matches!(chunks.last(), Some(StreamChunk::Done { finish_reason, .. })
            if finish_reason == "max_tokens")
        );
    }

    #[test]
    fn max_tokens_keeps_a_tool_call_that_finished() {
        let chunks = run(vec![
            tool_start(0, "t", "search"),
            tool_delta(0, r#"{"q":"red"}"#),
            stop(0),
            message_stop(StopReason::MaxTokens),
        ]);
        assert!(matches!(&chunks[0], StreamChunk::ToolCall { arguments, .. }
            if arguments == &serde_json::json!({"q": "red"})));
    }

    #[test]
    fn invalid_tool_input_is_sent_as_an_empty_object_not_null() {
        let chunks = run(vec![
            tool_start(0, "t", "search"),
            tool_delta(0, r#"{"q":"#),
            stop(0),
            message_stop(StopReason::ToolUse),
        ]);
        assert!(matches!(&chunks[0], StreamChunk::ToolCall { arguments, .. }
            if arguments == &serde_json::json!({})));
    }

    #[test]
    fn tool_calls_wait_for_message_stop() {
        let mut state = StreamState::default();
        let mut chunks = Vec::new();
        for event in [
            tool_start(0, "t", "add_to_cart"),
            tool_delta(0, r#"{"sku":"1"}"#),
            stop(0),
        ] {
            chunks.extend(state.on_event(event));
        }
        assert!(chunks.is_empty(), "{chunks:?}");
        let err = state.finish().unwrap_err();
        assert!(err.is_retryable(), "{err}");
    }

    #[tokio::test]
    async fn a_stream_that_goes_silent_mid_reply_times_out() {
        let events = stream::iter(vec![Ok(text(0, "Hel"))]).chain(stream::pending());
        let chunks: Vec<_> = drive_events(events, Duration::from_millis(50))
            .collect()
            .await;
        assert!(matches!(&chunks[0], Ok(StreamChunk::Text(t)) if t == "Hel"));
        match &chunks[1] {
            Err(err) => assert!(err.is_retryable(), "{err}"),
            other => panic!("expected a timeout error, got {other:?}"),
        }
        assert_eq!(chunks.len(), 2);
    }
}
