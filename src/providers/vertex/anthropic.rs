use crate::error::Error;
use crate::types::{
    ContentPart, GenerateRequest, GenerateResponse, ParameterSchema, Role, StreamChunk, Usage,
};

use super::client::VertexClient;
use super::config::{ResolvedAuth, regional_host};
use futures::Stream;
use serde::{Deserialize, Serialize};
use std::pin::Pin;

#[derive(Serialize)]
struct Request {
    anthropic_version: String,
    max_tokens: u32,
    messages: Vec<WireMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<WireSystem>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<WireToolDef>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

#[derive(Serialize, Deserialize)]
struct WireMessage {
    role: String,
    content: WireContent,
}

#[derive(Serialize, Deserialize)]
#[serde(untagged)]
enum WireContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

#[derive(Serialize)]
#[serde(untagged)]
enum WireSystem {
    Text(String),
    Blocks(Vec<SystemBlock>),
}

#[derive(Serialize)]
struct SystemBlock {
    r#type: &'static str,
    text: String,
    cache_control: CacheControl,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
struct CacheControl {
    r#type: CacheType,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum CacheType {
    Ephemeral,
}

const EPHEMERAL: CacheControl = CacheControl {
    r#type: CacheType::Ephemeral,
};

#[derive(Serialize, Deserialize)]
#[serde(tag = "type")]
enum ContentBlock {
    #[serde(rename = "text")]
    Text {
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
}

impl ContentBlock {
    fn mark_cached(&mut self) {
        let (ContentBlock::Text { cache_control, .. }
        | ContentBlock::ToolUse { cache_control, .. }
        | ContentBlock::ToolResult { cache_control, .. }) = self;
        *cache_control = Some(EPHEMERAL);
    }
}

#[derive(Serialize)]
struct WireToolDef {
    name: String,
    description: String,
    input_schema: ParameterSchema,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_control: Option<CacheControl>,
}

#[derive(Deserialize)]
struct Response {
    content: Vec<ContentBlock>,
    stop_reason: Option<String>,
    usage: Option<WireUsage>,
}

#[derive(Deserialize)]
struct WireUsage {
    input_tokens: Option<u32>,
    output_tokens: Option<u32>,
    cache_creation_input_tokens: Option<u32>,
    cache_read_input_tokens: Option<u32>,
}

impl From<WireUsage> for Usage {
    fn from(u: WireUsage) -> Self {
        let input = sum_opt(
            u.input_tokens,
            sum_opt(u.cache_creation_input_tokens, u.cache_read_input_tokens),
        );
        Usage {
            input_tokens: input,
            cached_input_tokens: u.cache_read_input_tokens,
            cache_write_input_tokens: u.cache_creation_input_tokens,
            output_tokens: u.output_tokens,
            reasoning_tokens: None,
            total_tokens: sum_opt(input, u.output_tokens),
        }
    }
}

fn sum_opt(a: Option<u32>, b: Option<u32>) -> Option<u32> {
    match (a, b) {
        (None, None) => None,
        _ => Some(a.unwrap_or(0).saturating_add(b.unwrap_or(0))),
    }
}

fn merge_usage(prev: Option<Usage>, next: Usage) -> Usage {
    let Some(p) = prev else { return next };
    let input_tokens = next.input_tokens.or(p.input_tokens);
    let output_tokens = next.output_tokens.or(p.output_tokens);
    Usage {
        input_tokens,
        cached_input_tokens: next.cached_input_tokens.or(p.cached_input_tokens),
        cache_write_input_tokens: next.cache_write_input_tokens.or(p.cache_write_input_tokens),
        output_tokens,
        reasoning_tokens: None,
        total_tokens: sum_opt(input_tokens, output_tokens),
    }
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum StreamEvent {
    #[serde(rename = "message_start")]
    MessageStart {
        #[serde(default)]
        message: Option<StreamMessageInit>,
    },
    #[serde(rename = "content_block_start")]
    ContentBlockStart {
        #[serde(default)]
        content_block: Option<ContentBlockInfo>,
    },
    #[serde(rename = "content_block_delta")]
    ContentBlockDelta { delta: Delta },
    #[serde(rename = "content_block_stop")]
    ContentBlockStop {},
    #[serde(rename = "message_delta")]
    MessageDelta { delta: MessageDeltaBody },
    #[serde(rename = "message_stop")]
    MessageStop,
    #[serde(rename = "ping")]
    Ping,
}

#[derive(Deserialize)]
struct StreamMessageInit {
    #[serde(default)]
    usage: Option<WireUsage>,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum ContentBlockInfo {
    #[serde(rename = "tool_use")]
    ToolUse { id: String, name: String },
    #[serde(rename = "text")]
    Text {},
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum Delta {
    #[serde(rename = "text_delta")]
    TextDelta { text: String },
    #[serde(rename = "input_json_delta")]
    InputJsonDelta { partial_json: String },
}

#[derive(Deserialize)]
struct MessageDeltaBody {
    stop_reason: Option<String>,
    #[serde(default)]
    usage: Option<WireUsage>,
}

pub(crate) async fn generate(
    client: &VertexClient,
    request: &GenerateRequest,
) -> Result<GenerateResponse, Error> {
    let url = endpoint(&client.auth, &request.model, false)?;
    let req = client
        .authorize(client.http.post(&url).json(&to_wire(request)))
        .await?;

    let resp = client.send(req).await?;
    let wire: Response = resp
        .json()
        .await
        .map_err(|e| Error::provider("vertex-ai", format!("Parse failed: {e}")))?;

    Ok(from_wire(wire))
}

pub(crate) async fn stream_generate(
    client: &VertexClient,
    request: &GenerateRequest,
) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, Error>> + Send>>, Error> {
    let url = endpoint(&client.auth, &request.model, true)?;
    let mut wire = to_wire(request);
    wire.stream = Some(true);
    let req = client.authorize(client.http.post(&url).json(&wire)).await?;

    let resp = client.send(req).await?;
    Ok(parse_sse(resp))
}

fn endpoint(auth: &ResolvedAuth, model: &str, stream: bool) -> Result<String, Error> {
    let method = if stream {
        "streamRawPredict"
    } else {
        "rawPredict"
    };
    match auth {
        ResolvedAuth::ApiKey { .. } => Err(Error::config(
            "Anthropic models on Vertex AI require service account auth (VERTEX_PROJECT_ID + service_account_key)",
        )),
        ResolvedAuth::ServiceAccount {
            project_id, region, ..
        } => {
            let host = regional_host(region);
            Ok(format!(
                "https://{host}/v1/projects/{project_id}/locations/{region}/publishers/anthropic/models/{model}:{method}"
            ))
        }
    }
}

fn to_wire(req: &GenerateRequest) -> Request {
    if req.web_fetch {
        tracing::warn!(
            "web_fetch is unavailable for Anthropic on Vertex AI; the model cannot fetch pages"
        );
    }

    let mut messages: Vec<WireMessage> = req
        .messages
        .iter()
        .map(|m| WireMessage {
            role: match m.role {
                Role::User => "user".into(),
                Role::Assistant => "assistant".into(),
            },
            content: match m.content.as_slice() {
                [ContentPart::Text(text)] => WireContent::Text(text.clone()),
                parts => WireContent::Blocks(to_blocks(parts)),
            },
        })
        .collect();

    let tools = if req.tools.is_empty() {
        None
    } else {
        Some(
            req.tools
                .iter()
                .map(|t| WireToolDef {
                    name: t.name.clone(),
                    description: t.description.clone(),
                    input_schema: t.parameters.clone(),
                    cache_control: None,
                })
                .collect(),
        )
    };

    let mut tools: Option<Vec<WireToolDef>> = tools;
    let system = match req.system.clone() {
        Some(text) if req.prompt_cache => Some(WireSystem::Blocks(vec![SystemBlock {
            r#type: "text",
            text,
            cache_control: EPHEMERAL,
        }])),
        other => {
            if req.prompt_cache
                && let Some(last) = tools.as_mut().and_then(|t| t.last_mut())
            {
                last.cache_control = Some(EPHEMERAL);
            }
            other.map(WireSystem::Text)
        }
    };
    if req.prompt_cache {
        mark_last_message(&mut messages);
    }

    Request {
        anthropic_version: "vertex-2023-10-16".into(),
        max_tokens: req.max_tokens.unwrap_or(4096),
        messages,
        system,
        tools,
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: req.top_k,
        stream: None,
    }
}

fn mark_last_message(messages: &mut [WireMessage]) {
    let Some(last) = messages.last_mut() else {
        return;
    };
    if let WireContent::Text(text) = &mut last.content {
        last.content = WireContent::Blocks(vec![ContentBlock::Text {
            text: std::mem::take(text),
            cache_control: None,
        }]);
    }
    if let WireContent::Blocks(blocks) = &mut last.content
        && let Some(block) = blocks.last_mut()
    {
        block.mark_cached();
    }
}

fn to_blocks(parts: &[ContentPart]) -> Vec<ContentBlock> {
    parts
        .iter()
        .map(|p| match p {
            ContentPart::Text(text) => ContentBlock::Text {
                text: text.clone(),
                cache_control: None,
            },
            ContentPart::ToolCall {
                id,
                name,
                arguments,
                ..
            } => ContentBlock::ToolUse {
                id: id.clone(),
                name: name.clone(),
                input: arguments.clone(),
                cache_control: None,
            },
            ContentPart::ToolResult {
                tool_call_id,
                content,
                ..
            } => ContentBlock::ToolResult {
                tool_use_id: tool_call_id.clone(),
                content: value_to_tool_result_string(content),
                cache_control: None,
            },
        })
        .collect()
}

/// Anthropic's `tool_result.content` is a raw string. If our canonical content is
/// a JSON string, unwrap it (`Value::String("hi")` → `hi`). Any other JSON value
/// is serialized as-is so the model sees valid JSON.
fn value_to_tool_result_string(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

fn from_wire(resp: Response) -> GenerateResponse {
    let content = resp
        .content
        .into_iter()
        .map(|block| match block {
            ContentBlock::Text { text, .. } => ContentPart::Text(text),
            ContentBlock::ToolUse {
                id, name, input, ..
            } => ContentPart::ToolCall {
                id,
                name,
                arguments: input,
                thought_signature: None,
            },
            ContentBlock::ToolResult { .. } => ContentPart::Text("[tool_result]".into()),
        })
        .collect();

    GenerateResponse {
        content,
        stop_reason: resp.stop_reason,
        usage: resp.usage.map(Usage::from),
    }
}

fn parse_sse(
    resp: reqwest::Response,
) -> Pin<Box<dyn Stream<Item = Result<StreamChunk, Error>> + Send>> {
    let stream = async_stream::stream! {
        let mut byte_stream = futures::StreamExt::fuse(resp.bytes_stream());
        let mut buffer: Vec<u8> = Vec::new();

        let mut current_tool_id = String::new();
        let mut current_tool_name = String::new();
        let mut current_tool_json = String::new();

        let mut usage: Option<Usage> = None;
        let mut done_sent = false;

        while let Some(chunk) = futures::StreamExt::next(&mut byte_stream).await {
            let bytes = chunk.map_err(|e| {
                Error::provider("vertex-ai", format!("Stream read error: {e}"))
            })?;
            buffer.extend_from_slice(&bytes);

            while let Some((pos, sep_len)) = find_frame_boundary(&buffer) {
                let frame = buffer.drain(..pos + sep_len).collect::<Vec<u8>>();
                let Ok(event_block) = std::str::from_utf8(&frame[..frame.len() - sep_len]) else {
                    tracing::debug!("Anthropic SSE frame was not valid UTF-8");
                    continue;
                };

                let mut data_line = None;
                for line in event_block.lines() {
                    if let Some(d) = line.strip_prefix("data: ") {
                        data_line = Some(d);
                    }
                }

                let Some(data) = data_line else { continue };

                match serde_json::from_str::<StreamEvent>(data) {
                    Ok(event) => match event {
                        StreamEvent::MessageStart { message } => {
                            if let Some(m) = message
                                && let Some(u) = m.usage
                            {
                                usage = Some(merge_usage(usage.take(), u.into()));
                            }
                        }
                        StreamEvent::ContentBlockStart { content_block } => {
                            if let Some(ContentBlockInfo::ToolUse { id, name }) = content_block {
                                current_tool_id = id;
                                current_tool_name = name;
                                current_tool_json.clear();
                            }
                        }
                        StreamEvent::ContentBlockDelta {
                            delta: Delta::TextDelta { text },
                        } => {
                            yield Ok(StreamChunk::Text(text));
                        }
                        StreamEvent::ContentBlockDelta {
                            delta: Delta::InputJsonDelta { partial_json },
                        } => {
                            current_tool_json.push_str(&partial_json);
                        }
                        StreamEvent::ContentBlockStop {} => {
                            if !current_tool_name.is_empty() {
                                let arguments = if current_tool_json.is_empty() {
                                    serde_json::Value::Object(serde_json::Map::new())
                                } else {
                                    serde_json::from_str(&current_tool_json)
                                        .unwrap_or(serde_json::Value::Null)
                                };
                                yield Ok(StreamChunk::ToolCall {
                                    id: std::mem::take(&mut current_tool_id),
                                    name: std::mem::take(&mut current_tool_name),
                                    arguments,
                                    thought_signature: None,
                                });
                                current_tool_json.clear();
                            }
                        }
                        StreamEvent::MessageDelta { delta } => {
                            if let Some(u) = delta.usage {
                                usage = Some(merge_usage(usage.take(), u.into()));
                            }
                            if !done_sent && let Some(reason) = delta.stop_reason {
                                yield Ok(StreamChunk::Done {
                                    finish_reason: reason,
                                    usage: usage.clone(),
                                });
                                done_sent = true;
                            }
                        }
                        StreamEvent::MessageStop => {
                            if !done_sent {
                                yield Ok(StreamChunk::Done {
                                    finish_reason: "end_turn".into(),
                                    usage: usage.clone(),
                                });
                                done_sent = true;
                            }
                        }
                        StreamEvent::Ping => {}
                    },
                    Err(e) => {
                        tracing::debug!("Failed to parse Anthropic SSE chunk: {e}, data: {data}");
                    }
                }
            }
        }
    };

    Box::pin(stream)
}

fn find_frame_boundary(buf: &[u8]) -> Option<(usize, usize)> {
    let crlf = buf.windows(4).position(|w| w == b"\r\n\r\n");
    let lf = buf.windows(2).position(|w| w == b"\n\n");
    match (crlf, lf) {
        (Some(c), Some(l)) => {
            if c <= l {
                Some((c, 4))
            } else {
                Some((l, 2))
            }
        }
        (Some(c), None) => Some((c, 4)),
        (None, Some(l)) => Some((l, 2)),
        (None, None) => None,
    }
}

#[cfg(test)]
mod tests {
    fn sse_response(frames: &[&str]) -> reqwest::Response {
        let body: String = frames.iter().map(|f| format!("data: {f}\n\n")).collect();
        reqwest::Response::from(http::Response::new(body))
    }

    async fn done_usage(frames: &[&str]) -> Option<Usage> {
        let mut stream = parse_sse(sse_response(frames));
        let mut seen = Vec::new();
        while let Some(chunk) = futures::StreamExt::next(&mut stream).await {
            if let Ok(StreamChunk::Done { usage, .. }) = chunk {
                seen.push(usage);
            }
        }
        assert_eq!(seen.len(), 1, "exactly one Done per stream");
        seen.remove(0)
    }

    #[tokio::test]
    async fn stream_keeps_the_final_output_count_from_message_delta() {
        let usage = done_usage(&[
            r#"{"type":"message_start","message":{"usage":{"input_tokens":10,"cache_creation_input_tokens":6,"cache_read_input_tokens":4,"output_tokens":1}}}"#,
            r#"{"type":"content_block_delta","delta":{"type":"text_delta","text":"hi"}}"#,
            r#"{"type":"message_delta","delta":{"stop_reason":"end_turn","usage":{"output_tokens":42}}}"#,
            r#"{"type":"message_stop"}"#,
        ])
        .await
        .expect("Done carries usage");

        assert_eq!(usage.output_tokens, Some(42));
        assert_eq!(usage.input_tokens, Some(20));
        assert_eq!(usage.cached_input_tokens, Some(4));
        assert_eq!(usage.total_tokens, Some(62));
    }

    use super::*;

    fn cached_request() -> GenerateRequest {
        GenerateRequest::new(
            "m",
            vec![
                crate::types::Message::user("red shoes"),
                crate::types::Message::assistant("Here you go."),
                crate::types::Message::user("cheaper ones?"),
            ],
        )
        .with_tools(vec![crate::types::ToolDefinition::new("search", "Search")])
        .with_prompt_cache(true)
    }

    #[test]
    fn prompt_cache_marks_the_system_prompt_and_the_last_message() {
        let req = cached_request().with_system("You are a shop assistant.");
        let json = serde_json::to_value(to_wire(&req)).unwrap();
        assert_eq!(
            json["system"],
            serde_json::json!([{
                "type": "text",
                "text": "You are a shop assistant.",
                "cache_control": {"type": "ephemeral"},
            }])
        );
        assert!(json["tools"][0].get("cache_control").is_none());
        assert_eq!(json["messages"][0]["content"], "red shoes");
        assert_eq!(
            json["messages"][2]["content"],
            serde_json::json!([{
                "type": "text",
                "text": "cheaper ones?",
                "cache_control": {"type": "ephemeral"},
            }])
        );
    }

    #[test]
    fn prompt_cache_marks_the_tools_when_there_is_no_system_prompt() {
        let json = serde_json::to_value(to_wire(&cached_request())).unwrap();
        assert!(json.get("system").is_none());
        assert_eq!(
            json["tools"][0]["cache_control"],
            serde_json::json!({"type": "ephemeral"})
        );
    }

    #[test]
    fn no_cache_control_without_prompt_cache() {
        let req = cached_request().with_system("sys").with_prompt_cache(false);
        let json = serde_json::to_value(to_wire(&req)).unwrap();
        assert_eq!(json["system"], "sys");
        assert!(!json.to_string().contains("cache_control"));
    }

    #[test]
    fn web_fetch_is_dropped_rather_than_sent_to_anthropic() {
        let req = GenerateRequest::new("m", vec![]).with_web_fetch(true);
        let json = serde_json::to_value(to_wire(&req)).unwrap();
        assert!(json.get("tools").is_none());
    }

    #[test]
    fn value_to_tool_result_string_unwraps_json_string() {
        let v = serde_json::Value::String("hello".into());
        assert_eq!(value_to_tool_result_string(&v), "hello");
    }

    #[test]
    fn value_to_tool_result_string_serializes_object() {
        let v = serde_json::json!({"count": 3});
        let out = value_to_tool_result_string(&v);
        assert!(out.contains("\"count\""));
        assert!(out.contains("3"));
    }

    #[test]
    fn to_blocks_tool_result_sends_raw_string_to_anthropic() {
        let parts = vec![ContentPart::ToolResult {
            tool_call_id: "t1".into(),
            name: "search".into(),
            content: serde_json::Value::String("raw text".into()),
        }];
        let blocks = to_blocks(&parts);
        match &blocks[0] {
            ContentBlock::ToolResult { content, .. } => {
                assert_eq!(content, "raw text");
            }
            _ => panic!("expected tool_result block"),
        }
    }

    #[test]
    fn message_delta_usage_parses_without_input_tokens() {
        let event: StreamEvent = serde_json::from_str(
            r#"{"type":"message_delta","delta":{"stop_reason":"end_turn","usage":{"output_tokens":5}}}"#,
        )
        .unwrap();
        let StreamEvent::MessageDelta { delta } = event else {
            panic!("expected message_delta")
        };
        let u: Usage = delta.usage.unwrap().into();
        assert_eq!(u.output_tokens, Some(5));
        assert_eq!(u.input_tokens, None);
    }

    #[test]
    fn cache_buckets_fold_into_input_and_report_cache_reads() {
        let u: Usage = serde_json::from_str::<WireUsage>(
            r#"{"input_tokens":10,"output_tokens":5,"cache_creation_input_tokens":7,"cache_read_input_tokens":3}"#,
        )
        .unwrap()
        .into();
        assert_eq!(u.input_tokens, Some(20));
        assert_eq!(u.cached_input_tokens, Some(3));
        assert_eq!(u.total_tokens, Some(25));
    }

    #[test]
    fn merge_usage_keeps_input_from_message_start() {
        let start = Usage {
            input_tokens: Some(10),
            output_tokens: Some(1),
            ..Usage::default()
        };
        let delta = Usage {
            output_tokens: Some(42),
            ..Usage::default()
        };
        let merged = merge_usage(Some(start), delta);
        assert_eq!(merged.input_tokens, Some(10));
        assert_eq!(merged.output_tokens, Some(42));
        assert_eq!(merged.total_tokens, Some(52));
    }

    #[test]
    fn find_frame_boundary_lf() {
        let b = b"event: x\ndata: hi\n\nnext";
        assert_eq!(find_frame_boundary(b), Some((17, 2)));
    }

    #[test]
    fn find_frame_boundary_crlf() {
        let b = b"event: x\r\ndata: hi\r\n\r\nnext";
        assert_eq!(find_frame_boundary(b), Some((18, 4)));
    }
}
