use aws_sdk_bedrockruntime::types as bt;
use aws_smithy_types::Document;
use tracing::instrument;

use crate::error::Error;
use crate::provider::LlmStream;
use crate::types::{ContentPart, GenerateRequest, GenerateResponse, Message, Role, Usage};

use super::document::{to_document, to_value};
use super::error::{PROVIDER, from_sdk};
use super::stream;
use super::types::BedrockClient;

const LEADING_ASSISTANT_PREFIX: &str = "Summary of the earlier conversation:\n";

pub(super) struct ConverseParts {
    pub system: Vec<bt::SystemContentBlock>,
    pub messages: Vec<bt::Message>,
    pub inference: Option<bt::InferenceConfiguration>,
    pub tools: Option<bt::ToolConfiguration>,
    pub additional_fields: Option<Document>,
}

pub(super) fn to_converse(request: &GenerateRequest) -> Result<ConverseParts, Error> {
    if request.web_fetch {
        tracing::warn!("web_fetch is unavailable on Bedrock; the model cannot fetch pages");
    }
    if request.thinking_budget.is_some() {
        tracing::warn!("thinking_budget is not supported on Bedrock yet and is ignored");
    }

    let mut system = request
        .system
        .as_deref()
        .filter(|s| !s.is_empty())
        .map(|s| vec![bt::SystemContentBlock::Text(s.to_string())])
        .unwrap_or_default();
    let cache_tools = request.prompt_cache && system.is_empty();
    if request.prompt_cache && !system.is_empty() {
        system.push(bt::SystemContentBlock::CachePoint(build_cache_point()?));
    }

    Ok(ConverseParts {
        system,
        messages: to_messages(&request.messages, request.prompt_cache)?,
        inference: build_inference_config(request),
        tools: build_tool_config(request, cache_tools)?,
        additional_fields: build_additional_fields(request),
    })
}

fn build_cache_point() -> Result<bt::CachePointBlock, Error> {
    bt::CachePointBlock::builder()
        .r#type(bt::CachePointType::Default)
        .build()
        .map_err(|e| Error::internal(format!("bedrock cache point build failed: {e}")))
}

fn to_messages(messages: &[Message], prompt_cache: bool) -> Result<Vec<bt::Message>, Error> {
    let mut merged: Vec<(Role, Vec<bt::ContentBlock>)> = Vec::new();
    for message in messages {
        let mut role = message.role;
        let mut blocks = to_blocks(&message.content)?;
        if blocks.is_empty() {
            continue;
        }
        if merged.is_empty() && role == Role::Assistant {
            let text_only = blocks
                .iter()
                .all(|b| matches!(b, bt::ContentBlock::Text(_)));
            if !text_only {
                return Err(Error::config(
                    "Converse needs the conversation to start with a user message, not an assistant tool call",
                ));
            }
            role = Role::User;
            if let Some(bt::ContentBlock::Text(text)) = blocks.first_mut() {
                *text = format!("{LEADING_ASSISTANT_PREFIX}{text}");
            }
        }
        match merged.last_mut() {
            Some((last_role, last_blocks)) if *last_role == role => last_blocks.extend(blocks),
            _ => merged.push((role, blocks)),
        }
    }

    if prompt_cache && let Some((_, blocks)) = merged.last_mut() {
        blocks.push(bt::ContentBlock::CachePoint(build_cache_point()?));
    }

    merged
        .into_iter()
        .map(|(role, blocks)| {
            bt::Message::builder()
                .role(match role {
                    Role::User => bt::ConversationRole::User,
                    Role::Assistant => bt::ConversationRole::Assistant,
                })
                .set_content(Some(blocks))
                .build()
                .map_err(|e| Error::internal(format!("bedrock message build failed: {e}")))
        })
        .collect()
}

fn to_blocks(parts: &[ContentPart]) -> Result<Vec<bt::ContentBlock>, Error> {
    let mut blocks = Vec::with_capacity(parts.len());
    for part in parts {
        match part {
            ContentPart::Text(text) if text.is_empty() => {}
            ContentPart::Text(text) => blocks.push(bt::ContentBlock::Text(text.clone())),
            ContentPart::ToolCall {
                id,
                name,
                arguments,
                ..
            } => blocks.push(bt::ContentBlock::ToolUse(
                bt::ToolUseBlock::builder()
                    .tool_use_id(id)
                    .name(name)
                    .input(to_document(arguments))
                    .build()
                    .map_err(|e| Error::internal(format!("bedrock tool_use build failed: {e}")))?,
            )),
            ContentPart::ToolResult {
                tool_call_id,
                content,
                ..
            } => blocks.push(bt::ContentBlock::ToolResult(
                bt::ToolResultBlock::builder()
                    .tool_use_id(tool_call_id)
                    .content(bt::ToolResultContentBlock::Text(render_tool_result(
                        content,
                    )))
                    .build()
                    .map_err(|e| {
                        Error::internal(format!("bedrock tool_result build failed: {e}"))
                    })?,
            )),
        }
    }
    Ok(blocks)
}

fn render_tool_result(content: &serde_json::Value) -> String {
    let text = match content {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    };
    if text.is_empty() {
        "(empty result)".to_string()
    } else {
        text
    }
}

fn build_inference_config(request: &GenerateRequest) -> Option<bt::InferenceConfiguration> {
    if request.max_tokens.is_none() && request.temperature.is_none() && request.top_p.is_none() {
        return None;
    }
    Some(
        bt::InferenceConfiguration::builder()
            .set_max_tokens(
                request
                    .max_tokens
                    .map(|v| i32::try_from(v).unwrap_or(i32::MAX)),
            )
            .set_temperature(request.temperature)
            .set_top_p(request.top_p)
            .build(),
    )
}

fn build_tool_config(
    request: &GenerateRequest,
    cache: bool,
) -> Result<Option<bt::ToolConfiguration>, Error> {
    if request.tools.is_empty() {
        return Ok(None);
    }
    let mut tools = request
        .tools
        .iter()
        .map(|tool| {
            let schema = serde_json::to_value(&tool.parameters).map_err(|e| {
                Error::internal(format!("tool {} schema did not serialize: {e}", tool.name))
            })?;
            let mut spec = bt::ToolSpecification::builder()
                .name(&tool.name)
                .input_schema(bt::ToolInputSchema::Json(to_document(&schema)));
            if !tool.description.is_empty() {
                spec = spec.description(&tool.description);
            }
            spec.build()
                .map(bt::Tool::ToolSpec)
                .map_err(|e| Error::internal(format!("bedrock tool spec build failed: {e}")))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if cache {
        tools.push(bt::Tool::CachePoint(build_cache_point()?));
    }

    bt::ToolConfiguration::builder()
        .set_tools(Some(tools))
        .build()
        .map(Some)
        .map_err(|e| Error::internal(format!("bedrock tool config build failed: {e}")))
}

fn build_additional_fields(request: &GenerateRequest) -> Option<Document> {
    let top_k = request.top_k?;
    Some(to_document(&serde_json::json!({ "top_k": top_k })))
}

fn to_trace(enabled: bool) -> bt::GuardrailTrace {
    if enabled {
        bt::GuardrailTrace::Enabled
    } else {
        bt::GuardrailTrace::Disabled
    }
}

pub(super) fn build_guardrail(client: &BedrockClient) -> Option<bt::GuardrailConfiguration> {
    client.guardrail.as_ref().map(|g| {
        bt::GuardrailConfiguration::builder()
            .guardrail_identifier(&g.id)
            .guardrail_version(&g.version)
            .trace(to_trace(g.trace))
            .build()
    })
}

pub(super) fn build_stream_guardrail(
    client: &BedrockClient,
) -> Option<bt::GuardrailStreamConfiguration> {
    client.guardrail.as_ref().map(|g| {
        bt::GuardrailStreamConfiguration::builder()
            .guardrail_identifier(&g.id)
            .guardrail_version(&g.version)
            .trace(to_trace(g.trace))
            .build()
    })
}

pub(super) fn to_usage(usage: &bt::TokenUsage) -> Usage {
    let to_u32 = |v: i32| u32::try_from(v).unwrap_or(0);
    let cache_read = usage.cache_read_input_tokens().map(to_u32);
    let cache_write = usage.cache_write_input_tokens().map(to_u32);
    let input = to_u32(usage.input_tokens())
        .saturating_add(cache_read.unwrap_or(0))
        .saturating_add(cache_write.unwrap_or(0));
    let output = to_u32(usage.output_tokens());
    Usage {
        input_tokens: Some(input),
        cached_input_tokens: cache_read,
        cache_write_input_tokens: cache_write,
        output_tokens: Some(output),
        reasoning_tokens: None,
        total_tokens: Some(to_u32(usage.total_tokens()).max(input.saturating_add(output))),
    }
}

pub(super) fn from_content(blocks: &[bt::ContentBlock]) -> Vec<ContentPart> {
    blocks
        .iter()
        .filter_map(|block| match block {
            bt::ContentBlock::Text(text) => Some(ContentPart::Text(text.clone())),
            bt::ContentBlock::ToolUse(tool) => Some(ContentPart::ToolCall {
                id: tool.tool_use_id().to_string(),
                name: tool.name().to_string(),
                arguments: to_value(tool.input()),
                thought_signature: None,
            }),
            _ => None,
        })
        .collect()
}

#[instrument(skip_all, fields(model = %request.model))]
pub(super) async fn generate(
    client: &BedrockClient,
    request: &GenerateRequest,
) -> Result<GenerateResponse, Error> {
    let parts = to_converse(request)?;
    let output = client
        .inner
        .converse()
        .model_id(&request.model)
        .set_system(Some(parts.system).filter(|s| !s.is_empty()))
        .set_messages(Some(parts.messages))
        .set_inference_config(parts.inference)
        .set_tool_config(parts.tools)
        .set_additional_model_request_fields(parts.additional_fields)
        .set_guardrail_config(build_guardrail(client))
        .send()
        .await
        .map_err(|e| from_sdk("Converse", e))?;

    let content = match output.output() {
        Some(bt::ConverseOutput::Message(message)) => from_content(message.content()),
        _ => {
            return Err(Error::provider_permanent(
                PROVIDER,
                "Converse returned no message",
            ));
        }
    };

    Ok(GenerateResponse {
        content,
        stop_reason: Some(output.stop_reason().as_str().to_string()),
        usage: output.usage().map(to_usage),
    })
}

#[instrument(skip_all, fields(model = %request.model))]
pub(super) async fn stream_generate(
    client: &BedrockClient,
    request: &GenerateRequest,
) -> Result<LlmStream, Error> {
    let parts = to_converse(request)?;
    let output = client
        .inner
        .converse_stream()
        .model_id(&request.model)
        .set_system(Some(parts.system).filter(|s| !s.is_empty()))
        .set_messages(Some(parts.messages))
        .set_inference_config(parts.inference)
        .set_tool_config(parts.tools)
        .set_additional_model_request_fields(parts.additional_fields)
        .set_guardrail_config(build_stream_guardrail(client))
        .send()
        .await
        .map_err(|e| from_sdk("ConverseStream", e))?;

    Ok(stream::into_chunks(output.stream))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Message, ParameterSchema, ToolDefinition};
    use serde_json::json;

    fn roles(messages: &[bt::Message]) -> Vec<bt::ConversationRole> {
        messages.iter().map(|m| m.role().clone()).collect()
    }

    #[test]
    fn consecutive_messages_of_one_role_are_merged() {
        let history = vec![
            Message::user("find shoes"),
            Message::tool_call("t1", "search", json!({"q": "shoes"})),
            Message::tool_result("t1", "search", json!("3 results")),
            Message::assistant("Here are three."),
            Message::tool_call("t2", "recommend", json!({"ids": ["a"]})),
        ];
        let messages = to_messages(&history, false).unwrap();
        assert_eq!(
            roles(&messages),
            vec![
                bt::ConversationRole::User,
                bt::ConversationRole::Assistant,
                bt::ConversationRole::User,
                bt::ConversationRole::Assistant,
            ]
        );
        let last = messages.last().unwrap().content();
        assert!(matches!(&last[0], bt::ContentBlock::Text(t) if t == "Here are three."));
        assert!(matches!(&last[1], bt::ContentBlock::ToolUse(t) if t.name() == "recommend"));
    }

    #[test]
    fn leading_assistant_summary_is_sent_as_user() {
        let history = vec![
            Message::assistant("They wanted shoes."),
            Message::user("cheaper?"),
        ];
        let messages = to_messages(&history, false).unwrap();
        assert_eq!(roles(&messages), vec![bt::ConversationRole::User]);
        let blocks = messages[0].content();
        assert!(matches!(&blocks[0], bt::ContentBlock::Text(t)
            if t == "Summary of the earlier conversation:\nThey wanted shoes."));
        assert!(matches!(&blocks[1], bt::ContentBlock::Text(t) if t == "cheaper?"));
    }

    #[test]
    fn empty_text_is_dropped_and_empty_tool_result_gets_placeholder() {
        let history = vec![
            Message::user(""),
            Message::user("hi"),
            Message::tool_result("t1", "noop", json!("")),
        ];
        let messages = to_messages(&history, false).unwrap();
        let blocks = messages[0].content();
        assert_eq!(blocks.len(), 2);
        match &blocks[1] {
            bt::ContentBlock::ToolResult(r) => {
                assert!(
                    matches!(&r.content()[0], bt::ToolResultContentBlock::Text(t) if t == "(empty result)")
                )
            }
            other => panic!("expected tool result, got {other:?}"),
        }
    }

    #[test]
    fn tool_arguments_and_json_results_survive() {
        let history = vec![
            Message::user("find shoes"),
            Message::tool_call("t1", "search", json!({"q": "shoes", "limit": 5})),
            Message::tool_result("t1", "search", json!({"count": 2})),
        ];
        let messages = to_messages(&history, false).unwrap();
        match &messages[1].content()[0] {
            bt::ContentBlock::ToolUse(t) => {
                assert_eq!(t.tool_use_id(), "t1");
                assert_eq!(to_value(t.input()), json!({"q": "shoes", "limit": 5}));
            }
            other => panic!("expected tool use, got {other:?}"),
        }
        match &messages[2].content()[0] {
            bt::ContentBlock::ToolResult(r) => {
                assert!(
                    matches!(&r.content()[0], bt::ToolResultContentBlock::Text(t) if t == r#"{"count":2}"#)
                )
            }
            other => panic!("expected tool result, got {other:?}"),
        }
    }

    #[test]
    fn leading_tool_call_is_rejected() {
        let history = vec![
            Message::tool_call("t1", "search", json!({})),
            Message::tool_result("t1", "search", json!({})),
        ];
        let err = to_messages(&history, false).unwrap_err();
        assert!(!err.is_retryable(), "{err}");
    }

    #[test]
    fn empty_first_message_does_not_hide_a_leading_summary() {
        let history = vec![
            Message::assistant(""),
            Message::assistant("earlier summary"),
            Message::user("next"),
        ];
        let messages = to_messages(&history, false).unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role(), &bt::ConversationRole::User);
    }

    #[test]
    fn tools_carry_their_json_schema() {
        let tool = ToolDefinition::new("search", "Search the catalog").with_parameters(
            ParameterSchema::object()
                .with_property("q", ParameterSchema::string("query"))
                .with_required(["q"]),
        );
        let request = GenerateRequest::new("m", vec![Message::user("hi")]).with_tools(vec![tool]);
        let config = build_tool_config(&request, false).unwrap().unwrap();
        match &config.tools()[0] {
            bt::Tool::ToolSpec(spec) => {
                assert_eq!(spec.name(), "search");
                assert_eq!(spec.description(), Some("Search the catalog"));
                let Some(bt::ToolInputSchema::Json(schema)) = spec.input_schema() else {
                    panic!("schema missing");
                };
                assert_eq!(
                    to_value(schema),
                    json!({
                        "type": "object",
                        "properties": {"q": {"type": "string", "description": "query"}},
                        "required": ["q"],
                    })
                );
            }
            other => panic!("expected tool spec, got {other:?}"),
        }
    }

    fn cached_request(system: Option<&str>) -> GenerateRequest {
        let mut request = GenerateRequest::new(
            "m",
            vec![
                Message::user("red shoes"),
                Message::assistant("Here you go."),
                Message::user("cheaper ones?"),
            ],
        )
        .with_tools(vec![ToolDefinition::new("search", "Search the catalog")])
        .with_prompt_cache(true);
        if let Some(system) = system {
            request = request.with_system(system);
        }
        request
    }

    fn ends_with_cache_point(blocks: &[bt::ContentBlock]) -> bool {
        matches!(blocks.last(), Some(bt::ContentBlock::CachePoint(_)))
    }

    #[test]
    fn prompt_cache_marks_the_system_prompt_and_the_last_message() {
        let parts = to_converse(&cached_request(Some("You are a shop assistant."))).unwrap();
        assert!(matches!(
            parts.system.as_slice(),
            [
                bt::SystemContentBlock::Text(_),
                bt::SystemContentBlock::CachePoint(_)
            ]
        ));
        let tools = parts.tools.unwrap();
        assert!(
            !tools
                .tools()
                .iter()
                .any(|t| matches!(t, bt::Tool::CachePoint(_)))
        );
        let last = parts.messages.last().unwrap();
        assert!(ends_with_cache_point(last.content()));
        assert!(!ends_with_cache_point(parts.messages[0].content()));
    }

    #[test]
    fn prompt_cache_marks_the_tools_when_there_is_no_system_prompt() {
        let parts = to_converse(&cached_request(None)).unwrap();
        assert!(parts.system.is_empty());
        let tools = parts.tools.unwrap();
        assert!(matches!(
            tools.tools().last(),
            Some(bt::Tool::CachePoint(_))
        ));
        assert!(ends_with_cache_point(
            parts.messages.last().unwrap().content()
        ));
    }

    #[test]
    fn no_cache_points_without_prompt_cache() {
        let request = cached_request(Some("sys")).with_prompt_cache(false);
        let parts = to_converse(&request).unwrap();
        assert_eq!(parts.system.len(), 1);
        assert!(
            !parts
                .tools
                .unwrap()
                .tools()
                .iter()
                .any(|t| matches!(t, bt::Tool::CachePoint(_)))
        );
        assert!(
            parts
                .messages
                .iter()
                .all(|m| !ends_with_cache_point(m.content()))
        );
    }

    #[test]
    fn inference_settings_map_and_top_k_rides_additional_fields() {
        let request = GenerateRequest::new("m", vec![Message::user("hi")])
            .with_max_tokens(512)
            .with_temperature(0.2)
            .with_top_k(40);
        let inference = build_inference_config(&request).unwrap();
        assert_eq!(inference.max_tokens(), Some(512));
        assert_eq!(inference.temperature(), Some(0.2));
        assert_eq!(
            to_value(&build_additional_fields(&request).unwrap()),
            json!({"top_k": 40})
        );
        assert!(build_inference_config(&GenerateRequest::new("m", vec![])).is_none());
    }

    #[test]
    fn usage_counts_cache_reads_and_writes_as_input() {
        let usage = bt::TokenUsage::builder()
            .input_tokens(100)
            .output_tokens(20)
            .total_tokens(170)
            .cache_read_input_tokens(40)
            .cache_write_input_tokens(10)
            .build()
            .unwrap();
        let u = to_usage(&usage);
        assert_eq!(u.input_tokens, Some(150));
        assert_eq!(u.cached_input_tokens, Some(40));
        assert_eq!(u.cache_write_input_tokens, Some(10));
        assert_eq!(u.output_tokens, Some(20));
        assert_eq!(u.total_tokens, Some(170));
    }
}
