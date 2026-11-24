//! Provider implementations of [`crate::LlmProvider`], each behind its own
//! feature flag.

#[cfg(feature = "vertex")]
pub mod vertex;

#[cfg(feature = "bedrock")]
pub mod bedrock;

#[cfg(any(feature = "vertex", feature = "bedrock"))]
use serde_json::Value;

#[cfg(any(feature = "vertex", feature = "bedrock"))]
pub(crate) fn render_tool_content(content: &Value) -> String {
    match content {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}
