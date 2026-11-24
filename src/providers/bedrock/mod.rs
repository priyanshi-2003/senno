#![cfg_attr(
    not(test),
    deny(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing,
        clippy::unreachable,
        clippy::todo,
        clippy::unimplemented,
        clippy::absolute_paths
    )
)]

mod client;
mod config;
mod converse;
mod document;
mod embed;
mod error;
mod stream;
mod types;

pub use types::{BedrockClient, BedrockConfig, BedrockGuardrail};

use crate::error::Error;

pub async fn get_bedrock_client(config: Option<BedrockConfig>) -> Result<BedrockClient, Error> {
    let config = config.unwrap_or_default();
    let sdk = config::resolve(&config).await?;
    Ok(BedrockClient::new(sdk, config.guardrail))
}
