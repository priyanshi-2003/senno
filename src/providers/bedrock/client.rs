use std::future::Future;
use std::pin::Pin;

use crate::error::Error;
use crate::provider::{EmbeddingProvider, LlmProvider, LlmStream};
use crate::types::{EmbedRequest, EmbedResponse, GenerateRequest, GenerateResponse};

use super::types::BedrockClient;
use super::{converse, embed};

impl LlmProvider for BedrockClient {
    fn generate<'a>(
        &'a self,
        request: &'a GenerateRequest,
    ) -> Pin<Box<dyn Future<Output = Result<GenerateResponse, Error>> + Send + 'a>> {
        Box::pin(converse::generate(self, request))
    }

    fn stream_generate<'a>(
        &'a self,
        request: &'a GenerateRequest,
    ) -> Pin<Box<dyn Future<Output = Result<LlmStream, Error>> + Send + 'a>> {
        Box::pin(converse::stream_generate(self, request))
    }
}

impl EmbeddingProvider for BedrockClient {
    fn embed<'a>(
        &'a self,
        request: &'a EmbedRequest,
    ) -> Pin<Box<dyn Future<Output = Result<EmbedResponse, Error>> + Send + 'a>> {
        Box::pin(embed::embed(self, request))
    }
}
