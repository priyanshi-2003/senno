#![cfg(feature = "bedrock")]

mod common;

use std::time::{Duration, Instant};

use aws_config::{BehaviorVersion, Region, SdkConfig};
use aws_sdk_bedrockruntime::config::{Credentials, SharedCredentialsProvider};
use common::{Reply, StubServer};
use senno::providers::bedrock::{
    BedrockClient, BedrockConfig, BedrockGuardrail, get_bedrock_client,
};
use senno::{
    EmbedImage, EmbedRequest, EmbedTaskType, EmbeddingProvider, GenerateRequest, LlmProvider,
    Message, ParameterSchema, ToolDefinition,
};
use serde_json::json;

fn sdk_config() -> SdkConfig {
    SdkConfig::builder()
        .behavior_version(BehaviorVersion::latest())
        .region(Region::new("eu-central-1"))
        .credentials_provider(SharedCredentialsProvider::new(Credentials::new(
            "AKIDTEST", "secret", None, None, "test",
        )))
        .build()
}

async fn client(server: &StubServer, config: BedrockConfig) -> BedrockClient {
    get_bedrock_client(Some(
        config
            .with_sdk_config(sdk_config())
            .with_endpoint_url(&server.base_url)
            .with_max_attempts(1),
    ))
    .await
    .expect("bedrock client")
}

#[tokio::test]
async fn converse_sends_history_tools_and_guardrail_and_parses_tool_call() {
    let server = StubServer::start(|_| {
        Reply::ok(
            json!({
                "output": {"message": {"role": "assistant", "content": [
                    {"text": "Searching."},
                    {"toolUse": {"toolUseId": "tu-1", "name": "search_catalog",
                                 "input": {"catalog": {"query": "red shoes"}}}}
                ]}},
                "stopReason": "tool_use",
                "usage": {"inputTokens": 900, "outputTokens": 40, "totalTokens": 1040,
                          "cacheReadInputTokens": 100},
                "metrics": {"latencyMs": 12}
            })
            .to_string(),
        )
    })
    .await;
    let client = client(
        &server,
        BedrockConfig::default().with_guardrail(BedrockGuardrail::new("gr-123", "2")),
    )
    .await;

    let tool = ToolDefinition::new("search_catalog", "Search").with_parameters(
        ParameterSchema::object().with_property("query", ParameterSchema::string("q")),
    );
    let request = GenerateRequest::new(
        "anthropic.claude-sonnet-5",
        vec![
            Message::user("red shoes please"),
            Message::assistant("Sure."),
            Message::assistant("One moment."),
        ],
    )
    .with_system("You are a shop assistant.")
    .with_max_tokens(256)
    .with_tools(vec![tool]);

    let response = client.generate(&request).await.expect("converse");

    let sent = &server.requests()[0];
    assert_eq!(sent.method, "POST");
    assert_eq!(sent.path, "/model/anthropic.claude-sonnet-5/converse");
    assert!(
        sent.header("authorization")
            .is_some_and(|a| a.starts_with("AWS4-HMAC-SHA256")),
        "request must be SigV4-signed"
    );
    let body = sent.json();
    assert_eq!(
        body["system"],
        json!([{"text": "You are a shop assistant."}])
    );
    assert_eq!(body["inferenceConfig"]["maxTokens"], 256);
    assert_eq!(body["guardrailConfig"]["guardrailIdentifier"], "gr-123");
    assert_eq!(body["guardrailConfig"]["guardrailVersion"], "2");
    assert_eq!(
        body["toolConfig"]["tools"][0]["toolSpec"]["name"],
        "search_catalog"
    );
    let messages = body["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[1]["role"], "assistant");
    assert_eq!(messages[1]["content"].as_array().unwrap().len(), 2);

    assert_eq!(response.text().as_deref(), Some("Searching."));
    assert_eq!(response.stop_reason(), Some("tool_use"));
    let call: serde_json::Value = response.parse().unwrap();
    assert_eq!(call, json!({"catalog": {"query": "red shoes"}}));
    let usage = response.usage.expect("usage");
    assert_eq!(usage.input_tokens, Some(1000));
    assert_eq!(usage.cached_input_tokens, Some(100));
    assert_eq!(usage.output_tokens, Some(40));
}

#[tokio::test]
async fn throttling_maps_to_a_retryable_error() {
    let server = StubServer::start(|_| {
        Reply::error(429, r#"{"message":"Too many tokens, please wait"}"#)
            .with_header("x-amzn-ErrorType", "ThrottlingException")
    })
    .await;
    let client = client(&server, BedrockConfig::default()).await;

    let err = client
        .generate(&GenerateRequest::new("m", vec![Message::user("hi")]))
        .await
        .unwrap_err();
    assert!(err.is_retryable(), "{err}");
    assert!(err.to_string().contains("ThrottlingException"), "{err}");
}

#[tokio::test]
async fn a_stalled_reply_times_out_as_a_retryable_error() {
    let server = StubServer::start(|_| {
        Reply::ok(json!({"output": {}}).to_string()).after(Duration::from_secs(3))
    })
    .await;
    let client = client(
        &server,
        BedrockConfig::default().with_timeout(Duration::from_millis(200)),
    )
    .await;

    let started = Instant::now();
    let err = client
        .generate(&GenerateRequest::one_shot("m", "", "hi"))
        .await
        .unwrap_err();

    assert!(started.elapsed() < Duration::from_secs(2), "{err}");
    assert!(err.is_retryable(), "{err}");
}

#[tokio::test]
async fn access_denied_is_permanent() {
    let server = StubServer::start(|_| {
        Reply::error(403, r#"{"message":"not authorized to invoke"}"#)
            .with_header("x-amzn-ErrorType", "AccessDeniedException")
    })
    .await;
    let client = client(&server, BedrockConfig::default()).await;

    let err = client
        .generate(&GenerateRequest::new("m", vec![Message::user("hi")]))
        .await
        .unwrap_err();
    assert!(err.is_provider_error());
    assert!(!err.is_retryable(), "{err}");
}

#[tokio::test]
async fn titan_multimodal_embeds_text_and_image_per_item() {
    let server = StubServer::start(|request| {
        let body = request.json();
        let has_text = body.get("inputText").is_some() as u8 as f32;
        let has_image = body.get("inputImage").is_some() as u8 as f32;
        Reply::ok(json!({"embedding": [has_text, has_image], "inputTextTokenCount": 3}).to_string())
    })
    .await;
    let client = client(&server, BedrockConfig::default()).await;

    let request = EmbedRequest::new("amazon.titan-embed-image-v1", ["red shoe", "blue bag"])
        .with_images([EmbedImage::new(vec![1, 2, 3]), EmbedImage::new(vec![4])])
        .with_output_dimensionality(1024);
    let response = client.embed(&request).await.expect("embed");

    let vectors: Vec<Vec<f32>> = response
        .into_embeddings()
        .unwrap()
        .into_iter()
        .map(|e| e.values)
        .collect();
    assert_eq!(vectors, vec![vec![1.0, 1.0], vec![1.0, 1.0]]);

    let sent = server.requests();
    assert_eq!(sent.len(), 2);
    assert_eq!(sent[0].path, "/model/amazon.titan-embed-image-v1/invoke");
    let first = sent
        .iter()
        .map(|r| r.json())
        .find(|b| b["inputText"] == "red shoe")
        .expect("first item sent");
    assert_eq!(first["inputImage"], "AQID");
    assert_eq!(first["embeddingConfig"]["outputEmbeddingLength"], 1024);
}

#[tokio::test]
async fn one_failed_embedding_keeps_the_others() {
    let server = StubServer::start(|request| {
        if request.json()["inputText"] == "bad" {
            Reply::error(400, r#"{"message":"input too long"}"#)
                .with_header("x-amzn-ErrorType", "ValidationException")
        } else {
            Reply::ok(json!({"embedding": [0.5], "inputTextTokenCount": 1}).to_string())
        }
    })
    .await;
    let client = client(&server, BedrockConfig::default()).await;

    let response = client
        .embed(&EmbedRequest::new(
            "amazon.titan-embed-text-v2:0",
            ["ok", "bad", "ok"],
        ))
        .await
        .expect("embed");
    assert_eq!(
        response.failures().map(|(i, _)| i).collect::<Vec<_>>(),
        vec![1]
    );
    assert_eq!(response.total_token_count, Some(2));
}

const PNG: &[u8] = &[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];

#[tokio::test]
async fn cohere_batches_items_and_pairs_text_with_image() {
    let server = StubServer::start(|request| {
        let body = request.json();
        let n = body["inputs"].as_array().map_or(0, Vec::len);
        let vectors: Vec<Vec<f32>> = (0..n).map(|i| vec![i as f32, 1.0]).collect();
        Reply::ok(
            json!({
                "id": "x",
                "response_type": "embeddings_by_type",
                "embeddings": {"float": vectors},
                "meta": {"billed_units": {"input_tokens": 7}}
            })
            .to_string(),
        )
    })
    .await;
    let client = client(&server, BedrockConfig::default()).await;

    let request = EmbedRequest::new("eu.cohere.embed-v4:0", ["pantofi roșii", "geantă albastră"])
        .with_images([EmbedImage::new(PNG.to_vec()), EmbedImage::new(PNG.to_vec())])
        .with_task_type(EmbedTaskType::RetrievalDocument)
        .with_output_dimensionality(1024);
    let response = client.embed(&request).await.expect("embed");

    let sent = server.requests();
    assert_eq!(sent.len(), 1);
    assert_eq!(sent[0].path, "/model/eu.cohere.embed-v4%3A0/invoke");
    let body = sent[0].json();
    assert_eq!(body["input_type"], "search_document");
    assert_eq!(body["output_dimension"], 1024);
    assert_eq!(body["inputs"][1]["content"][0]["text"], "geantă albastră");
    assert!(
        body["inputs"][1]["content"][1]["image_url"]["url"]
            .as_str()
            .unwrap()
            .starts_with("data:image/png;base64,")
    );

    assert_eq!(response.total_token_count, Some(7));
    let vectors: Vec<Vec<f32>> = response
        .into_embeddings()
        .unwrap()
        .into_iter()
        .map(|e| e.values)
        .collect();
    assert_eq!(vectors, vec![vec![0.0, 1.0], vec![1.0, 1.0]]);
}

#[tokio::test]
async fn cohere_failed_batch_fails_only_its_items() {
    let server = StubServer::start(|request| {
        let texts = request.json()["texts"].as_array().unwrap().clone();
        if texts.iter().any(|t| t == "bad") {
            Reply::error(400, r#"{"message":"bad input"}"#)
                .with_header("x-amzn-ErrorType", "ValidationException")
        } else {
            let vectors: Vec<Vec<f32>> = texts.iter().map(|_| vec![0.5]).collect();
            Reply::ok(json!({"embeddings": vectors}).to_string())
        }
    })
    .await;
    let client = client(&server, BedrockConfig::default()).await;

    let mut texts: Vec<String> = (0..96).map(|i| format!("ok {i}")).collect();
    texts.push("bad".into());
    texts.push("ok last".into());
    let response = client
        .embed(&EmbedRequest::new("cohere.embed-v4:0", texts))
        .await
        .expect("embed");

    assert_eq!(server.requests().len(), 2);
    let failed: Vec<usize> = response.failures().map(|(i, _)| i).collect();
    assert_eq!(failed, vec![96, 97]);
}

#[tokio::test]
async fn nova_embeds_romanian_text_and_an_image_separately() {
    let server = StubServer::start(|request| {
        let params = &request.json()["singleEmbeddingParams"];
        let kind = if params.get("image").is_some() {
            "IMAGE"
        } else {
            "TEXT"
        };
        Reply::ok(
            json!({"embeddings": [{"embeddingType": kind, "embedding": [0.25, 0.75]}]}).to_string(),
        )
    })
    .await;
    let client = client(&server, BedrockConfig::default()).await;

    let model = "amazon.nova-2-multimodal-embeddings-v1:0";
    let text = client
        .embed(
            &EmbedRequest::new(model, ["pantofi sport roșii"])
                .with_task_type(EmbedTaskType::RetrievalQuery)
                .with_output_dimensionality(1024),
        )
        .await
        .expect("text embed");
    let image = client
        .embed(&EmbedRequest::from_images(
            model,
            [EmbedImage::new(PNG.to_vec())],
        ))
        .await
        .expect("image embed");

    assert_eq!(text.into_embeddings().unwrap()[0].values, vec![0.25, 0.75]);
    assert_eq!(image.into_embeddings().unwrap()[0].values, vec![0.25, 0.75]);

    let sent = server.requests();
    let text_body = sent[0].json();
    assert_eq!(text_body["taskType"], "SINGLE_EMBEDDING");
    assert_eq!(
        text_body["singleEmbeddingParams"]["embeddingPurpose"],
        "GENERIC_RETRIEVAL"
    );
    assert_eq!(
        text_body["singleEmbeddingParams"]["text"]["value"],
        "pantofi sport roșii"
    );
    let image_body = sent[1].json();
    assert_eq!(
        image_body["singleEmbeddingParams"]["image"]["format"],
        "png"
    );
    assert_eq!(
        image_body["singleEmbeddingParams"]["embeddingPurpose"],
        "GENERIC_INDEX"
    );
}
