use std::fmt::Display;
use std::mem;

use aws_sdk_bedrockruntime::primitives::Blob;
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use futures::StreamExt;
use futures::stream;
use serde::Deserialize;
use tracing::instrument;

use crate::error::Error;
use crate::types::{EmbedImage, EmbedRequest, EmbedResponse, EmbedTaskType, Embedding};

use super::error::{PROVIDER, from_sdk};
use super::types::BedrockClient;

const REQUEST_CONCURRENCY: usize = 4;

const COHERE_MAX_ITEMS: usize = 96;
const COHERE_MAX_BATCH_BYTES: usize = 15_000_000;

const NOVA_SCHEMA_VERSION: &str = "nova-multimodal-embed-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum EmbedModel {
    TitanTextV1,
    TitanTextV2,
    TitanMultimodal,
    CohereV4,
    NovaMultimodal,
}

impl EmbedModel {
    fn is_titan(self) -> bool {
        matches!(
            self,
            Self::TitanTextV1 | Self::TitanTextV2 | Self::TitanMultimodal
        )
    }

    fn takes_images(self) -> bool {
        matches!(
            self,
            Self::TitanMultimodal | Self::CohereV4 | Self::NovaMultimodal
        )
    }

    fn list_dimensions(self) -> Option<&'static [u32]> {
        match self {
            Self::TitanTextV1 => None,
            Self::TitanTextV2 => Some(&[256, 512, 1024]),
            Self::TitanMultimodal => Some(&[256, 384, 1024]),
            Self::CohereV4 => Some(&[256, 512, 1024, 1536]),
            Self::NovaMultimodal => Some(&[256, 384, 1024, 3072]),
        }
    }

    fn needs_image_format(self) -> bool {
        matches!(self, Self::CohereV4 | Self::NovaMultimodal)
    }
}

pub(super) fn detect_model(model: &str) -> Result<EmbedModel, Error> {
    if model.contains("cohere.embed-v4") {
        Ok(EmbedModel::CohereV4)
    } else if model.contains("nova-2-multimodal-embeddings")
        || model.contains("nova-multimodal-embeddings")
    {
        Ok(EmbedModel::NovaMultimodal)
    } else if model.contains("titan-embed-image") {
        Ok(EmbedModel::TitanMultimodal)
    } else if model.contains("titan-embed-text-v2") {
        Ok(EmbedModel::TitanTextV2)
    } else if model.contains("titan-embed-text") {
        Ok(EmbedModel::TitanTextV1)
    } else {
        Err(Error::config(format!(
            "Bedrock embeddings support Titan (titan-embed-text-v1/v2, titan-embed-image-v1), \
             Cohere Embed v4 (cohere.embed-v4) and Nova Multimodal Embeddings \
             (nova-2-multimodal-embeddings); got '{model}'"
        )))
    }
}

pub(super) fn validate(request: &EmbedRequest) -> Result<EmbedModel, Error> {
    let model = detect_model(&request.model)?;
    if request.title.is_some() {
        return Err(Error::config(
            "Bedrock embedding models do not accept a title",
        ));
    }
    if request.task_type.is_some() && model.is_titan() {
        return Err(Error::config("Titan embeddings do not accept a task_type"));
    }
    if !request.images.is_empty() && !model.takes_images() {
        return Err(Error::config(format!(
            "'{}' embeds text only; use titan-embed-image-v1, cohere.embed-v4 or nova-2-multimodal-embeddings for images",
            request.model
        )));
    }
    let paired = !request.texts.is_empty() && !request.images.is_empty();
    if paired && request.texts.len() != request.images.len() {
        return Err(Error::config(format!(
            "texts ({}) and images ({}) pair up item by item and must be the same length",
            request.texts.len(),
            request.images.len()
        )));
    }
    if paired && model == EmbedModel::NovaMultimodal {
        return Err(Error::config(
            "Nova Multimodal Embeddings takes one input per item: send texts and images as \
             separate requests, or use cohere.embed-v4 to embed a text and its image together",
        ));
    }
    match (request.output_dimensionality, model.list_dimensions()) {
        (Some(_), None) => {
            return Err(Error::config(format!(
                "'{}' has a fixed dimension; output_dimensionality is not accepted",
                request.model
            )));
        }
        (Some(d), Some(allowed)) if !allowed.contains(&d) => {
            return Err(Error::config(format!(
                "'{}' supports output_dimensionality {allowed:?}; got {d}",
                request.model
            )));
        }
        _ => {}
    }
    if model.needs_image_format() {
        for (index, image) in request.images.iter().enumerate() {
            if detect_image_format(&image.data).is_none() {
                return Err(Error::config(format!(
                    "image {index} is not PNG, JPEG, GIF or WebP"
                )));
            }
        }
    }
    Ok(model)
}

pub(super) fn detect_image_format(data: &[u8]) -> Option<&'static str> {
    if data.starts_with(&[0x89, b'P', b'N', b'G']) {
        Some("png")
    } else if data.starts_with(&[0xFF, 0xD8, 0xFF]) {
        Some("jpeg")
    } else if data.starts_with(b"GIF8") {
        Some("gif")
    } else if data.get(..4) == Some(b"RIFF".as_slice())
        && data.get(8..12) == Some(b"WEBP".as_slice())
    {
        Some("webp")
    } else {
        None
    }
}

fn encode_base64(data: &[u8]) -> String {
    STANDARD.encode(data)
}

fn build_data_uri(image: &EmbedImage) -> String {
    let format = detect_image_format(&image.data).unwrap_or("png");
    format!("data:image/{format};base64,{}", encode_base64(&image.data))
}

fn map_cohere_input_type(task: Option<EmbedTaskType>) -> &'static str {
    match task {
        Some(
            EmbedTaskType::RetrievalQuery
            | EmbedTaskType::QuestionAnswering
            | EmbedTaskType::FactVerification
            | EmbedTaskType::CodeRetrievalQuery,
        ) => "search_query",
        Some(EmbedTaskType::Classification) => "classification",
        Some(EmbedTaskType::Clustering) => "clustering",
        _ => "search_document",
    }
}

fn map_nova_purpose(task: Option<EmbedTaskType>) -> &'static str {
    match task {
        Some(
            EmbedTaskType::RetrievalQuery
            | EmbedTaskType::QuestionAnswering
            | EmbedTaskType::FactVerification
            | EmbedTaskType::CodeRetrievalQuery,
        ) => "GENERIC_RETRIEVAL",
        Some(EmbedTaskType::Classification) => "CLASSIFICATION",
        Some(EmbedTaskType::Clustering) => "CLUSTERING",
        _ => "GENERIC_INDEX",
    }
}

pub(super) fn build_item_body(
    model: EmbedModel,
    text: Option<&str>,
    image: Option<&EmbedImage>,
    dimensions: Option<u32>,
    task: Option<EmbedTaskType>,
) -> Result<serde_json::Value, Error> {
    let mut body = serde_json::Map::new();
    match model {
        EmbedModel::TitanTextV1 => {
            if let Some(text) = text {
                body.insert("inputText".into(), text.into());
            }
        }
        EmbedModel::TitanTextV2 => {
            if let Some(text) = text {
                body.insert("inputText".into(), text.into());
            }
            if let Some(d) = dimensions {
                body.insert("dimensions".into(), d.into());
            }
            body.insert("normalize".into(), true.into());
        }
        EmbedModel::TitanMultimodal => {
            if let Some(text) = text {
                body.insert("inputText".into(), text.into());
            }
            if let Some(image) = image {
                body.insert("inputImage".into(), encode_base64(&image.data).into());
            }
            if let Some(d) = dimensions {
                body.insert(
                    "embeddingConfig".into(),
                    serde_json::json!({ "outputEmbeddingLength": d }),
                );
            }
        }
        EmbedModel::NovaMultimodal => {
            let mut params = serde_json::Map::new();
            params.insert("embeddingPurpose".into(), map_nova_purpose(task).into());
            if let Some(d) = dimensions {
                params.insert("embeddingDimension".into(), d.into());
            }
            if let Some(text) = text {
                params.insert(
                    "text".into(),
                    serde_json::json!({ "truncationMode": "END", "value": text }),
                );
            } else if let Some(image) = image {
                params.insert(
                    "image".into(),
                    serde_json::json!({
                        "format": detect_image_format(&image.data).unwrap_or("png"),
                        "source": { "bytes": encode_base64(&image.data) },
                    }),
                );
            }
            body.insert("schemaVersion".into(), NOVA_SCHEMA_VERSION.into());
            body.insert("taskType".into(), "SINGLE_EMBEDDING".into());
            body.insert("singleEmbeddingParams".into(), params.into());
        }
        EmbedModel::CohereV4 => {
            return Err(Error::internal(
                "Cohere embeddings are sent in batches, not one item per call",
            ));
        }
    }
    Ok(serde_json::Value::Object(body))
}

fn build_cohere_item(request: &EmbedRequest, index: usize) -> Result<serde_json::Value, Error> {
    let missing = || Error::internal(format!("embedding item {index} is out of range"));
    let text = request
        .texts
        .get(index)
        .map(|text| serde_json::json!({ "type": "text", "text": text }));
    let image = request.images.get(index).map(
        |image| serde_json::json!({ "type": "image_url", "image_url": { "url": build_data_uri(image) } }),
    );
    let content: Vec<serde_json::Value> = text.into_iter().chain(image).collect();
    if content.is_empty() {
        return Err(missing());
    }
    Ok(serde_json::json!({ "content": content }))
}

pub(super) fn build_cohere_body(
    request: &EmbedRequest,
    items: &[usize],
) -> Result<serde_json::Value, Error> {
    let missing = |index: usize| Error::internal(format!("embedding item {index} is out of range"));
    let mut body = serde_json::Map::new();
    body.insert(
        "input_type".into(),
        map_cohere_input_type(request.task_type).into(),
    );
    body.insert("embedding_types".into(), serde_json::json!(["float"]));
    body.insert("truncate".into(), "RIGHT".into());
    if let Some(d) = request.output_dimensionality {
        body.insert("output_dimension".into(), d.into());
    }

    match (request.texts.is_empty(), request.images.is_empty()) {
        (false, true) => {
            let texts = items
                .iter()
                .map(|&i| {
                    request
                        .texts
                        .get(i)
                        .map(String::as_str)
                        .ok_or_else(|| missing(i))
                })
                .collect::<Result<Vec<&str>, Error>>()?;
            body.insert("texts".into(), serde_json::json!(texts));
        }
        (true, false) => {
            let images = items
                .iter()
                .map(|&i| {
                    request
                        .images
                        .get(i)
                        .map(build_data_uri)
                        .ok_or_else(|| missing(i))
                })
                .collect::<Result<Vec<String>, Error>>()?;
            body.insert("images".into(), serde_json::json!(images));
        }
        _ => {
            let inputs = items
                .iter()
                .map(|&i| build_cohere_item(request, i))
                .collect::<Result<Vec<serde_json::Value>, Error>>()?;
            body.insert("inputs".into(), serde_json::json!(inputs));
        }
    }
    Ok(serde_json::Value::Object(body))
}

pub(super) fn split_cohere_batches(request: &EmbedRequest) -> Vec<Vec<usize>> {
    let mut batches: Vec<Vec<usize>> = Vec::new();
    let mut current: Vec<usize> = Vec::new();
    let mut bytes = 0usize;
    for index in 0..request.item_count() {
        let item_bytes = request.texts.get(index).map_or(0, String::len)
            + request
                .images
                .get(index)
                .map_or(0, |img| img.data.len().div_ceil(3) * 4 + 32);
        if !current.is_empty()
            && (current.len() >= COHERE_MAX_ITEMS || bytes + item_bytes > COHERE_MAX_BATCH_BYTES)
        {
            batches.push(mem::take(&mut current));
            bytes = 0;
        }
        current.push(index);
        bytes += item_bytes;
    }
    if !current.is_empty() {
        batches.push(current);
    }
    batches
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct TitanResponse {
    embedding: Vec<f32>,
    #[serde(default)]
    input_text_token_count: Option<u32>,
}

#[derive(Deserialize)]
struct NovaResponse {
    embeddings: Vec<NovaEmbedding>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct NovaEmbedding {
    embedding: Vec<f32>,
    #[serde(default)]
    truncated_char_length: Option<u32>,
}

#[derive(Deserialize)]
struct CohereResponse {
    embeddings: CohereEmbeddings,
    #[serde(default)]
    meta: Option<CohereMeta>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum CohereEmbeddings {
    ByType { float: Vec<Vec<f32>> },
    Floats(Vec<Vec<f32>>),
}

#[derive(Deserialize)]
struct CohereMeta {
    #[serde(default)]
    billed_units: Option<CohereBilledUnits>,
}

#[derive(Deserialize)]
struct CohereBilledUnits {
    #[serde(default)]
    input_tokens: Option<u32>,
}

#[instrument(skip_all, fields(model = %request.model, items = request.item_count()))]
pub(super) async fn embed(
    client: &BedrockClient,
    request: &EmbedRequest,
) -> Result<EmbedResponse, Error> {
    let model = validate(request)?;
    if model == EmbedModel::CohereV4 {
        return embed_cohere(client, request).await;
    }

    let embeddings: Vec<Result<Embedding, Error>> = stream::iter(0..request.item_count())
        .map(|i| embed_one(client, request, model, i))
        .buffered(REQUEST_CONCURRENCY)
        .collect()
        .await;

    let counted: Vec<u32> = embeddings
        .iter()
        .filter_map(|r| r.as_ref().ok().and_then(|e| e.token_count))
        .collect();
    let total_token_count = (!counted.is_empty()).then(|| counted.iter().sum());

    Ok(EmbedResponse {
        embeddings,
        total_token_count,
    })
}

async fn invoke(
    client: &BedrockClient,
    model_id: &str,
    body: &serde_json::Value,
) -> Result<Vec<u8>, Error> {
    let payload = serde_json::to_vec(body)
        .map_err(|e| Error::internal(format!("embedding request did not serialize: {e}")))?;
    let output = client
        .inner
        .invoke_model()
        .model_id(model_id)
        .content_type("application/json")
        .accept("application/json")
        .body(Blob::new(payload))
        .send()
        .await
        .map_err(|e| from_sdk("InvokeModel", e))?;
    Ok(output.body().as_ref().to_vec())
}

fn report_unreadable(model: &str, e: impl Display) -> Error {
    Error::provider_permanent(PROVIDER, format!("{model} response was not readable: {e}"))
}

async fn embed_one(
    client: &BedrockClient,
    request: &EmbedRequest,
    model: EmbedModel,
    index: usize,
) -> Result<Embedding, Error> {
    let body = build_item_body(
        model,
        request.texts.get(index).map(String::as_str),
        request.images.get(index),
        request.output_dimensionality,
        request.task_type,
    )?;
    let raw = invoke(client, &request.model, &body).await?;

    let embedding = if model == EmbedModel::NovaMultimodal {
        let parsed: NovaResponse =
            serde_json::from_slice(&raw).map_err(|e| report_unreadable("Nova", e))?;
        let first = parsed
            .embeddings
            .into_iter()
            .next()
            .ok_or_else(|| report_unreadable("Nova", "no embeddings returned"))?;
        Embedding {
            values: first.embedding,
            token_count: None,
            truncated: Some(first.truncated_char_length.is_some()),
        }
    } else {
        let parsed: TitanResponse =
            serde_json::from_slice(&raw).map_err(|e| report_unreadable("Titan", e))?;
        Embedding {
            values: parsed.embedding,
            token_count: parsed.input_text_token_count,
            truncated: None,
        }
    };

    if embedding.values.is_empty() {
        return Err(Error::provider_permanent(
            PROVIDER,
            "the model returned an empty embedding",
        ));
    }
    Ok(embedding)
}

type CohereBatch = (Vec<Vec<f32>>, Option<u32>);

async fn send_cohere_batch(
    client: &BedrockClient,
    request: &EmbedRequest,
    items: Vec<usize>,
) -> (Vec<usize>, Result<CohereBatch, Error>) {
    let result = request_cohere_batch(client, request, &items).await;
    (items, result)
}

async fn request_cohere_batch(
    client: &BedrockClient,
    request: &EmbedRequest,
    items: &[usize],
) -> Result<CohereBatch, Error> {
    let body = build_cohere_body(request, items)?;
    let raw = invoke(client, &request.model, &body).await?;
    let parsed: CohereResponse =
        serde_json::from_slice(&raw).map_err(|e| report_unreadable("Cohere", e))?;
    let vectors = match parsed.embeddings {
        CohereEmbeddings::ByType { float } => float,
        CohereEmbeddings::Floats(v) => v,
    };
    if vectors.len() != items.len() {
        return Err(report_unreadable(
            "Cohere",
            format!("{} vectors for {} items", vectors.len(), items.len()),
        ));
    }
    let tokens = parsed
        .meta
        .and_then(|m| m.billed_units)
        .and_then(|b| b.input_tokens);
    Ok((vectors, tokens))
}

async fn embed_cohere(
    client: &BedrockClient,
    request: &EmbedRequest,
) -> Result<EmbedResponse, Error> {
    let batches = split_cohere_batches(request);

    let results: Vec<(Vec<usize>, Result<CohereBatch, Error>)> = stream::iter(batches)
        .map(|items| send_cohere_batch(client, request, items))
        .buffered(REQUEST_CONCURRENCY)
        .collect()
        .await;

    let mut embeddings: Vec<Option<Result<Embedding, Error>>> =
        (0..request.item_count()).map(|_| None).collect();
    let mut total: Option<u32> = None;
    for (items, result) in results {
        match result {
            Ok((vectors, tokens)) => {
                if let Some(t) = tokens {
                    total = Some(total.unwrap_or(0).saturating_add(t));
                }
                for (&index, values) in items.iter().zip(vectors) {
                    let Some(slot) = embeddings.get_mut(index) else {
                        continue;
                    };
                    *slot = Some(if values.is_empty() {
                        Err(Error::provider_permanent(
                            PROVIDER,
                            "Cohere returned an empty embedding",
                        ))
                    } else {
                        Ok(Embedding {
                            values,
                            token_count: None,
                            truncated: None,
                        })
                    });
                }
            }
            Err(e) => {
                for &index in &items {
                    if let Some(slot) = embeddings.get_mut(index) {
                        *slot = Some(Err(e.clone()));
                    }
                }
            }
        }
    }

    Ok(EmbedResponse {
        embeddings: embeddings
            .into_iter()
            .map(|e| e.unwrap_or_else(|| Err(Error::internal("embedding item was never sent"))))
            .collect(),
        total_token_count: total,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const PNG: &[u8] = &[0x89, b'P', b'N', b'G', 0x0D, 0x0A];
    const JPEG: &[u8] = &[0xFF, 0xD8, 0xFF, 0xE0];

    #[test]
    fn model_ids_map_to_their_kind_with_or_without_profile_prefix() {
        let cases = [
            ("amazon.titan-embed-image-v1", EmbedModel::TitanMultimodal),
            ("amazon.titan-embed-text-v2:0", EmbedModel::TitanTextV2),
            ("amazon.titan-embed-text-v1", EmbedModel::TitanTextV1),
            ("cohere.embed-v4:0", EmbedModel::CohereV4),
            ("eu.cohere.embed-v4:0", EmbedModel::CohereV4),
            (
                "amazon.nova-2-multimodal-embeddings-v1:0",
                EmbedModel::NovaMultimodal,
            ),
        ];
        for (id, kind) in cases {
            assert_eq!(detect_model(id).unwrap(), kind, "{id}");
        }
        assert!(detect_model("cohere.embed-english-v3").is_err());
    }

    #[test]
    fn image_formats_are_detected_from_bytes() {
        assert_eq!(detect_image_format(PNG), Some("png"));
        assert_eq!(detect_image_format(JPEG), Some("jpeg"));
        assert_eq!(detect_image_format(b"GIF89a"), Some("gif"));
        assert_eq!(detect_image_format(b"RIFF\0\0\0\0WEBPVP8 "), Some("webp"));
        assert_eq!(detect_image_format(b"%PDF-1.7"), None);
    }

    #[test]
    fn titan_multimodal_body_carries_text_image_and_length() {
        let image = EmbedImage::new(vec![1u8, 2, 3]);
        let b = build_item_body(
            EmbedModel::TitanMultimodal,
            Some("red shoe"),
            Some(&image),
            Some(1024),
            None,
        )
        .unwrap();
        assert_eq!(
            b,
            json!({
                "inputText": "red shoe",
                "inputImage": "AQID",
                "embeddingConfig": {"outputEmbeddingLength": 1024},
            })
        );
    }

    #[test]
    fn titan_text_v2_body_normalizes_and_sets_dimensions() {
        assert_eq!(
            build_item_body(EmbedModel::TitanTextV2, Some("hi"), None, Some(512), None).unwrap(),
            json!({"inputText": "hi", "dimensions": 512, "normalize": true})
        );
    }

    #[test]
    fn nova_body_uses_index_purpose_for_documents_and_retrieval_for_queries() {
        let doc = build_item_body(
            EmbedModel::NovaMultimodal,
            Some("pantofi sport"),
            None,
            Some(1024),
            Some(EmbedTaskType::RetrievalDocument),
        )
        .unwrap();
        assert_eq!(
            doc,
            json!({
                "schemaVersion": "nova-multimodal-embed-v1",
                "taskType": "SINGLE_EMBEDDING",
                "singleEmbeddingParams": {
                    "embeddingPurpose": "GENERIC_INDEX",
                    "embeddingDimension": 1024,
                    "text": {"truncationMode": "END", "value": "pantofi sport"},
                },
            })
        );
        let query = build_item_body(
            EmbedModel::NovaMultimodal,
            Some("q"),
            None,
            None,
            Some(EmbedTaskType::RetrievalQuery),
        )
        .unwrap();
        assert_eq!(
            query["singleEmbeddingParams"]["embeddingPurpose"],
            "GENERIC_RETRIEVAL"
        );
    }

    #[test]
    fn nova_image_body_names_the_format() {
        let image = EmbedImage::new(JPEG.to_vec());
        let b =
            build_item_body(EmbedModel::NovaMultimodal, None, Some(&image), None, None).unwrap();
        assert_eq!(b["singleEmbeddingParams"]["image"]["format"], "jpeg");
        assert_eq!(
            b["singleEmbeddingParams"]["image"]["source"]["bytes"],
            "/9j/4A=="
        );
    }

    #[test]
    fn cohere_body_picks_texts_images_or_paired_inputs() {
        let texts = EmbedRequest::new("cohere.embed-v4:0", ["a", "b"])
            .with_task_type(EmbedTaskType::RetrievalQuery)
            .with_output_dimensionality(1024);
        let b = build_cohere_body(&texts, &[0, 1]).unwrap();
        assert_eq!(b["texts"], json!(["a", "b"]));
        assert_eq!(b["input_type"], "search_query");
        assert_eq!(b["output_dimension"], 1024);
        assert_eq!(b["embedding_types"], json!(["float"]));

        let images =
            EmbedRequest::from_images("cohere.embed-v4:0", [EmbedImage::new(PNG.to_vec())]);
        let b = build_cohere_body(&images, &[0]).unwrap();
        assert!(
            b["images"][0]
                .as_str()
                .unwrap()
                .starts_with("data:image/png;base64,")
        );
        assert_eq!(b["input_type"], "search_document");

        let paired = EmbedRequest::new("cohere.embed-v4:0", ["red shoe"])
            .with_images([EmbedImage::new(JPEG.to_vec())]);
        let b = build_cohere_body(&paired, &[0]).unwrap();
        let content = &b["inputs"][0]["content"];
        assert_eq!(content[0], json!({"type": "text", "text": "red shoe"}));
        assert!(
            content[1]["image_url"]["url"]
                .as_str()
                .unwrap()
                .starts_with("data:image/jpeg;base64,")
        );
    }

    #[test]
    fn cohere_batches_split_at_96_items_and_by_size() {
        let many = EmbedRequest::new("cohere.embed-v4:0", (0..200).map(|i| i.to_string()));
        let sizes: Vec<usize> = split_cohere_batches(&many).iter().map(Vec::len).collect();
        assert_eq!(sizes, vec![96, 96, 8]);

        let big = vec![0u8; 6_000_000];
        let heavy = EmbedRequest::from_images(
            "cohere.embed-v4:0",
            [EmbedImage::new(big.clone()), EmbedImage::new(big)],
        );
        assert_eq!(split_cohere_batches(&heavy), vec![vec![0], vec![1]]);
    }

    #[test]
    fn cohere_is_never_sent_one_item_at_a_time() {
        let err = build_item_body(EmbedModel::CohereV4, Some("a"), None, None, None).unwrap_err();
        assert!(err.to_string().contains("batches"), "{err}");
    }

    #[test]
    fn out_of_range_cohere_items_are_an_error_not_a_panic() {
        let request = EmbedRequest::new("cohere.embed-v4:0", ["a"]);
        assert!(build_cohere_body(&request, &[0, 5]).is_err());
    }

    #[test]
    fn validation_rules() {
        let r = EmbedRequest::new("amazon.titan-embed-text-v2:0", ["a"])
            .with_images([EmbedImage::new(PNG.to_vec())]);
        assert!(validate(&r).is_err());
        let r = EmbedRequest::new("cohere.embed-v4:0", ["a", "b"])
            .with_images([EmbedImage::new(PNG.to_vec())]);
        assert!(validate(&r).is_err());
        let r = EmbedRequest::new("amazon.nova-2-multimodal-embeddings-v1:0", ["a"])
            .with_images([EmbedImage::new(PNG.to_vec())]);
        assert!(validate(&r).unwrap_err().to_string().contains("cohere"));
        let r = EmbedRequest::new("cohere.embed-v4:0", ["a"]).with_output_dimensionality(384);
        assert!(validate(&r).is_err());
        let r =
            EmbedRequest::new("amazon.titan-embed-text-v1", ["a"]).with_output_dimensionality(256);
        assert!(validate(&r).is_err());
        let r = EmbedRequest::from_images("cohere.embed-v4:0", [EmbedImage::new(b"%PDF".to_vec())]);
        assert!(validate(&r).is_err());
        let r = EmbedRequest::new("amazon.titan-embed-image-v1", ["a"])
            .with_task_type(EmbedTaskType::RetrievalDocument);
        assert!(validate(&r).unwrap_err().to_string().contains("task_type"));
        let r = EmbedRequest::new("cohere.embed-v4:0", ["a"])
            .with_task_type(EmbedTaskType::RetrievalQuery);
        assert_eq!(validate(&r).unwrap(), EmbedModel::CohereV4);
        let r =
            EmbedRequest::from_images("amazon.titan-embed-image-v1", [EmbedImage::new(vec![1])]);
        assert_eq!(validate(&r).unwrap(), EmbedModel::TitanMultimodal);
        assert_eq!(r.item_count(), 1);
        let r = EmbedRequest::from_images(
            "amazon.nova-2-multimodal-embeddings-v1:0",
            [EmbedImage::new(PNG.to_vec())],
        )
        .with_output_dimensionality(3072);
        assert_eq!(validate(&r).unwrap(), EmbedModel::NovaMultimodal);
    }
}
