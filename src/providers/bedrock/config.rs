use aws_config::retry::RetryConfig;
use aws_config::{BehaviorVersion, Region};

use crate::error::Error;

use super::types::BedrockConfig;

/// Region env var consulted before the AWS default chain, so a host can point
/// Bedrock at a different region than its other AWS clients.
const REGION_ENV: &str = "BEDROCK_REGION";

/// Attempts per call, the first included. The SDK's standard retry mode backs
/// off on throttling and transient server errors.
const DEFAULT_MAX_ATTEMPTS: u32 = 3;

pub(super) async fn resolve(
    config: &BedrockConfig,
) -> Result<aws_sdk_bedrockruntime::Config, Error> {
    resolve_with_env_region(config, std::env::var(REGION_ENV).ok()).await
}

async fn resolve_with_env_region(
    config: &BedrockConfig,
    env_region: Option<String>,
) -> Result<aws_sdk_bedrockruntime::Config, Error> {
    let explicit_region = config
        .region
        .clone()
        .or_else(|| env_region.filter(|r| !r.trim().is_empty()));

    let sdk = match &config.sdk_config {
        Some(sdk) => sdk.clone(),
        None => {
            let mut loader = aws_config::defaults(BehaviorVersion::latest());
            if let Some(region) = &explicit_region {
                loader = loader.region(Region::new(region.clone()));
            }
            loader.load().await
        }
    };

    let has_region = explicit_region.is_some() || sdk.region().is_some();
    let mut builder = aws_sdk_bedrockruntime::config::Builder::from(&sdk);
    if sdk.behavior_version().is_none() {
        builder = builder.behavior_version(BehaviorVersion::latest());
    }
    if let Some(region) = explicit_region {
        builder = builder.region(Region::new(region));
    }
    if !has_region {
        return Err(Error::config(
            "Bedrock needs a region: set BedrockConfig::with_region, BEDROCK_REGION or AWS_REGION",
        ));
    }
    if let Some(url) = &config.endpoint_url {
        builder = builder.endpoint_url(url.trim_end_matches('/'));
    }
    let attempts = config.max_attempts.unwrap_or(DEFAULT_MAX_ATTEMPTS).max(1);
    builder = builder.retry_config(RetryConfig::standard().with_max_attempts(attempts));

    Ok(builder.build())
}

#[cfg(test)]
mod tests {
    use super::*;
    use aws_config::SdkConfig;

    #[tokio::test]
    async fn missing_region_is_a_config_error() {
        let no_region = SdkConfig::builder()
            .behavior_version(BehaviorVersion::latest())
            .build();
        let err =
            resolve_with_env_region(&BedrockConfig::default().with_sdk_config(no_region), None)
                .await
                .unwrap_err();
        assert!(err.to_string().contains("region"), "{err}");
    }

    #[tokio::test]
    async fn env_region_is_used_when_config_has_none() {
        let sdk = SdkConfig::builder()
            .behavior_version(BehaviorVersion::latest())
            .build();
        let config = resolve_with_env_region(
            &BedrockConfig::default().with_sdk_config(sdk),
            Some("eu-central-1".into()),
        )
        .await
        .unwrap();
        assert_eq!(config.region().map(|r| r.as_ref()), Some("eu-central-1"));
    }

    #[tokio::test]
    async fn sdk_config_without_behavior_version_does_not_panic() {
        let sdk = SdkConfig::builder()
            .region(Region::new("us-east-1"))
            .build();
        let config = resolve_with_env_region(&BedrockConfig::default().with_sdk_config(sdk), None)
            .await
            .unwrap();
        let _client = aws_sdk_bedrockruntime::Client::from_conf(config);
    }
}
