use std::env;
use std::time::Duration;

use aws_config::retry::RetryConfig;
use aws_config::timeout::TimeoutConfig;
use aws_config::{BehaviorVersion, Region};
use aws_sdk_bedrockruntime::config::Builder;

use crate::error::Error;

use super::types::BedrockConfig;

const REGION_ENV: &str = "BEDROCK_REGION";

const DEFAULT_MAX_ATTEMPTS: u32 = 3;

const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);

pub(super) async fn resolve(
    config: &BedrockConfig,
) -> Result<aws_sdk_bedrockruntime::Config, Error> {
    resolve_with_env_region(config, env::var(REGION_ENV).ok()).await
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
    let mut builder = Builder::from(&sdk);
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
    builder = builder.timeout_config(
        TimeoutConfig::builder()
            .connect_timeout(CONNECT_TIMEOUT)
            .read_timeout(config.timeout.unwrap_or(DEFAULT_TIMEOUT))
            .build(),
    );

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
    async fn every_call_gets_a_connect_and_read_timeout() {
        let sdk = SdkConfig::builder()
            .behavior_version(BehaviorVersion::latest())
            .region(Region::new("us-east-1"))
            .build();
        let default =
            resolve_with_env_region(&BedrockConfig::default().with_sdk_config(sdk.clone()), None)
                .await
                .unwrap();
        let timeouts = default.timeout_config().unwrap();
        assert_eq!(timeouts.connect_timeout(), Some(CONNECT_TIMEOUT));
        assert_eq!(timeouts.read_timeout(), Some(DEFAULT_TIMEOUT));

        let custom = resolve_with_env_region(
            &BedrockConfig::default()
                .with_sdk_config(sdk)
                .with_timeout(Duration::from_secs(20)),
            None,
        )
        .await
        .unwrap();
        assert_eq!(
            custom.timeout_config().unwrap().read_timeout(),
            Some(Duration::from_secs(20))
        );
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
