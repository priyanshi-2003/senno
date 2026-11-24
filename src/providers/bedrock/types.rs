use std::fmt::{self, Debug, Formatter};
use std::time::Duration;

use aws_config::SdkConfig;

#[derive(Clone)]
pub struct BedrockClient {
    pub(super) inner: aws_sdk_bedrockruntime::Client,
    pub(super) guardrail: Option<BedrockGuardrail>,
}

impl Debug for BedrockClient {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("BedrockClient")
            .field("region", &self.inner.config().region())
            .field("guardrail", &self.guardrail)
            .finish_non_exhaustive()
    }
}

impl BedrockClient {
    pub(super) fn new(
        config: aws_sdk_bedrockruntime::Config,
        guardrail: Option<BedrockGuardrail>,
    ) -> Self {
        Self {
            inner: aws_sdk_bedrockruntime::Client::from_conf(config),
            guardrail,
        }
    }

    pub fn region(&self) -> Option<&str> {
        self.inner.config().region().map(|r| r.as_ref())
    }

    pub fn sdk_client(&self) -> &aws_sdk_bedrockruntime::Client {
        &self.inner
    }
}

#[derive(Debug, Clone, Default)]
pub struct BedrockConfig {
    pub region: Option<String>,
    pub endpoint_url: Option<String>,
    pub max_attempts: Option<u32>,
    pub timeout: Option<Duration>,
    pub guardrail: Option<BedrockGuardrail>,
    pub sdk_config: Option<SdkConfig>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BedrockGuardrail {
    pub id: String,
    pub version: String,
    pub trace: bool,
}

impl BedrockGuardrail {
    pub fn new(id: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            version: version.into(),
            trace: false,
        }
    }

    pub fn with_trace(mut self, trace: bool) -> Self {
        self.trace = trace;
        self
    }
}

impl BedrockConfig {
    pub fn with_region(mut self, val: impl Into<String>) -> Self {
        self.region = Some(val.into());
        self
    }

    pub fn with_endpoint_url(mut self, val: impl Into<String>) -> Self {
        self.endpoint_url = Some(val.into());
        self
    }

    pub fn with_max_attempts(mut self, val: u32) -> Self {
        self.max_attempts = Some(val);
        self
    }

    pub fn with_timeout(mut self, val: Duration) -> Self {
        self.timeout = Some(val);
        self
    }

    pub fn with_guardrail(mut self, guardrail: BedrockGuardrail) -> Self {
        self.guardrail = Some(guardrail);
        self
    }

    pub fn with_sdk_config(mut self, sdk_config: SdkConfig) -> Self {
        self.sdk_config = Some(sdk_config);
        self
    }
}
