use aws_config::SdkConfig;

/// Amazon Bedrock client: chat through Converse, embeddings through Titan.
#[derive(Clone)]
pub struct BedrockClient {
    pub(super) inner: aws_sdk_bedrockruntime::Client,
    pub(super) guardrail: Option<BedrockGuardrail>,
}

impl std::fmt::Debug for BedrockClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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

    /// The region calls go to.
    pub fn region(&self) -> Option<&str> {
        self.inner.config().region().map(|r| r.as_ref())
    }

    /// The underlying AWS SDK client, for Bedrock features senno does not wrap.
    pub fn sdk_client(&self) -> &aws_sdk_bedrockruntime::Client {
        &self.inner
    }
}

#[derive(Debug, Clone, Default)]
pub struct BedrockConfig {
    /// AWS region, e.g. `eu-central-1`. Falls back to `BEDROCK_REGION`, then
    /// the AWS default chain (`AWS_REGION`, profile).
    pub region: Option<String>,
    /// Replace the regional endpoint, e.g. for a VPC endpoint without private
    /// DNS or a test stub.
    pub endpoint_url: Option<String>,
    /// Attempts per call including the first; default 3.
    pub max_attempts: Option<u32>,
    /// Apply this Bedrock Guardrail to every chat call.
    pub guardrail: Option<BedrockGuardrail>,
    /// A fully built AWS config to use instead of the default chain (custom
    /// credentials provider, HTTP client, ...). `region`, `endpoint_url` and
    /// `max_attempts` still override it.
    pub sdk_config: Option<SdkConfig>,
}

/// A Bedrock Guardrail applied to every Converse call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BedrockGuardrail {
    /// Guardrail id or ARN.
    pub id: String,
    /// Published version number, or `DRAFT`.
    pub version: String,
    /// Ask Bedrock to return the guardrail trace (for debugging).
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

    pub fn with_guardrail(mut self, guardrail: BedrockGuardrail) -> Self {
        self.guardrail = Some(guardrail);
        self
    }

    pub fn with_sdk_config(mut self, sdk_config: SdkConfig) -> Self {
        self.sdk_config = Some(sdk_config);
        self
    }
}
