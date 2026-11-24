use std::error::Error as StdError;
use std::fmt::Debug;

use aws_sdk_bedrockruntime::error::{DisplayErrorContext, ProvideErrorMetadata, SdkError};

use crate::error::Error;

pub(super) const PROVIDER: &str = "bedrock";

const RETRYABLE_CODES: &[&str] = &[
    "ThrottlingException",
    "ServiceUnavailableException",
    "InternalServerException",
    "ModelNotReadyException",
    "ModelTimeoutException",
    "ModelStreamErrorException",
];

pub(super) fn from_code(operation: &str, code: Option<&str>, message: Option<&str>) -> Error {
    let code_str = code.unwrap_or("Unknown");
    let detail = match message {
        Some(m) if !m.is_empty() => format!("{operation} {code_str}: {m}"),
        _ => format!("{operation} {code_str}"),
    };
    if code.is_some_and(|c| RETRYABLE_CODES.contains(&c)) {
        Error::provider(PROVIDER, detail)
    } else {
        Error::provider_permanent(PROVIDER, detail)
    }
}

pub(super) fn from_sdk<E, R>(operation: &str, err: SdkError<E, R>) -> Error
where
    E: ProvideErrorMetadata + StdError + Send + Sync + 'static,
    R: Debug,
{
    match &err {
        SdkError::ServiceError(service) => {
            let e = service.err();
            from_code(operation, e.code(), e.message())
        }
        SdkError::ConstructionFailure(_) => Error::config(format!(
            "{operation}: request could not be built: {}",
            DisplayErrorContext(&err)
        )),
        SdkError::TimeoutError(_) => Error::provider(
            PROVIDER,
            format!("{operation}: {}", DisplayErrorContext(&err)),
        ),
        SdkError::DispatchFailure(failure) if failure.is_timeout() || failure.is_io() => {
            Error::provider(
                PROVIDER,
                format!("{operation}: {}", DisplayErrorContext(&err)),
            )
        }
        _ => Error::provider_permanent(
            PROVIDER,
            format!("{operation}: {}", DisplayErrorContext(&err)),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn throttling_is_retryable() {
        let e = from_code(
            "Converse",
            Some("ThrottlingException"),
            Some("Too many tokens"),
        );
        assert!(e.is_retryable());
        assert_eq!(
            e.to_string(),
            "bedrock: Converse ThrottlingException: Too many tokens"
        );
    }

    #[test]
    fn validation_and_access_denied_are_permanent() {
        for code in [
            "ValidationException",
            "AccessDeniedException",
            "ResourceNotFoundException",
        ] {
            let e = from_code("Converse", Some(code), None);
            assert!(e.is_provider_error());
            assert!(!e.is_retryable(), "{code} must not be retryable");
        }
    }

    fn dispatch(connector: aws_sdk_bedrockruntime::error::ConnectorError) -> Error {
        let err: SdkError<
            aws_sdk_bedrockruntime::operation::converse::ConverseError,
            aws_sdk_bedrockruntime::config::http::HttpResponse,
        > = SdkError::dispatch_failure(connector);
        from_sdk("Converse", err)
    }

    #[test]
    fn credential_failures_are_permanent_but_io_and_timeouts_retry() {
        use aws_sdk_bedrockruntime::error::ConnectorError;
        assert!(!dispatch(ConnectorError::other("no credentials".into(), None)).is_retryable());
        assert!(dispatch(ConnectorError::io("connection reset".into())).is_retryable());
        assert!(dispatch(ConnectorError::timeout("read timed out".into())).is_retryable());
    }

    #[test]
    fn missing_code_is_permanent() {
        assert!(!from_code("InvokeModel", None, None).is_retryable());
    }
}
