// Ported from engine/server/src/api/openai.{h,cpp} @ 4171493 (error bodies in vLLM's shape).
//! Request errors and their wire form.

use std::error::Error;
use std::fmt;

use ink_text::RequestError;
use serde_json::json;

use crate::frontend::dump;

/// The `type` of an error body.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorKind {
    /// HTTP 400.
    BadRequest,
    /// HTTP 404.
    NotFound,
    /// HTTP 500.
    Internal,
    /// HTTP 503: the back end is not running.
    Unavailable,
}

impl ErrorKind {
    /// The spelling config B puts in `error.type`.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::BadRequest => "BadRequestError",
            Self::NotFound => "NotFoundError",
            Self::Internal => "InternalServerError",
            Self::Unavailable => "ServiceUnavailableError",
        }
    }
}

/// A request the front end rejects, or a failure while generating, with the HTTP status and body config B returns.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ApiError {
    status: u16,
    kind: ErrorKind,
    message: String,
    param: Option<String>,
}

impl ApiError {
    /// HTTP 400.
    pub(crate) fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: 400,
            kind: ErrorKind::BadRequest,
            message: message.into(),
            param: None,
        }
    }

    /// HTTP 400 naming the offending request parameter.
    pub(crate) fn bad_param(message: impl Into<String>, param: &str) -> Self {
        Self {
            param: Some(param.to_owned()),
            ..Self::bad_request(message)
        }
    }

    /// HTTP 404 for an unknown model.
    pub(crate) fn unknown_model(model: &str) -> Self {
        Self {
            status: 404,
            kind: ErrorKind::NotFound,
            message: format!("The model `{model}` does not exist."),
            param: Some("model".to_owned()),
        }
    }

    /// HTTP 500.
    pub(crate) fn internal(message: impl Into<String>) -> Self {
        Self {
            status: 500,
            kind: ErrorKind::Internal,
            message: message.into(),
            param: None,
        }
    }

    /// HTTP 503 for a request that arrives while the back end is down.
    pub(crate) fn unavailable() -> Self {
        Self {
            status: 503,
            kind: ErrorKind::Unavailable,
            message: "the back end is not running".to_owned(),
            param: None,
        }
    }

    /// The HTTP status.
    #[must_use]
    pub fn status(&self) -> u16 {
        self.status
    }

    /// The error type.
    #[must_use]
    pub fn kind(&self) -> ErrorKind {
        self.kind
    }

    /// The message.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }

    /// The request parameter the error names, if any.
    #[must_use]
    pub fn param(&self) -> Option<&str> {
        self.param.as_deref()
    }

    /// The JSON error body: `{"error": {"message", "type", "param", "code"}}`.
    #[must_use]
    pub fn body(&self) -> String {
        dump(&json!({"error": {
            "message": self.message,
            "type": self.kind.as_str(),
            "param": self.param,
            "code": self.status,
        }}))
    }
}

impl fmt::Display for ApiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}: {}", self.status, self.kind.as_str(), self.message)
    }
}

impl Error for ApiError {}

impl From<RequestError> for ApiError {
    fn from(e: RequestError) -> Self {
        Self {
            param: e.param().map(str::to_owned),
            ..Self::bad_request(e.message())
        }
    }
}

/// A failure while a request was running (back end error, bad token id, decoder failure): HTTP 500 or an error
/// frame, and the request counts as failed.
#[derive(Debug)]
pub(crate) struct RunError(pub(crate) String);

impl fmt::Display for RunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for RunError {}

impl From<ink_text::DetokError> for RunError {
    fn from(e: ink_text::DetokError) -> Self {
        Self(e.to_string())
    }
}

impl From<ink_text::TokenizerError> for RunError {
    fn from(e: ink_text::TokenizerError) -> Self {
        Self(e.to_string())
    }
}
