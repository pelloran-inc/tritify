//! The OpenAI wire format: request bodies, OpenAI-style errors, and the small conversions between them. Unknown
//! request fields are ignored, as OpenAI-compatible servers do.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Deserialize;
use serde_json::{json, Value};

/// An error in OpenAI's shape: `{"error": {"message", "type", "param", "code"}}`.
#[derive(Debug)]
pub(crate) struct ApiError {
    pub status: StatusCode,
    pub message: String,
    pub kind: &'static str,
    pub code: Option<&'static str>,
}

impl ApiError {
    pub fn bad(message: impl Into<String>) -> ApiError {
        ApiError { status: StatusCode::BAD_REQUEST, message: message.into(), kind: "invalid_request_error", code: None }
    }
    pub fn unknown_model(asked: &str, served: &str) -> ApiError {
        ApiError {
            status: StatusCode::NOT_FOUND,
            message: format!("The model '{asked}' does not exist; this server serves '{served}'"),
            kind: "invalid_request_error",
            code: Some("model_not_found"),
        }
    }
    /// A request this server is configured to refuse (the runtime switch when it is off).
    pub fn forbidden(message: impl Into<String>, code: &'static str) -> ApiError {
        ApiError {
            status: StatusCode::FORBIDDEN,
            message: message.into(),
            kind: "invalid_request_error",
            code: Some(code),
        }
    }
    pub fn internal(message: impl Into<String>) -> ApiError {
        ApiError {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: message.into(),
            kind: "server_error",
            code: None,
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = json!({"error": {"message": self.message, "type": self.kind, "param": null, "code": self.code}});
        (self.status, Json(body)).into_response()
    }
}

/// Parse a JSON body into `T`, answering a malformed one with an OpenAI-style 400.
pub(crate) fn parse<T: for<'de> Deserialize<'de>>(body: &[u8]) -> Result<T, ApiError> {
    serde_json::from_slice(body).map_err(|e| ApiError::bad(format!("invalid request body: {e}")))
}

#[derive(Deserialize)]
pub(crate) struct ChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: Value,
}

/// The fields shared by chat and completion requests.
#[derive(Deserialize, Default)]
pub(crate) struct Common {
    pub model: Option<String>,
    pub max_tokens: Option<usize>,
    pub max_completion_tokens: Option<usize>,
    #[serde(default)]
    pub stream: bool,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub n: Option<usize>,
    pub stop: Option<Value>,
}

#[derive(Deserialize)]
pub(crate) struct ChatRequest {
    pub messages: Vec<ChatMessage>,
    #[serde(flatten)]
    pub common: Common,
}

#[derive(Deserialize)]
pub(crate) struct CompletionRequest {
    pub prompt: Value,
    #[serde(flatten)]
    pub common: Common,
}

#[derive(Deserialize)]
pub(crate) struct EmbeddingRequest {
    pub model: Option<String>,
    pub input: Value,
    pub encoding_format: Option<String>,
}

#[derive(Deserialize)]
pub(crate) struct VerifyRequest {
    /// `/v1/chat/completions`, `/v1/completions` or `/v1/embeddings`.
    pub endpoint: String,
    pub request: Value,
    pub hash: String,
}

#[derive(Deserialize)]
pub(crate) struct RuntimeRequest {
    pub backend: Option<String>,
    pub max_threads: Option<usize>,
}

/// A message's text: a string, or an array of `{"type": "text", "text": …}` parts; `null` is empty.
pub(crate) fn content_text(v: &Value) -> Result<String, ApiError> {
    match v {
        Value::Null => Ok(String::new()),
        Value::String(s) => Ok(s.clone()),
        Value::Array(parts) => parts
            .iter()
            .map(|p| match (p.get("type").and_then(Value::as_str), p.get("text").and_then(Value::as_str)) {
                (Some("text"), Some(t)) => Ok(t.to_string()),
                _ => Err(ApiError::bad("only text content parts are supported")),
            })
            .collect(),
        _ => Err(ApiError::bad("message content must be a string or an array of text parts")),
    }
}

/// Standard base64 (RFC 4648, with padding), for `encoding_format: "base64"` embeddings.
pub(crate) fn base64(bytes: &[u8]) -> String {
    const A: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut s = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for c in bytes.chunks(3) {
        let n = (c[0] as u32) << 16 | (*c.get(1).unwrap_or(&0) as u32) << 8 | *c.get(2).unwrap_or(&0) as u32;
        for i in 0..4 {
            s.push(if i <= c.len() { A[(n >> (18 - 6 * i) & 63) as usize] as char } else { '=' });
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_matches_rfc_4648() {
        assert_eq!(base64(b""), "");
        assert_eq!(base64(b"f"), "Zg==");
        assert_eq!(base64(b"fo"), "Zm8=");
        assert_eq!(base64(b"foo"), "Zm9v");
        assert_eq!(base64(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn content_is_a_string_or_text_parts() {
        assert_eq!(content_text(&json!("hi")).unwrap(), "hi");
        assert_eq!(content_text(&json!([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])).unwrap(), "ab");
        assert!(content_text(&json!([{"type": "image_url", "image_url": {}}])).is_err());
    }
}
