// Ported from engine/server/src/api/server.{h,cpp} @ 4171493 (cpp-httplib there, axum here).
//! The HTTP transport: [`router`] serves the OpenAI endpoints of a [`Frontend`], [`bind_loopback`] opens the only
//! listener the server may have.
//!
//! | Route | Answer |
//! |---|---|
//! | `GET /health` | 200 with an empty body, 503 once the back end is down |
//! | `GET /v1/models`, `GET /metrics` | [`Frontend::models_json`], [`Frontend::metrics_text`] |
//! | `GET /version` | the body given to [`router`] (the build identity) |
//! | `POST /v1/chat/completions`, `POST /v1/completions` | JSON, or server-sent events when the body has `stream: true` |
//!
//! A request is one task. A streaming request runs [`Frontend::execute`] in a spawned task whose writer pushes
//! frames into a bounded channel; the channel is the response body. A client that goes away drops the body, which
//! closes the channel: the task notices at once (the writer fails, or it is waiting on the back end and
//! `Sender::closed` fires), drops the `execute` future and so cancels the request in the back end. A non-streaming
//! request runs in the handler itself, which hyper drops when the connection closes. A client that stops reading
//! fills the channel and is treated as gone, like a write timeout.

use std::convert::Infallible;
use std::io;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use axum::Router;
use axum::body::{Body, Bytes};
use axum::extract::{DefaultBodyLimit, State};
use axum::http::header::{CACHE_CONTROL, CONTENT_TYPE};
use axum::http::{HeaderMap, StatusCode};
use axum::response::Response as HttpResponse;
use axum::routing::{get, post};
use futures_core::Stream;
use tokio::net::TcpListener;
use tokio::sync::mpsc;

use crate::config::RequestMeta;
use crate::error::ApiError;
use crate::frontend::{Frontend, Prepared, Response};

/// Largest request body: 1M-token prompts are 4 to 8 MB of JSON, long agent histories more.
pub const BODY_LIMIT_BYTES: usize = 256 << 20;

/// Frames queued for a client before it counts as stalled. One model step produces a handful of frames.
const FRAME_QUEUE: usize = 1024;

const JSON: &str = "application/json";

/// Why [`bind_loopback`] refused or failed.
#[derive(Debug, thiserror::Error)]
pub enum BindError {
    /// The host is not a loopback address (AGENTS hard rule 5).
    #[error("refusing to bind {0}: endpoints bind loopback only")]
    NotLoopback(String),
    /// The socket could not be bound.
    #[error("bind failed on {addr}")]
    Io {
        /// The address that was tried.
        addr: SocketAddr,
        /// The operating system's error.
        #[source]
        source: io::Error,
    },
}

/// The loopback address `host` names: `localhost` or an IP literal in the loopback range.
fn loopback_ip(host: &str) -> Result<IpAddr, BindError> {
    if host == "localhost" {
        return Ok(Ipv4Addr::LOCALHOST.into());
    }
    host.parse::<IpAddr>()
        .ok()
        .filter(IpAddr::is_loopback)
        .ok_or_else(|| BindError::NotLoopback(host.to_owned()))
}

/// Binds `host:port` (port 0 picks a free one). This is the only way the server opens a listener.
///
/// # Errors
/// [`BindError::NotLoopback`] for any host that is not `localhost` or a loopback IP literal, [`BindError::Io`] when
/// the bind fails.
pub async fn bind_loopback(host: &str, port: u16) -> Result<TcpListener, BindError> {
    let addr = SocketAddr::new(loopback_ip(host)?, port);
    TcpListener::bind(addr)
        .await
        .map_err(|source| BindError::Io { addr, source })
}

#[derive(Clone)]
struct App {
    fe: Arc<Frontend>,
    version: Arc<str>,
}

/// The routes of `fe`. `version` is the JSON body of `/version`.
pub fn router(fe: Arc<Frontend>, version: String) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/version", get(version_body))
        .route("/metrics", get(metrics))
        .route("/v1/chat/completions", post(chat))
        .route("/v1/completions", post(completions))
        .layer(DefaultBodyLimit::max(BODY_LIMIT_BYTES))
        .with_state(App {
            fe,
            version: version.into(),
        })
}

fn reply(status: StatusCode, content_type: &'static str, body: String) -> HttpResponse {
    HttpResponse::builder()
        .status(status)
        .header(CONTENT_TYPE, content_type)
        .body(Body::from(body))
        .expect("a status, a static header and a body make a response")
}

fn error_reply(e: &ApiError) -> HttpResponse {
    let status = StatusCode::from_u16(e.status()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    reply(status, JSON, e.body())
}

async fn health(State(app): State<App>) -> HttpResponse {
    let status = if app.fe.healthy() {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    reply(status, "text/plain", String::new())
}

async fn models(State(app): State<App>) -> HttpResponse {
    reply(StatusCode::OK, JSON, app.fe.models_json())
}

async fn version_body(State(app): State<App>) -> HttpResponse {
    reply(StatusCode::OK, JSON, app.version.to_string())
}

async fn metrics(State(app): State<App>) -> HttpResponse {
    reply(
        StatusCode::OK,
        "text/plain; version=0.0.4; charset=utf-8",
        app.fe.metrics_text(),
    )
}

#[derive(Clone, Copy)]
enum Endpoint {
    Chat,
    Completions,
}

async fn chat(State(app): State<App>, headers: HeaderMap, body: Bytes) -> HttpResponse {
    generate(app, Endpoint::Chat, headers, body).await
}

async fn completions(State(app): State<App>, headers: HeaderMap, body: Bytes) -> HttpResponse {
    generate(app, Endpoint::Completions, headers, body).await
}

/// Validates the body; the prompt is rendered and tokenized here, which is CPU work on up to megabytes of text.
fn prepare(fe: &Frontend, endpoint: Endpoint, body: &[u8], meta: &RequestMeta) -> Result<Prepared, ApiError> {
    let text = std::str::from_utf8(body)
        .map_err(|e| ApiError::bad_request(format!("JSON decode error: the body is not UTF-8: {e}")))?;
    match endpoint {
        Endpoint::Chat => fe.prepare_chat(text, meta),
        Endpoint::Completions => fe.prepare_completions(text, meta),
    }
}

async fn generate(app: App, endpoint: Endpoint, headers: HeaderMap, body: Bytes) -> HttpResponse {
    if !app.fe.healthy() {
        return error_reply(&ApiError::unavailable());
    }
    let meta = RequestMeta {
        session_header: headers
            .get("x-session-id")
            .and_then(|v| v.to_str().ok())
            .unwrap_or_default()
            .to_owned(),
    };
    let fe = Arc::clone(&app.fe);
    let prepared = match tokio::task::spawn_blocking(move || prepare(&fe, endpoint, &body, &meta)).await {
        Ok(Ok(prepared)) => prepared,
        Ok(Err(e)) => return error_reply(&e),
        Err(e) => return error_reply(&ApiError::internal(format!("request preparation failed: {e}"))),
    };
    if prepared.stream() {
        stream(app.fe, prepared)
    } else {
        // Not spawned: when the client closes the connection hyper drops this future, which cancels the request.
        match app.fe.execute(prepared, |_| true).await {
            Response::Json { status, body } => reply(
                StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
                JSON,
                body,
            ),
            Response::Streamed | Response::ClientGone => {
                error_reply(&ApiError::internal("a non-streaming request ended without a body"))
            }
        }
    }
}

/// The response body: frames from the request's task until it ends.
struct Frames(mpsc::Receiver<Bytes>);

impl Stream for Frames {
    type Item = Result<Bytes, Infallible>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.0.poll_recv(cx).map(|frame| frame.map(Ok))
    }
}

fn stream(fe: Arc<Frontend>, prepared: Prepared) -> HttpResponse {
    let (tx, rx) = mpsc::channel(FRAME_QUEUE);
    tokio::spawn(async move {
        let watch = tx.clone();
        let write = move |frame: &str| {
            if frame.is_empty() {
                // liveness probe: no frame, only whether anyone still reads
                return !tx.is_closed();
            }
            // Full means the client stopped reading; Closed means it is gone.
            tx.try_send(Bytes::copy_from_slice(frame.as_bytes())).is_ok()
        };
        tokio::select! {
            // the losing branch is dropped: dropping `execute` cancels the request in the back end
            _ = fe.execute(prepared, write) => {}
            () = watch.closed() => {}
        }
    });
    HttpResponse::builder()
        .status(StatusCode::OK)
        .header(CONTENT_TYPE, "text/event-stream")
        .header(CACHE_CONTROL, "no-cache")
        .body(Body::from_stream(Frames(rx)))
        .expect("a status, static headers and a body make a response")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_loopback_hosts_are_accepted() {
        for host in ["127.0.0.1", "localhost", "::1", "127.0.0.2"] {
            assert!(loopback_ip(host).is_ok(), "{host}");
        }
        for host in [
            "0.0.0.0",
            "::",
            "192.168.1.5",
            "10.0.0.1",
            "example.com",
            "",
            "::ffff:127.0.0.1",
            "127.0.0.1:80",
        ] {
            assert!(
                matches!(loopback_ip(host), Err(BindError::NotLoopback(h)) if h == host),
                "{host:?}"
            );
        }
    }
}
