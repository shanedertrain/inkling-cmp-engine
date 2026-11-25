//! OpenAI request handling with config B's wire behaviour: validation, prompt rendering, streaming and
//! non-streaming response assembly, usage, logprobs.
//!
//! The core is transport-free: a transport calls [`Frontend::prepare_chat`] or
//! [`Frontend::prepare_completions`] to validate a request body and get an HTTP error before anything streams, then
//! [`Frontend::execute`] to run it against a [`Backend`](ink_backend::Backend). Streaming frames go through a writer
//! closure; dropping the `execute` future cancels the request in the back end, which is how a client disconnect
//! propagates.
//!
//! [`http`] is the transport (package F5): the axum router, SSE streaming, cancel on a dropped response stream and
//! the loopback-only bind. It is ported from `engine/server/src/api/server.*`; the channel pattern is TGI's router
//! and vLLM's `AsyncLLM`: one task per request, a bounded channel from it to the response body.
//!
//! Ported from `engine/server/src/api/openai.*` (package F4 of `docs/RUST-PLAN.md`), a port of vLLM's OpenAI serving
//! layer (`chat_completion/serving.py`, `stream_processor.py`, `message_assembly.py`, `completion/serving.py`;
//! Apache-2.0, Copyright contributors to the vLLM project).
#![forbid(unsafe_code)]

mod chat;
mod completions;
mod config;
mod error;
mod frontend;
mod gen_params;
pub mod http;
mod logprobs;
mod run;
mod stats;

pub use config::{Hooks, RequestMeta, ServerConfig};
pub use error::{ApiError, ErrorKind};
pub use frontend::{ChatPrepared, CompletionPrepared, Frontend, FrontendError, Prepared, Response};
pub use stats::FrontendStats;
