// Ported from engine/server/src/api/openai.h @ 4171493 (ServerConfig, RequestMeta, hooks).
//! Server configuration, per-request transport metadata and the deterministic hooks of tests and replay.

use std::fmt;
use std::hash::{BuildHasher, Hasher, RandomState};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use ink_parse::IdFactory;

/// What the front end reports about itself and enforces on every request.
#[derive(Clone, Debug)]
pub struct ServerConfig {
    /// `model` in responses; a request naming another model gets HTTP 404.
    pub served_model_name: String,
    /// The model directory (`/v1/models` root; its `special_tokens_map.json` feeds the parsers). Empty: none.
    pub model_dir: PathBuf,
    /// Prompt plus output token limit.
    pub max_model_len: u32,
    /// Largest `logprobs` / `top_logprobs` a request may ask for.
    pub max_logprobs: u32,
    /// Report cached prompt tokens in `usage.prompt_tokens_details` (config B: off).
    pub prompt_tokens_details: bool,
    /// `system_fingerprint` in responses.
    pub system_fingerprint: String,
    /// The end-of-sequence id (`<|content_model_end_sampling|>`, `generation_config.json`).
    pub eos_token_id: u32,
    /// The model's vocabulary size as the runtime reports it: `/v1/completions` refuses prompt token ids at or above
    /// it with HTTP 400 (vLLM `validate_token_ids`). `None`: no model behind the back end, ids are not checked.
    pub vocab_size: Option<u32>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            served_model_name: "inkling-2-mini".to_owned(),
            model_dir: PathBuf::new(),
            max_model_len: 262_144,
            max_logprobs: 20,
            prompt_tokens_details: false,
            system_fingerprint: "ink-srv".to_owned(),
            eos_token_id: 200_006,
            vocab_size: None,
        }
    }
}

/// What the transport knows about a request besides its body.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RequestMeta {
    /// The `x-session-id` header; empty when absent.
    pub session_header: String,
}

/// Sources of randomness and time, replaceable so that tests and the replay get reproducible output.
#[derive(Clone)]
pub struct Hooks {
    /// The 16 hex characters after `chatcmpl-` / `cmpl-`.
    pub request_hex: Arc<dyn Fn() -> String + Send + Sync>,
    /// A full tool-call id (`chatcmpl-tool-` and 16 hex characters).
    pub tool_call_id: IdFactory,
    /// Unix seconds.
    pub now: Arc<dyn Fn() -> i64 + Send + Sync>,
}

impl fmt::Debug for Hooks {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Hooks").finish_non_exhaustive()
    }
}

/// 16 hex digits from the process's random hasher state; ids need to be unique, not unpredictable.
fn random_hex16() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let mut hasher = RandomState::new().build_hasher();
    hasher.write_u64(COUNTER.fetch_add(1, Ordering::Relaxed));
    format!("{:016x}", hasher.finish())
}

impl Default for Hooks {
    fn default() -> Self {
        Self {
            request_hex: Arc::new(random_hex16),
            tool_call_id: Arc::new(|| format!("chatcmpl-tool-{}", random_hex16())),
            now: Arc::new(|| {
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map_or(0, |d| i64::try_from(d.as_secs()).unwrap_or(i64::MAX))
            }),
        }
    }
}
