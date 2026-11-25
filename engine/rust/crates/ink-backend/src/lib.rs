//! The [`Backend`] trait between the HTTP front end and whatever produces tokens, and its types.
//!
//! Port of `engine/server/src/backend/backend.h`. The front end owns text and the wire format; a backend owns the
//! model: it turns a prompt into a stream of token ids, one `on_step` call per model step. The front end decides when
//! a request is finished and calls [`Backend::finish`], or [`Backend::cancel`] on client disconnect; a backend may
//! overshoot by the tokens of the step in flight.
#![forbid(unsafe_code)]

mod mock;

pub use mock::{Ending, Generator, MockBackend, Script};

/// Sampling controls of one request.
#[derive(Debug, Clone, PartialEq)]
pub struct SamplingParams {
    /// Softmax temperature.
    pub temperature: f64,
    /// Nucleus mass; 1.0 is off.
    pub top_p: f64,
    /// Number of candidates kept; 0 is off.
    pub top_k: u32,
    /// Minimum probability relative to the top token; 0.0 is off.
    pub min_p: f64,
    /// Sampling seed.
    pub seed: Option<i64>,
    /// `None`: no logprobs; `Some(k)`: the sampled token plus the top `k` per position.
    pub num_logprobs: Option<u32>,
}

/// Temperatures below this sample greedily (vLLM `_SAMPLING_EPS`).
const GREEDY_BELOW: f64 = 1e-5;

impl SamplingParams {
    /// Whether these controls sample greedily: the first largest logit, no generator.
    #[must_use]
    pub fn is_greedy(&self) -> bool {
        self.temperature < GREEDY_BELOW
    }
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            seed: None,
            num_logprobs: None,
        }
    }
}

/// One generation request, assigned an id by the front end.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct GenRequest {
    /// Unique per request.
    pub id: u64,
    /// Session identity for prefix reuse and parking; empty means none.
    pub session: String,
    /// Requests with different salts never share a cached prefix.
    pub cache_salt: String,
    /// Prompt token ids.
    pub prompt: Vec<i32>,
    /// Upper bound the backend may use for planning; the front end enforces the exact limit.
    pub max_new_tokens: u32,
    /// Token ids that end the request once at least `min_new_tokens` are out (EOS unless ignored, and the request's
    /// stop token ids): a backend that emits several tokens per step keeps none after the first of them, so the KV it
    /// keeps for the session ends where the client's output ends.
    pub stop_token_ids: Vec<i32>,
    /// Stop token ids count from this many output tokens on.
    pub min_new_tokens: u32,
    /// Token ids the back end never samples while fewer than `min_new_tokens` are out (vLLM's `all_stop_token_ids`:
    /// EOS, also when it is ignored, and the request's stop token ids).
    pub min_tokens_masked: Vec<i32>,
    /// Sampling controls.
    pub sampling: SamplingParams,
    /// Scheduling priority.
    pub priority: i32,
}

/// Logprobs of one generated position.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenLogprobs {
    /// Logprob of the sampled token.
    pub logprob: f32,
    /// Rank of the sampled token; 1 is the top.
    pub rank: u32,
    /// Top-k token ids in rank order.
    pub top_ids: Vec<i32>,
    /// Logprobs matching `top_ids`.
    pub top_logprobs: Vec<f32>,
}

/// What a backend reports after one model step.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct StepOutput {
    /// Tokens produced this step; at least one unless finished or failed.
    pub tokens: Vec<i32>,
    /// One entry per token when logprobs were requested.
    pub logprobs: Vec<TokenLogprobs>,
    /// Set on the first step: prompt tokens served from the prefix cache.
    pub num_cached_tokens: Option<u32>,
    /// Set when the request failed; the message names the cause.
    pub error: Option<String>,
    /// Set with `done` (and `error` unset) when the per-session cap (prompt + output) ended the request; the front
    /// end reports `finish_reason: "length"` (the C++ back end's `error_message` "session cap reached").
    pub session_cap: bool,
    /// The backend will produce nothing more for this request.
    pub done: bool,
}

/// Callback receiving a request's steps in order, from the backend's thread. It must not block for long.
pub type StepFn = Box<dyn FnMut(StepOutput) + Send>;

/// Counters and gauges a backend exposes for `/metrics`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BackendMetrics {
    /// Requests running.
    pub running: u32,
    /// Requests waiting.
    pub waiting: u32,
    /// Sessions parked for prefix reuse.
    pub parked: u32,
    /// KV pages in use.
    pub kv_pages_used: u64,
    /// KV pages in total.
    pub kv_pages_total: u64,
    /// Prompt tokens looked up in the prefix cache.
    pub prefix_cache_queries: u64,
    /// Prompt tokens found in the prefix cache.
    pub prefix_cache_hits: u64,
    /// Prompt tokens processed.
    pub prompt_tokens: u64,
    /// Tokens generated.
    pub generation_tokens: u64,
    /// Draft tokens proposed.
    pub spec_draft_tokens: u64,
    /// Draft tokens accepted.
    pub spec_accepted_tokens: u64,
}

/// Everything that produces tokens for the front end.
pub trait Backend: Send + Sync {
    /// Starts generating; `on_step` is called until a [`StepOutput`] with `done` set.
    fn submit(&self, req: GenRequest, on_step: StepFn);
    /// The front end found the request finished: stop generating it. Its KV may stay parked for prefix reuse.
    fn finish(&self, id: u64);
    /// The client went away: stop as soon as possible.
    fn cancel(&self, id: u64);
    /// Checks a prompt of `tokens` tokens against the back end's fixed admission limits (per-session cap, KV pool)
    /// before it is submitted, so the front end refuses it as a client error (HTTP 400) instead of failing it while
    /// it runs. The default admits every prompt.
    ///
    /// # Errors
    ///
    /// The message of the refusal.
    fn check_prompt(&self, tokens: usize) -> Result<(), String> {
        let _ = tokens;
        Ok(())
    }
    /// Current counters and gauges.
    fn metrics(&self) -> BackendMetrics;
    /// Whether the backend accepts requests.
    fn healthy(&self) -> bool;
}
