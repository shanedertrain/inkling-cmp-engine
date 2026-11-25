// Ported from engine/server/src/api/openai.{h,cpp} @ 4171493, which ports vLLM's entrypoints/openai/ request
// handling (Apache-2.0, Copyright contributors to the vLLM
// project; config B overlay Hiutaky/vllm @ 6e9d1cce).
//! [`Frontend`]: request validation (`prepare_*`) and execution against a back end.

use std::error::Error;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use ink_backend::{Backend, GenRequest};
use ink_parse::{ParseRequest, ParserTables, ToolChoice, special_tokens_from_dir};
use ink_text::pyfmt::{NonAscii, Separators, dumps};
use ink_text::{ChatRenderer, MissingSpecialToken, Specials, Tokenizer};
use serde_json::{Value, json};

use crate::config::{Hooks, RequestMeta, ServerConfig};
use crate::error::ApiError;
use crate::gen_params::{GenParams, int, parse_gen, present, truthy_field};
use crate::stats::FrontendStats;

/// `json.dumps(value, ensure_ascii=False, separators=(",", ":"))`: how every body is written.
pub(crate) fn dump(value: &Value) -> String {
    dumps(value, NonAscii::Keep, Separators::Compact)
}

/// One server-sent-events frame.
pub(crate) fn sse(value: &Value) -> String {
    format!("data: {}\n\n", dump(value))
}

/// The frame that ends every stream.
pub(crate) const DONE_FRAME: &str = "data: [DONE]\n\n";

/// Why a [`Frontend`] could not be built.
#[derive(Debug)]
pub enum FrontendError {
    /// The tokenizer lacks a token the chat renderer needs.
    Tokenizer(MissingSpecialToken),
    /// `special_tokens_map.json` of the model directory could not be read.
    SpecialTokens(ink_parse::SpecialTokensError),
}

impl fmt::Display for FrontendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tokenizer(e) => e.fmt(f),
            Self::SpecialTokens(e) => e.fmt(f),
        }
    }
}

impl Error for FrontendError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Tokenizer(e) => Some(e),
            Self::SpecialTokens(e) => Some(e),
        }
    }
}

/// A validated chat request, rendered to its prompt.
#[derive(Debug)]
pub struct ChatPrepared {
    pub(crate) t0: Instant,
    pub(crate) prompt: Vec<u32>,
    pub(crate) opts: GenParams,
    pub(crate) parse: ParseRequest,
    /// `tool_choice` is `required` or a named tool.
    pub(crate) forced: bool,
    /// `tool_choice` is a named tool.
    pub(crate) named: bool,
    /// Tool calls are reported at all (forced, or `auto` with tools).
    pub(crate) keep_tool_calls: bool,
    /// `parallel_tool_calls` is false: only the first call is kept.
    pub(crate) parallel_false: bool,
    /// `top_logprobs` when `logprobs` is on.
    pub(crate) logprobs: Option<u32>,
}

/// A validated completions request.
#[derive(Debug)]
pub struct CompletionPrepared {
    pub(crate) t0: Instant,
    pub(crate) prompt: Vec<u32>,
    pub(crate) opts: GenParams,
    pub(crate) logprobs: Option<u32>,
}

/// A request that passed validation and is ready to [`execute`](Frontend::execute).
#[derive(Debug)]
pub enum Prepared {
    /// `/v1/chat/completions`.
    Chat(Box<ChatPrepared>),
    /// `/v1/completions`.
    Completion(Box<CompletionPrepared>),
}

impl Prepared {
    /// Whether the response is a server-sent-events stream.
    #[must_use]
    pub fn stream(&self) -> bool {
        match self {
            Self::Chat(p) => p.opts.stream,
            Self::Completion(p) => p.opts.stream,
        }
    }
}

/// How an executed request ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Response {
    /// A non-streaming reply: status 200 and the response, or 500 and an error body.
    Json {
        /// HTTP status.
        status: u16,
        /// JSON body.
        body: String,
    },
    /// The frames, the error frame included, went through the writer.
    Streamed,
    /// The writer reported that the client is gone; the request was cancelled in the back end.
    ClientGone,
}

/// The front end: turns request bodies into back-end requests and back-end steps into responses.
pub struct Frontend {
    pub(crate) tok: Arc<Tokenizer>,
    pub(crate) tables: Arc<ParserTables>,
    pub(crate) cfg: ServerConfig,
    pub(crate) backend: Arc<dyn Backend>,
    pub(crate) hooks: Hooks,
    next_id: AtomicU64,
    stats: FrontendStats,
    created_at: i64,
}

impl fmt::Debug for Frontend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Frontend")
            .field("cfg", &self.cfg)
            .finish_non_exhaustive()
    }
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Frontend>();
};

impl Frontend {
    /// A front end over `tok` and `backend`. The parsers' special-token list is read from
    /// `cfg.model_dir/special_tokens_map.json`; with an empty `model_dir` it is empty.
    ///
    /// # Errors
    /// The tokenizer is not an Inkling-2 tokenizer, or the special-token list cannot be read.
    pub fn new(
        tok: Arc<Tokenizer>,
        cfg: ServerConfig,
        backend: Arc<dyn Backend>,
        hooks: Hooks,
    ) -> Result<Self, FrontendError> {
        ChatRenderer::new(&tok).map_err(FrontendError::Tokenizer)?;
        let specials = if cfg.model_dir.as_os_str().is_empty() {
            Vec::new()
        } else {
            special_tokens_from_dir(&cfg.model_dir).map_err(FrontendError::SpecialTokens)?
        };
        let tables = ParserTables::new(Arc::clone(&tok), &specials);
        let created_at = (hooks.now)();
        Ok(Self {
            tok,
            tables,
            cfg,
            backend,
            hooks,
            next_id: AtomicU64::new(1),
            stats: FrontendStats::default(),
            created_at,
        })
    }

    /// The configuration.
    #[must_use]
    pub fn config(&self) -> &ServerConfig {
        &self.cfg
    }

    /// Whether the back end accepts requests.
    #[must_use]
    pub fn healthy(&self) -> bool {
        self.backend.healthy()
    }

    /// The request counters.
    #[must_use]
    pub fn stats(&self) -> &FrontendStats {
        &self.stats
    }

    pub(crate) fn next_request_id(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::Relaxed)
    }

    /// The `/v1/models` body.
    #[must_use]
    pub fn models_json(&self) -> String {
        dump(&json!({"object": "list", "data": [{
            "id": self.cfg.served_model_name,
            "object": "model",
            "created": self.created_at,
            "owned_by": "inkling-cmp-engine",
            "root": self.cfg.model_dir.to_string_lossy(),
            "parent": null,
            "max_model_len": self.cfg.max_model_len,
        }]}))
    }

    /// The `/metrics` body.
    #[must_use]
    pub fn metrics_text(&self) -> String {
        self.stats.render(&self.cfg.served_model_name, &self.backend.metrics())
    }

    /// Parses and checks a request body shared by both endpoints.
    fn parse_body(&self, text: &str) -> Result<Value, ApiError> {
        let body: Value =
            serde_json::from_str(text).map_err(|e| ApiError::bad_request(format!("JSON decode error: {e}")))?;
        match present(&body, "model") {
            None => {}
            Some(Value::String(model)) if *model == self.cfg.served_model_name => {}
            Some(Value::String(model)) => return Err(ApiError::unknown_model(model)),
            Some(_) => return Err(ApiError::bad_param("model must be a string", "model")),
        }
        Ok(body)
    }

    fn counted<T>(&self, result: Result<T, ApiError>) -> Result<T, ApiError> {
        if result.is_err() {
            self.stats.requests_failed.fetch_add(1, Ordering::Relaxed);
        }
        result
    }

    /// Validates a `/v1/chat/completions` body and renders its prompt.
    ///
    /// # Errors
    /// The HTTP 400 or 404 config B answers with.
    pub fn prepare_chat(&self, body: &str, meta: &RequestMeta) -> Result<Prepared, ApiError> {
        self.stats.requests_total.fetch_add(1, Ordering::Relaxed);
        let t0 = Instant::now();
        let result = self.chat_request(body, meta, t0);
        self.counted(result).map(|p| Prepared::Chat(Box::new(p)))
    }

    fn chat_request(&self, text: &str, meta: &RequestMeta, t0: Instant) -> Result<ChatPrepared, ApiError> {
        let body = self.parse_body(text)?;
        if !body.is_object() {
            return Err(ApiError::bad_request("Request body must be a JSON object"));
        }
        let params = ink_text::resolve_chat_params(&body)?;
        let renderer = ChatRenderer::new(&self.tok).map_err(|e| ApiError::internal(e.to_string()))?;
        let prompt = renderer.render(&params)?;
        let mut opts = parse_gen(&body, prompt.len(), &self.cfg, true)?;
        // after the context-length checks, as the engine's admission refusal (session cap, KV pool) is the later one
        self.backend.check_prompt(prompt.len()).map_err(ApiError::bad_request)?;
        // the header wins over `user`
        if !meta.session_header.is_empty() {
            opts.session.clone_from(&meta.session_header);
        }
        let mut logprobs = None;
        if truthy_field(&body, "logprobs") {
            let k = present(&body, "top_logprobs").map_or(Ok(0), |v| int(v, "top_logprobs"))?;
            if k > i64::from(self.cfg.max_logprobs) {
                return Err(ApiError::bad_request(format!(
                    "Requested sample logprobs of {k}, which is greater than max allowed: {}",
                    self.cfg.max_logprobs
                )));
            }
            // a negative count turns logprobs off, as -1 does in the C++ port
            logprobs = u32::try_from(k).ok();
        } else if let Some(v) = present(&body, "top_logprobs")
            && int(v, "top_logprobs")? > 0
        {
            return Err(ApiError::bad_param(
                "when using `top_logprobs`, `logprobs` must be set to true.",
                "top_logprobs",
            ));
        }
        opts.sampling.num_logprobs = logprobs;
        // check_tool_usage already defaulted tool_choice: "auto" with tools, "none" without
        let tool_choice = params
            .tool_choice
            .as_ref()
            .map_or(ToolChoice::None, ToolChoice::from_value);
        let include_reasoning =
            !(present(&body, "include_reasoning").is_some() && !truthy_field(&body, "include_reasoning"));
        let parse = ParseRequest {
            tools: params
                .tools
                .as_ref()
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default(),
            tool_choice,
            include_reasoning,
        };
        let forced = matches!(parse.tool_choice, ToolChoice::Required | ToolChoice::Named);
        let keep_tool_calls = forced
            || (parse.tool_choice == ToolChoice::Auto && params.tools.as_ref().is_some_and(ink_text::pyjson::truthy));
        Ok(ChatPrepared {
            t0,
            prompt,
            opts,
            named: parse.tool_choice == ToolChoice::Named,
            forced,
            keep_tool_calls,
            // absent: true; null: unset (no restriction); else its truthiness
            parallel_false: present(&body, "parallel_tool_calls").is_some()
                && !truthy_field(&body, "parallel_tool_calls"),
            logprobs,
            parse,
        })
    }

    /// Validates a `/v1/completions` body and encodes its prompt.
    ///
    /// # Errors
    /// The HTTP 400 or 404 config B answers with.
    pub fn prepare_completions(&self, body: &str, meta: &RequestMeta) -> Result<Prepared, ApiError> {
        self.stats.requests_total.fetch_add(1, Ordering::Relaxed);
        let t0 = Instant::now();
        let result = self.completion_request(body, meta, t0);
        self.counted(result).map(|p| Prepared::Completion(Box::new(p)))
    }

    fn completion_request(&self, text: &str, meta: &RequestMeta, t0: Instant) -> Result<CompletionPrepared, ApiError> {
        const NO_PROMPT: &str = "Either prompt or prompt_embeds must be provided and non-empty.";
        let body = self.parse_body(text)?;
        let prompt_value = present(&body, "prompt").ok_or_else(|| ApiError::bad_request(NO_PROMPT))?;
        let encode = |s: &str| -> Result<Vec<u32>, ApiError> {
            self.tok
                .encode(s, Specials::Recognize)
                .map_err(|e| ApiError::bad_request(e.to_string()))
        };
        let prompt = match prompt_value {
            Value::String(s) => encode(s)?,
            Value::Array(items) if items.first().is_some_and(Value::is_number) => items
                .iter()
                .map(|v| {
                    v.as_u64()
                        .and_then(|id| u32::try_from(id).ok())
                        .ok_or_else(|| ApiError::bad_param("prompt token ids must be non-negative integers", "prompt"))
                })
                .collect::<Result<_, _>>()?,
            Value::Array(items) if items.len() == 1 && items[0].is_string() => {
                encode(items[0].as_str().unwrap_or_default())?
            }
            _ => {
                return Err(ApiError::bad_param(
                    "Only a single prompt (string or token ids) per request is supported by this server",
                    "prompt",
                ));
            }
        };
        if prompt.is_empty() {
            return Err(ApiError::bad_request(NO_PROMPT));
        }
        if let (Some(vocab), Some(&max)) = (self.cfg.vocab_size, prompt.iter().max())
            && max >= vocab
        {
            return Err(ApiError::bad_request(format!("Token id {max} is out of vocabulary")));
        }
        let mut opts = parse_gen(&body, prompt.len(), &self.cfg, false)?;
        self.backend.check_prompt(prompt.len()).map_err(ApiError::bad_request)?;
        // quirk kept from the C++ port: on this endpoint `user` wins over the header
        if opts.session.is_empty() {
            opts.session.clone_from(&meta.session_header);
        }
        let mut logprobs = None;
        if let Some(v) = present(&body, "logprobs") {
            let k = int(v, "logprobs")?;
            if k > i64::from(self.cfg.max_logprobs) {
                return Err(ApiError::bad_request(format!(
                    "Requested sample logprobs of {k}, which is greater than max allowed: {}",
                    self.cfg.max_logprobs
                )));
            }
            logprobs = u32::try_from(k).ok();
        }
        opts.sampling.num_logprobs = logprobs;
        if truthy_field(&body, "echo") {
            return Err(ApiError::bad_param("echo is not supported by this server", "echo"));
        }
        Ok(CompletionPrepared {
            t0,
            prompt,
            opts,
            logprobs,
        })
    }

    pub(crate) fn gen_request(&self, id: u64, prompt: &[u32], opts: &GenParams) -> GenRequest {
        GenRequest {
            id,
            session: opts.session.clone(),
            cache_salt: opts.cache_salt.clone(),
            prompt: prompt.iter().map(|&t| i32::try_from(t).unwrap_or(i32::MAX)).collect(),
            max_new_tokens: u32::try_from(opts.stop.max_tokens).unwrap_or(u32::MAX),
            stop_token_ids: opts
                .stop
                .eos_token_id
                .filter(|_| !opts.stop.ignore_eos)
                .into_iter()
                .chain(opts.stop.stop_token_ids.iter().copied())
                .map(|t| i32::try_from(t).unwrap_or(i32::MAX))
                .collect(),
            min_new_tokens: u32::try_from(opts.stop.min_tokens).unwrap_or(u32::MAX),
            // vLLM's all_stop_token_ids: the model's EOS (also with ignore_eos) and the stop token ids
            min_tokens_masked: {
                let mut ids: Vec<i32> = opts
                    .stop
                    .eos_token_id
                    .into_iter()
                    .chain(opts.stop.stop_token_ids.iter().copied())
                    .map(|t| i32::try_from(t).unwrap_or(i32::MAX))
                    .collect();
                ids.sort_unstable();
                ids.dedup();
                ids
            },
            sampling: opts.sampling.clone(),
            priority: 0,
        }
    }

    /// Runs a prepared request. A streaming request writes its frames (`data: ...\n\n`) through `write`, which
    /// returns false once the client is gone; the request is then cancelled in the back end. Dropping the returned
    /// future cancels it the same way.
    pub async fn execute<W>(&self, prepared: Prepared, mut write: W) -> Response
    where
        W: FnMut(&str) -> bool + Send,
    {
        match prepared {
            Prepared::Chat(p) => self.execute_chat(*p, &mut write).await,
            Prepared::Completion(p) => self.execute_completion(*p, &mut write).await,
        }
    }

    /// `prepare_chat` and `execute` in one call; a rejected request becomes a [`Response::Json`] error.
    pub async fn chat<W>(&self, body: &str, meta: &RequestMeta, write: W) -> Response
    where
        W: FnMut(&str) -> bool + Send,
    {
        match self.prepare_chat(body, meta) {
            Ok(p) => self.execute(p, write).await,
            Err(e) => Response::Json {
                status: e.status(),
                body: e.body(),
            },
        }
    }

    /// `prepare_completions` and `execute` in one call.
    pub async fn completions<W>(&self, body: &str, meta: &RequestMeta, write: W) -> Response
    where
        W: FnMut(&str) -> bool + Send,
    {
        match self.prepare_completions(body, meta) {
            Ok(p) => self.execute(p, write).await,
            Err(e) => Response::Json {
                status: e.status(),
                body: e.body(),
            },
        }
    }
}
