// Ported from engine/server/src/api/openai.{h,cpp} @ 4171493, which ports vLLM's
// entrypoints/openai/chat_completion/{serving,stream_processor,message_assembly}.py (Apache-2.0, Copyright
// contributors to the vLLM project; config B overlay Hiutaky/vllm @ 6e9d1cce).
//! `/v1/chat/completions`: streaming chunks (`chat_completion_stream_generator`, `ChatStreamProcessor`) and the
//! full response (`chat_completion_full_generator`).

use std::time::Instant;

use ink_backend::StepOutput;
use ink_parse::{DeltaMessage, InklingParser};
use ink_text::{OutputState, StopReason};
use serde_json::{Value, json};

use crate::config::ServerConfig;
use crate::error::{ApiError, RunError};
use crate::frontend::{ChatPrepared, DONE_FRAME, Frontend, Response, dump, sse};
use crate::logprobs::{LogprobDecoder, Row, chat_logprobs, no_logprobs, token_ids};
use crate::run::Run;
use crate::stats::{FrontendStats, Inflight};

/// What one step of a chat request did to the response.
enum Flow {
    /// Wait for the next step.
    Next,
    /// The request finished.
    Done,
    /// The writer refused a frame.
    ClientGone,
}

pub(crate) fn stop_reason_json(reason: Option<&StopReason>) -> Value {
    match reason {
        Some(StopReason::TokenId(id)) => json!(id),
        Some(StopReason::Text(text)) => json!(text),
        None => Value::Null,
    }
}

/// `usage` of a chat response; `cached` becomes `prompt_tokens_details`.
fn usage(prompt: usize, completion: usize, reasoning: usize, cached: Option<u32>) -> Value {
    let mut usage = json!({
        "prompt_tokens": prompt,
        "total_tokens": prompt + completion,
        "completion_tokens": completion,
        "completion_tokens_details": {"reasoning_tokens": reasoning},
    });
    if let Some(cached) = cached {
        usage["prompt_tokens_details"] = json!({"cached_tokens": cached});
    }
    usage
}

/// The state of one chat request while its steps arrive.
struct ChatExec<'a> {
    cfg: &'a ServerConfig,
    p: &'a ChatPrepared,
    request_id: String,
    created: i64,
    os: OutputState<'a>,
    parser: InklingParser,
    decoder: LogprobDecoder<'a>,
    num_cached: Option<u32>,
    first: bool,
    tools_streamed: bool,
    num_output_tokens: usize,
    all_ids: Vec<u32>,
    all_rows: Vec<Row>,
    engine_finish: &'static str,
    stop_reason: Option<StopReason>,
    t_first: Option<Instant>,
}

impl ChatExec<'_> {
    /// `usage.prompt_tokens_details` source: the cached count when the server reports it.
    fn cached(&self) -> Option<u32> {
        self.cfg.prompt_tokens_details.then_some(self.num_cached).flatten()
    }

    fn reasoning_tokens(&self) -> usize {
        self.parser.count_reasoning_tokens(&self.all_ids)
    }

    fn chunk(&self, choices: Value) -> Value {
        json!({"id": self.request_id, "object": "chat.completion.chunk", "created": self.created,
               "model": self.cfg.served_model_name, "choices": choices})
    }

    fn step(&mut self, s: StepOutput, write: &mut dyn FnMut(&str) -> bool) -> Result<Flow, RunError> {
        let p = self.p;
        if self.first
            && let Some(n) = s.num_cached_tokens
        {
            self.num_cached = Some(n);
        }
        let mut r = self.os.append(&token_ids(&s.tokens)?)?;
        let mut finished = r.finish.is_some();
        if let Some(reason) = r.finish {
            self.engine_finish = reason.as_str();
            self.stop_reason = r.stop_reason.take();
        } else if s.done {
            // the back end ended the request itself: the session cap is "length", anything else "abort"
            finished = true;
            self.engine_finish = if s.session_cap { "length" } else { "abort" };
        }
        let delta_text = self.os.next_text(finished);
        let new_ids = r.new_ids;
        let rows = match p.logprobs {
            Some(k) => {
                let mut lps = s.logprobs;
                lps.resize_with(new_ids.len(), no_logprobs);
                self.decoder.rows(
                    &new_ids,
                    &lps,
                    usize::try_from(k).unwrap_or(usize::MAX).saturating_add(1),
                )?
            }
            None => Vec::new(),
        };
        let hide_metadata = !p.parse.include_reasoning;
        let step_logprobs = match p.logprobs {
            Some(k) if p.opts.stream && !hide_metadata => chat_logprobs(&rows, k),
            _ => Value::Null,
        };
        self.all_ids.extend_from_slice(&new_ids);
        self.all_rows.extend(rows);
        if !new_ids.is_empty() && self.t_first.is_none() {
            self.t_first = Some(Instant::now());
        }

        if !p.opts.stream {
            self.first = false;
            self.num_output_tokens += new_ids.len();
            return Ok(if finished { Flow::Done } else { Flow::Next });
        }
        if self.first {
            self.first = false;
            let delta = json!({"role": "assistant", "content": ""});
            let mut chunk = self.chunk(json!([{"index": 0, "delta": delta, "logprobs": null, "finish_reason": null}]));
            if p.opts.continuous_usage {
                chunk["usage"] = usage(p.prompt.len(), 0, 0, self.cached());
            }
            chunk["prompt_token_ids"] = if p.opts.return_token_ids {
                json!(p.prompt)
            } else {
                Value::Null
            };
            chunk["prompt_text"] = Value::Null;
            if !write(&sse(&chunk)) {
                return Ok(Flow::ClientGone);
            }
        }

        // ChatStreamProcessor.step
        let mut delta: Option<DeltaMessage> =
            if delta_text.is_empty() && new_ids.is_empty() && self.num_output_tokens == 0 {
                Some(DeltaMessage::default())
            } else {
                self.parser.parse_delta(&delta_text, &new_ids, &p.prompt, finished)
            };
        let mut drop_delta = false;
        if let Some(d) = delta.as_mut()
            && !d.tool_calls.is_empty()
        {
            if p.keep_tool_calls {
                self.tools_streamed = true;
            } else {
                d.tool_calls.clear();
                drop_delta = !d.content.has_text() && !d.reasoning.has_text();
            }
        }
        if drop_delta {
            delta = None;
        }
        self.num_output_tokens += new_ids.len();
        if p.parallel_false
            && let Some(d) = delta.as_mut()
        {
            d.tool_calls.retain(|call| call.index == 0);
        }
        let include_token_ids = p.opts.return_token_ids && !hide_metadata;
        let delta = match delta {
            Some(d) => d,
            None if !finished && step_logprobs.is_null() && !include_token_ids => return Ok(Flow::Next),
            None => DeltaMessage::default(),
        };
        let token_ids = if include_token_ids { json!(new_ids) } else { Value::Null };
        let mut choice = json!({"index": 0, "delta": delta.to_value(), "logprobs": step_logprobs});
        if finished {
            let mut reason = self.engine_finish;
            if self.tools_streamed && reason == "stop" && !p.named {
                reason = "tool_calls";
            }
            choice["finish_reason"] = json!(reason);
            choice["stop_reason"] = stop_reason_json(self.stop_reason.as_ref());
            choice["token_ids"] = token_ids;
            choice["weight_versions"] = Value::Null;
        } else {
            choice["finish_reason"] = Value::Null;
            choice["token_ids"] = token_ids;
        }
        let mut chunk = self.chunk(json!([choice]));
        if !p.opts.include_usage && finished {
            chunk["system_fingerprint"] = json!(self.cfg.system_fingerprint);
        }
        if p.opts.continuous_usage {
            chunk["usage"] = usage(p.prompt.len(), self.num_output_tokens, self.reasoning_tokens(), None);
        }
        chunk["kv_transfer_params"] = Value::Null;
        if !write(&sse(&chunk)) {
            return Ok(Flow::ClientGone);
        }
        Ok(if finished { Flow::Done } else { Flow::Next })
    }

    /// The usage chunk (when asked for) and `[DONE]`.
    fn end_stream(&self, write: &mut dyn FnMut(&str) -> bool) {
        if self.p.opts.include_usage {
            let mut chunk = self.chunk(json!([]));
            chunk["usage"] = usage(
                self.p.prompt.len(),
                self.num_output_tokens,
                self.reasoning_tokens(),
                self.cached(),
            );
            chunk["system_fingerprint"] = json!(self.cfg.system_fingerprint);
            write(&sse(&chunk));
        }
        write(DONE_FRAME);
    }

    /// `chat_completion_full_generator`: the whole response body.
    fn full_body(&mut self) -> String {
        let p = self.p;
        let text = self.os.full_text(true).to_owned();
        let parsed = self.parser.parse(&text);
        let logprobs = match p.logprobs {
            Some(k) if p.parse.include_reasoning => chat_logprobs(&self.all_rows, k),
            _ => Value::Null,
        };
        let reasoning = if p.parse.include_reasoning {
            parsed.reasoning
        } else {
            None
        };
        let mut content = parsed.content;
        let mut tool_calls: Vec<Value> = Vec::new();
        if p.keep_tool_calls {
            if p.forced {
                content = Some(content.unwrap_or_default());
            }
            let mut calls = parsed.tool_calls.unwrap_or_default();
            if p.parallel_false {
                calls.truncate(1);
            }
            tool_calls = calls
                .into_iter()
                .map(
                    |c| json!({"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}),
                )
                .collect();
        }
        let mut finish_reason = self.engine_finish;
        if !tool_calls.is_empty() && finish_reason == "stop" && !p.named {
            finish_reason = "tool_calls";
        }
        let mut message = json!({"role": "assistant", "content": content, "refusal": null, "audio": null,
                                 "function_call": null});
        if !tool_calls.is_empty() {
            message["tool_calls"] = Value::Array(tool_calls);
        }
        message["reasoning"] = json!(reasoning);
        let choice = json!({
            "index": 0,
            "message": message,
            "logprobs": logprobs,
            "finish_reason": finish_reason,
            "stop_reason": stop_reason_json(self.stop_reason.as_ref()),
            "token_ids": if p.opts.return_token_ids && p.parse.include_reasoning { json!(self.all_ids) } else { Value::Null },
            "routed_experts": null,
            "weight_versions": null,
        });
        dump(&json!({
            "id": self.request_id,
            "object": "chat.completion",
            "created": self.created,
            "model": self.cfg.served_model_name,
            "choices": [choice],
            "service_tier": null,
            "system_fingerprint": self.cfg.system_fingerprint,
            "usage": usage(p.prompt.len(), self.all_ids.len(), self.reasoning_tokens(), self.cached()),
            "prompt_logprobs": null,
            "prompt_token_ids": if p.opts.return_token_ids { json!(p.prompt) } else { Value::Null },
            "prompt_text": null,
            "kv_transfer_params": null,
            "ec_transfer_params": null,
            "metrics": null,
        }))
    }
}

impl Frontend {
    pub(crate) async fn execute_chat(&self, p: ChatPrepared, write: &mut (dyn FnMut(&str) -> bool + Send)) -> Response {
        let request_id = format!("chatcmpl-{}", (self.hooks.request_hex)());
        let created = (self.hooks.now)();
        let id = self.next_request_id();
        let mut run = Run::start(self.backend.clone(), self.gen_request(id, &p.prompt, &p.opts));
        let inflight = Inflight::start(self.stats());
        FrontendStats::add_tokens(&self.stats().prompt_tokens_total, p.prompt.len());

        let mut exec = ChatExec {
            cfg: &self.cfg,
            p: &p,
            request_id,
            created,
            os: OutputState::new(&self.tok, p.opts.stop.clone()),
            parser: InklingParser::new(
                self.tables.clone(),
                p.parse.clone(),
                Some(self.hooks.tool_call_id.clone()),
            ),
            decoder: LogprobDecoder::new(&self.tok),
            num_cached: None,
            first: true,
            tools_streamed: false,
            num_output_tokens: 0,
            all_ids: Vec::new(),
            all_rows: Vec::new(),
            engine_finish: "stop",
            stop_reason: None,
            t_first: None,
        };
        let outcome: Result<Flow, RunError> = loop {
            let Some(s) = run.next().await else {
                break Err(RunError(
                    "the back end ended the request without a final step".to_owned(),
                ));
            };
            if let Some(e) = s.error {
                break Err(RunError(e));
            }
            match exec.step(s, write) {
                Ok(Flow::Next) => {}
                other => break other,
            }
        };
        let client_gone = matches!(outcome, Ok(Flow::ClientGone));
        run.stop(client_gone);

        let stats = self.stats();
        FrontendStats::add_tokens(&stats.generation_tokens_total, exec.num_output_tokens);
        if let Some(n) = exec.num_cached {
            stats
                .cached_prompt_tokens_total
                .fetch_add(i64::from(n), std::sync::atomic::Ordering::Relaxed);
        }
        if let Some(t) = exec.t_first {
            stats.record_ttft(t.duration_since(p.t0));
        }
        stats.record_e2e(p.t0.elapsed());
        match outcome {
            Ok(Flow::ClientGone) => {
                inflight.client_gone();
                Response::ClientGone
            }
            Err(RunError(message)) => {
                inflight.complete();
                stats.requests_failed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let body = ApiError::internal(message).body();
                if p.opts.stream {
                    write(&format!("data: {body}\n\n"));
                    write(DONE_FRAME);
                    Response::Streamed
                } else {
                    Response::Json { status: 500, body }
                }
            }
            Ok(Flow::Next | Flow::Done) => {
                inflight.complete();
                if p.opts.stream {
                    exec.end_stream(write);
                    Response::Streamed
                } else {
                    Response::Json {
                        status: 200,
                        body: exec.full_body(),
                    }
                }
            }
        }
    }
}
