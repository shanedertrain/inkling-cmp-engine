// Ported from engine/server/src/api/openai.{h,cpp} @ 4171493, which ports vLLM's
// entrypoints/openai/completion/serving.py (Apache-2.0, Copyright contributors to the vLLM
// project; config B overlay Hiutaky/vllm @ 6e9d1cce).
//! `/v1/completions`: streaming chunks and the full response (vLLM's `completion/serving.py`).

use ink_backend::StepOutput;
use ink_text::{OutputState, StopReason};
use serde_json::{Value, json};

use crate::chat::stop_reason_json;
use crate::config::ServerConfig;
use crate::error::{ApiError, RunError};
use crate::frontend::{CompletionPrepared, DONE_FRAME, Frontend, Response, dump, sse};
use crate::logprobs::{LogprobDecoder, Row, completion_logprobs, no_logprobs, token_ids};
use crate::run::Run;
use crate::stats::{FrontendStats, Inflight};

enum Flow {
    Next,
    Done,
    ClientGone,
}

/// The state of one completions request while its steps arrive.
struct CompletionExec<'a> {
    cfg: &'a ServerConfig,
    p: &'a CompletionPrepared,
    request_id: String,
    created: i64,
    os: OutputState<'a>,
    decoder: LogprobDecoder<'a>,
    num_cached: Option<u32>,
    prompt_ids_sent: bool,
    num_output: usize,
    all_ids: Vec<u32>,
    all_rows: Vec<Row>,
    engine_finish: &'static str,
    stop_reason: Option<StopReason>,
    text_offset: usize,
}

impl CompletionExec<'_> {
    fn chunk(&self, choices: Value) -> Value {
        json!({"id": self.request_id, "object": "text_completion", "created": self.created,
               "model": self.cfg.served_model_name, "choices": choices})
    }

    fn step(&mut self, s: StepOutput, write: &mut dyn FnMut(&str) -> bool) -> Result<Flow, RunError> {
        let p = self.p;
        if self.num_cached.is_none() {
            self.num_cached = s.num_cached_tokens;
        }
        let mut r = self.os.append(&token_ids(&s.tokens)?)?;
        let mut finished = r.finish.is_some();
        if let Some(reason) = r.finish {
            self.engine_finish = reason.as_str();
            self.stop_reason = r.stop_reason.take();
        } else if s.done {
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
        let step_logprobs = if p.opts.stream && p.logprobs.is_some() {
            completion_logprobs(&rows, &mut self.text_offset)
        } else {
            Value::Null
        };
        self.all_ids.extend_from_slice(&new_ids);
        self.all_rows.extend(rows);
        if !p.opts.stream {
            self.num_output += new_ids.len();
            return Ok(if finished { Flow::Done } else { Flow::Next });
        }
        if delta_text.is_empty() && new_ids.is_empty() && self.num_output == 0 && !finished {
            return Ok(Flow::Next);
        }
        let plain = p.logprobs.is_none() && !p.opts.return_token_ids && !p.opts.continuous_usage && !finished;
        let mut choice = json!({
            "index": 0,
            "text": delta_text,
            "logprobs": step_logprobs,
            "finish_reason": if finished { json!(self.engine_finish) } else { Value::Null },
            "stop_reason": if finished { stop_reason_json(self.stop_reason.as_ref()) } else { Value::Null },
        });
        if !plain {
            choice["prompt_token_ids"] = if p.opts.return_token_ids && !self.prompt_ids_sent {
                json!(p.prompt)
            } else {
                Value::Null
            };
            self.prompt_ids_sent |= p.opts.return_token_ids;
            choice["token_ids"] = if p.opts.return_token_ids {
                json!(new_ids)
            } else {
                Value::Null
            };
            choice["weight_versions"] = Value::Null;
        }
        self.num_output += new_ids.len();
        let mut chunk = self.chunk(json!([choice]));
        if !plain {
            if !p.opts.include_usage && finished {
                chunk["system_fingerprint"] = json!(self.cfg.system_fingerprint);
            }
            if p.opts.continuous_usage {
                chunk["usage"] = self.usage();
            }
            chunk["kv_transfer_params"] = Value::Null;
        }
        if !write(&sse(&chunk)) {
            return Ok(Flow::ClientGone);
        }
        Ok(if finished { Flow::Done } else { Flow::Next })
    }

    /// The three token counts of `usage`.
    fn usage(&self) -> Value {
        let (prompt, completion) = (self.p.prompt.len(), self.num_output);
        json!({"prompt_tokens": prompt, "total_tokens": prompt + completion, "completion_tokens": completion})
    }

    /// `usage` of a final frame: the counts, and the cached count when the server reports it.
    fn final_usage(&self, null_details: bool) -> Value {
        let mut usage = self.usage();
        if null_details {
            usage["completion_tokens_details"] = Value::Null;
        }
        if let Some(cached) = self.num_cached.filter(|_| self.cfg.prompt_tokens_details) {
            usage["prompt_tokens_details"] = json!({"cached_tokens": cached});
        }
        usage
    }

    fn end_stream(&self, write: &mut dyn FnMut(&str) -> bool) {
        if self.p.opts.include_usage {
            let mut chunk = self.chunk(json!([]));
            chunk["usage"] = self.final_usage(false);
            chunk["system_fingerprint"] = json!(self.cfg.system_fingerprint);
            write(&sse(&chunk));
        }
        write(DONE_FRAME);
    }

    fn full_body(&self) -> String {
        let p = self.p;
        let mut offset = 0;
        let choice = json!({
            "index": 0,
            "text": self.os.full_text(true),
            "logprobs": if p.logprobs.is_some() { completion_logprobs(&self.all_rows, &mut offset) } else { Value::Null },
            "finish_reason": self.engine_finish,
            "stop_reason": stop_reason_json(self.stop_reason.as_ref()),
            "token_ids": if p.opts.return_token_ids { json!(self.all_ids) } else { Value::Null },
            "prompt_logprobs": null,
            "prompt_token_ids": if p.opts.return_token_ids { json!(p.prompt) } else { Value::Null },
            "routed_experts": null,
            "weight_versions": null,
        });
        dump(&json!({
            "id": self.request_id,
            "object": "text_completion",
            "created": self.created,
            "model": self.cfg.served_model_name,
            "choices": [choice],
            "service_tier": null,
            "system_fingerprint": self.cfg.system_fingerprint,
            "usage": self.final_usage(true),
            "kv_transfer_params": null,
            "ec_transfer_params": null,
            "metrics": null,
        }))
    }
}

impl Frontend {
    pub(crate) async fn execute_completion(
        &self,
        p: CompletionPrepared,
        write: &mut (dyn FnMut(&str) -> bool + Send),
    ) -> Response {
        let request_id = format!("cmpl-{}", (self.hooks.request_hex)());
        let created = (self.hooks.now)();
        let id = self.next_request_id();
        let mut run = Run::start(self.backend.clone(), self.gen_request(id, &p.prompt, &p.opts));
        let inflight = Inflight::start(self.stats());
        FrontendStats::add_tokens(&self.stats().prompt_tokens_total, p.prompt.len());

        let mut exec = CompletionExec {
            cfg: &self.cfg,
            p: &p,
            request_id,
            created,
            os: OutputState::new(&self.tok, p.opts.stop.clone()),
            decoder: LogprobDecoder::new(&self.tok),
            num_cached: None,
            prompt_ids_sent: false,
            num_output: 0,
            all_ids: Vec::new(),
            all_rows: Vec::new(),
            engine_finish: "stop",
            stop_reason: None,
            text_offset: 0,
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
        FrontendStats::add_tokens(&stats.generation_tokens_total, exec.num_output);
        stats.record_e2e(p.t0.elapsed());
        match outcome {
            Ok(Flow::ClientGone) => {
                inflight.client_gone();
                Response::ClientGone
            }
            Err(RunError(message)) => {
                inflight.complete();
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
