//! Ports of the transport-free tests of `engine/server/tests/test_api.cpp` (the HTTP ones are package F5's), and the
//! behaviours the C++ port found against config B. They run on a small byte-level tokenizer and need no data.

mod common;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use common::{Fixture, encode, fixture, reply, special};
use ink_api::{Frontend, RequestMeta, Response, ServerConfig};
use ink_backend::{Backend, BackendMetrics, Ending, GenRequest, Script, StepFn, StepOutput};
use serde_json::{Value, json};

const BODY: &str = r#"{"model":"inkling-2-mini","messages":[{"role":"user","content":"Hi"}],"max_tokens":100"#;

fn body(extra: &str) -> String {
    format!("{BODY}{extra}}}")
}

/// Runs a streaming request and returns the response and its frames.
async fn stream(fe: &Frontend, text: &str) -> (Response, Vec<String>) {
    let mut frames = Vec::new();
    let r = fe
        .chat(text, &RequestMeta::default(), |f| {
            frames.push(f.to_owned());
            true
        })
        .await;
    (r, frames)
}

fn json_of(frame: &str) -> Value {
    serde_json::from_str(
        frame
            .strip_prefix("data: ")
            .and_then(|f| f.strip_suffix("\n\n"))
            .expect("an SSE frame"),
    )
    .expect("a JSON frame")
}

fn json_body(r: Response) -> Value {
    match r {
        Response::Json { status: 200, body } => serde_json::from_str(&body).expect("JSON body"),
        other => panic!("expected a 200 JSON response, got {other:?}"),
    }
}

fn keys(v: &Value) -> Vec<&str> {
    v.as_object().expect("an object").keys().map(String::as_str).collect()
}

#[tokio::test]
async fn api_chat_stream_and_full() {
    let Fixture { tok, backend, fe } = fixture();
    backend.push(reply(&tok, "Think.", "Hello there.", 2));
    let (r, frames) = stream(&fe, &body(r#","stream":true,"stream_options":{"include_usage":true}"#)).await;
    assert_eq!(r, Response::Streamed);
    assert!(frames.len() >= 4);
    assert_eq!(frames.last().map(String::as_str), Some("data: [DONE]\n\n"));
    let (mut reasoning, mut content, mut finish, mut usage) = (String::new(), String::new(), None, None);
    for frame in &frames[..frames.len() - 1] {
        let j = json_of(frame);
        if j["choices"].as_array().expect("choices").is_empty() {
            usage = Some(j["usage"].clone());
            continue;
        }
        let delta = &j["choices"][0]["delta"];
        reasoning.push_str(delta["reasoning"].as_str().unwrap_or_default());
        content.push_str(delta["content"].as_str().unwrap_or_default());
        if let Some(f) = j["choices"][0]["finish_reason"].as_str() {
            finish = Some(f.to_owned());
        }
    }
    assert_eq!((reasoning.as_str(), content.as_str()), ("Think.", "Hello there."));
    assert_eq!(finish.as_deref(), Some("stop"));
    let usage = usage.expect("a usage chunk");
    let prompt_tokens = backend.submitted()[0].prompt.len();
    assert_eq!(usage["prompt_tokens"], prompt_tokens);
    assert!(
        usage["completion_tokens_details"]["reasoning_tokens"]
            .as_u64()
            .expect("count")
            > 0
    );
    assert!(
        usage.get("prompt_tokens_details").is_none(),
        "config B reports no cached-token detail"
    );

    backend.push(reply(&tok, "Think.", "Hello there.", 2));
    let full = json_body(fe.chat(&body(""), &RequestMeta::default(), |_| true).await);
    assert_eq!(full["choices"][0]["message"]["content"], "Hello there.");
    assert_eq!(full["choices"][0]["message"]["reasoning"], "Think.");
    assert_eq!(full["object"], "chat.completion");
}

#[tokio::test]
async fn api_errors() {
    let Fixture { fe, backend, .. } = fixture();
    let status = |r: Response| match r {
        Response::Json { status, .. } => status,
        other => panic!("expected an error response, got {other:?}"),
    };
    let meta = RequestMeta::default();
    let chat = |text: &'static str| fe.chat(text, &meta, |_| true);
    assert_eq!(
        status(chat(r#"{"model":"other","messages":[{"role":"user","content":"x"}]}"#).await),
        404
    );
    assert_eq!(status(chat(r#"{"messages":[]}"#).await), 400);
    assert_eq!(
        status(chat(r#"{"messages":[{"role":"user","content":"x"}],"tool_choice":"required"}"#).await),
        400
    );
    assert_eq!(status(chat("{not json").await), 400);
    assert_eq!(
        status(chat(r#"{"messages":[{"role":"user","content":"x"}],"max_tokens":300000}"#).await),
        400
    );
    assert_eq!(
        status(fe.completions(r#"{"prompt":"x","logprobs":50}"#, &meta, |_| true).await),
        400
    );
    // tool and system message parts must be text
    for role in ["system", "tool"] {
        let text = format!(
            r#"{{"messages":[{{"role":"{role}","content":[{{"type":"image_url","image_url":{{"url":"x"}}}}]}},{{"role":"user","content":"x"}}]}}"#
        );
        assert_eq!(status(fe.chat(&text, &meta, |_| true).await), 400, "{role}");
    }
    assert!(backend.submitted().is_empty(), "nothing reached the back end");
    let error: Value = serde_json::from_str(
        &fe.prepare_chat(r#"{"messages":[]}"#, &meta)
            .expect_err("rejected")
            .body(),
    )
    .expect("error body");
    assert_eq!(keys(&error["error"]), ["message", "type", "param", "code"]);
    assert_eq!(error["error"]["type"], "BadRequestError");
    assert_eq!(error["error"]["param"], "messages");
}

#[tokio::test]
async fn api_back_end_ending_the_request_without_a_stop_is_an_abort() {
    let Fixture { tok, backend, fe } = fixture();
    backend.push(Script {
        groups: vec![
            vec![special(&tok, "<|message_model|>"), special(&tok, "<|content_text|>")],
            encode(&tok, "partial"),
        ],
        ..Script::default()
    });
    let j = json_body(fe.chat(&body(""), &RequestMeta::default(), |_| true).await);
    assert_eq!(j["choices"][0]["finish_reason"], "abort");
    assert_eq!(j["choices"][0]["message"]["content"], "partial");
}

#[tokio::test]
async fn api_session_cap_with_an_empty_final_event_finishes_length() {
    let Fixture { tok, backend, fe } = fixture();
    backend.push(Script {
        groups: vec![
            vec![special(&tok, "<|message_model|>"), special(&tok, "<|content_text|>")],
            encode(&tok, "cut"),
        ],
        ending: Ending::SessionCap,
        ..Script::default()
    });
    let j = json_body(fe.chat(&body(""), &RequestMeta::default(), |_| true).await);
    assert_eq!(j["choices"][0]["finish_reason"], "length");
    assert_eq!(j["choices"][0]["message"]["content"], "cut");

    backend.push(Script {
        groups: vec![encode(&tok, "x")],
        ending: Ending::SessionCap,
        ..Script::default()
    });
    let (_, frames) = stream(&fe, &body(r#","stream":true"#)).await;
    let finishes: Vec<_> = frames[..frames.len() - 1]
        .iter()
        .filter_map(|f| json_of(f)["choices"][0]["finish_reason"].as_str().map(str::to_owned))
        .collect();
    assert_eq!(finishes, ["length"]);
}

/// A back end that answers every request with one event: the last tokens, `done` and `session_cap` together, as
/// the Rust scheduler reports the session cap.
struct CapWithTokens(Vec<i32>);

impl Backend for CapWithTokens {
    fn submit(&self, _: GenRequest, mut on_step: StepFn) {
        on_step(StepOutput {
            tokens: self.0.clone(),
            done: true,
            session_cap: true,
            ..StepOutput::default()
        });
    }
    fn finish(&self, _: u64) {}
    fn cancel(&self, _: u64) {}
    fn metrics(&self) -> BackendMetrics {
        BackendMetrics::default()
    }
    fn healthy(&self) -> bool {
        true
    }
}

fn frontend_over(backend: Arc<dyn Backend>) -> (Arc<ink_text::Tokenizer>, Frontend) {
    let tok = common::mini_tokenizer();
    let fe = Frontend::new(
        Arc::clone(&tok),
        ServerConfig::default(),
        backend,
        common::fixed_hooks(),
    )
    .expect("front end");
    (tok, fe)
}

#[tokio::test]
async fn api_session_cap_on_the_event_with_the_last_tokens_finishes_length() {
    let tok = common::mini_tokenizer();
    let mut tokens = vec![special(&tok, "<|message_model|>"), special(&tok, "<|content_text|>")];
    tokens.extend(encode(&tok, "last words"));
    let (_, fe) = frontend_over(Arc::new(CapWithTokens(tokens)));

    let j = json_body(fe.chat(&body(""), &RequestMeta::default(), |_| true).await);
    assert_eq!(j["choices"][0]["finish_reason"], "length");
    assert_eq!(j["choices"][0]["message"]["content"], "last words");
    assert_eq!(j["usage"]["completion_tokens"], 12);

    let (_, frames) = stream(&fe, &body(r#","stream":true,"stream_options":{"include_usage":true}"#)).await;
    let last = json_of(&frames[frames.len() - 3]);
    assert_eq!(last["choices"][0]["finish_reason"], "length");
    assert_eq!(last["choices"][0]["delta"]["content"], "last words");
    assert_eq!(json_of(&frames[frames.len() - 2])["usage"]["completion_tokens"], 12);
}

#[tokio::test]
async fn api_eos_is_checked_before_the_length_cap() {
    // max_tokens 3, and the third token is EOS: finish_reason stop, not length
    let Fixture { tok, backend, fe } = fixture();
    let eos = special(&tok, "<|content_model_end_sampling|>");
    backend.push(Script {
        groups: vec![vec![
            special(&tok, "<|message_model|>"),
            special(&tok, "<|content_text|>"),
            eos,
        ]],
        ..Script::default()
    });
    let j = json_body(
        fe.chat(
            r#"{"messages":[{"role":"user","content":"Hi"}],"max_tokens":3}"#,
            &RequestMeta::default(),
            |_| true,
        )
        .await,
    );
    assert_eq!(j["choices"][0]["finish_reason"], "stop");
    assert_eq!(j["usage"]["completion_tokens"], 3);
}

#[tokio::test]
async fn api_first_chunk_key_order_with_continuous_usage() {
    let Fixture { tok, backend, fe } = fixture();
    backend.push(reply(&tok, "R.", "A.", 3));
    let (_, frames) = stream(
        &fe,
        &body(r#","stream":true,"stream_options":{"include_usage":true,"continuous_usage_stats":true}"#),
    )
    .await;
    let first = json_of(&frames[0]);
    assert_eq!(
        keys(&first),
        [
            "id",
            "object",
            "created",
            "model",
            "choices",
            "usage",
            "prompt_token_ids",
            "prompt_text"
        ]
    );
    assert_eq!(
        keys(&first["choices"][0]),
        ["index", "delta", "logprobs", "finish_reason"]
    );
    assert_eq!(first["usage"]["completion_tokens_details"]["reasoning_tokens"], 0);
    let second = json_of(&frames[1]);
    assert_eq!(
        keys(&second),
        [
            "id",
            "object",
            "created",
            "model",
            "choices",
            "usage",
            "kv_transfer_params"
        ]
    );
    let finish = frames
        .iter()
        .map(|f| f.as_str())
        .filter(|f| *f != "data: [DONE]\n\n")
        .map(json_of)
        .find(|j| j["choices"][0]["finish_reason"].is_string())
        .expect("a finish chunk");
    assert_eq!(
        keys(&finish["choices"][0]),
        [
            "index",
            "delta",
            "logprobs",
            "finish_reason",
            "stop_reason",
            "token_ids",
            "weight_versions"
        ]
    );
}

fn tool_call_script(tok: &ink_text::Tokenizer, call: &str, group: usize, ending: bool) -> Script {
    let mut all = vec![special(tok, "<|message_model|>")];
    all.extend(encode(tok, "get_weather"));
    all.push(special(tok, "<|content_invoke_tool_json|>"));
    all.extend(encode(tok, call));
    if ending {
        all.push(special(tok, "<|end_message|>"));
    }
    all.push(special(tok, "<|content_model_end_sampling|>"));
    Script {
        groups: all.chunks(group).map(<[i32]>::to_vec).collect(),
        ..Script::default()
    }
}

const TOOLS: &str = r#","tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}}}}}]"#;
const CALL: &str = r#"{"name":"get_weather","args":{"city":"Paris","days":"3"}}"#;

#[tokio::test]
async fn api_tool_call_streaming_and_full_agree() {
    let Fixture { tok, backend, fe } = fixture();
    backend.push(tool_call_script(&tok, CALL, 4, true));
    let full = json_body(fe.chat(&body(TOOLS), &RequestMeta::default(), |_| true).await);
    let message = &full["choices"][0]["message"];
    let call = &message["tool_calls"][0];
    assert_eq!(call["function"]["name"], "get_weather");
    assert_eq!(call["function"]["arguments"], r#"{"city": "Paris", "days": 3}"#);
    assert_eq!(call["id"], "chatcmpl-tool-0000000000000001");
    assert_eq!(full["choices"][0]["finish_reason"], "tool_calls");
    assert_eq!(
        keys(message),
        [
            "role",
            "content",
            "refusal",
            "audio",
            "function_call",
            "tool_calls",
            "reasoning"
        ]
    );

    backend.push(tool_call_script(&tok, CALL, 4, true));
    let (_, frames) = stream(&fe, &body(&format!("{TOOLS},\"stream\":true"))).await;
    let chunks: Vec<Value> = frames[..frames.len() - 1].iter().map(|f| json_of(f)).collect();
    let mut arguments = String::new();
    for c in &chunks {
        let delta = &c["choices"][0]["delta"];
        assert!(
            delta
                .get("tool_calls")
                .is_none_or(|t| !t.as_array().expect("array").is_empty()),
            "an empty tool_calls list is never sent"
        );
        for call in delta["tool_calls"].as_array().into_iter().flatten() {
            arguments.push_str(call["function"]["arguments"].as_str().unwrap_or_default());
        }
    }
    assert_eq!(
        chunks.last().expect("a finish chunk")["choices"][0]["finish_reason"],
        "tool_calls"
    );
    assert!(arguments.contains("Paris"), "streamed arguments: {arguments}");
}

#[tokio::test]
async fn api_last_tool_call_arguments_become_empty_at_stream_end() {
    // the call is cut off after its name and the request ends: the finish chunk carries `"arguments": ""`
    let Fixture { tok, backend, fe } = fixture();
    backend.push(tool_call_script(&tok, r#"{"name":"get_weather""#, 1000, false));
    let (_, frames) = stream(&fe, &body(&format!("{TOOLS},\"stream\":true"))).await;
    let finish = json_of(&frames[frames.len() - 2]);
    assert_eq!(finish["choices"][0]["finish_reason"], "tool_calls");
    let calls = finish["choices"][0]["delta"]["tool_calls"]
        .as_array()
        .expect("tool calls");
    assert_eq!(calls[0]["function"], json!({"name": "get_weather", "arguments": ""}));
}

#[tokio::test]
async fn api_completions() {
    let Fixture { tok, backend, fe } = fixture();
    backend.push(Script {
        groups: encode(&tok, " Paris. And").into_iter().map(|t| vec![t]).collect(),
        ..Script::default()
    });
    let j = json_body(
        fe.completions(
            r#"{"prompt":"The capital of France is","max_tokens":10,"stop":["."]}"#,
            &RequestMeta::default(),
            |_| true,
        )
        .await,
    );
    assert_eq!(j["choices"][0]["text"], " Paris");
    assert_eq!(j["choices"][0]["finish_reason"], "stop");
    assert_eq!(j["choices"][0]["stop_reason"], ".");
    assert_eq!(
        j["usage"],
        json!({"prompt_tokens": 24, "total_tokens": 31, "completion_tokens": 7, "completion_tokens_details": null})
    );
}

/// Prompt token ids at or above the runtime's vocabulary are refused before anything reaches the back end.
#[tokio::test]
async fn api_completions_refuse_ids_outside_the_vocabulary() {
    let Fixture { tok, backend, .. } = fixture();
    let cfg = ServerConfig {
        vocab_size: Some(1000),
        ..ServerConfig::default()
    };
    let fe = Frontend::new(Arc::clone(&tok), cfg, backend.clone(), common::fixed_hooks()).expect("front end");
    let meta = RequestMeta::default();
    let r = fe
        .completions(r#"{"prompt":[5,1000,7],"max_tokens":3}"#, &meta, |_| true)
        .await;
    match r {
        Response::Json { status, body } => {
            assert_eq!(status, 400, "{body}");
            let j: Value = serde_json::from_str(&body).expect("JSON");
            assert_eq!(j["error"]["message"], "Token id 1000 is out of vocabulary", "{j}");
        }
        other => panic!("expected a 400, got {other:?}"),
    }
    assert!(backend.submitted().is_empty());
    backend.push(Script {
        groups: vec![vec![65]],
        ..Script::default()
    });
    let j = json_body(
        fe.completions(r#"{"prompt":[5,999,7],"max_tokens":1}"#, &meta, |_| true)
            .await,
    );
    assert_eq!(j["usage"]["prompt_tokens"], 3);
}

#[tokio::test]
async fn api_completions_stream_with_logprobs() {
    let Fixture { tok, backend, fe } = fixture();
    let ids = encode(&tok, "ab");
    backend.push(Script {
        groups: ids.iter().map(|&t| vec![t]).collect(),
        logprobs: ids
            .iter()
            .map(|&t| {
                vec![ink_backend::TokenLogprobs {
                    logprob: -0.5,
                    rank: 1,
                    top_ids: vec![t, 66],
                    top_logprobs: vec![-0.5, -2.0],
                }]
            })
            .collect(),
        ..Script::default()
    });
    let mut frames = Vec::new();
    fe.completions(
        r#"{"prompt":"x","max_tokens":2,"logprobs":2,"stream":true}"#,
        &RequestMeta::default(),
        |f| {
            frames.push(f.to_owned());
            true
        },
    )
    .await;
    let chunks: Vec<Value> = frames[..frames.len() - 1].iter().map(|f| json_of(f)).collect();
    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0]["choices"][0]["logprobs"]["text_offset"], json!([0]));
    assert_eq!(chunks[1]["choices"][0]["logprobs"]["text_offset"], json!([1]));
    assert_eq!(
        chunks[0]["choices"][0]["logprobs"]["top_logprobs"],
        json!([{"a": -0.5, "B": -2.0}])
    );
    assert_eq!(chunks[1]["choices"][0]["finish_reason"], "length");
    assert_eq!(
        keys(&chunks[0]),
        ["id", "object", "created", "model", "choices", "kv_transfer_params"]
    );
}

#[tokio::test]
async fn api_prompt_token_details_are_reported_when_enabled() {
    let tok = common::mini_tokenizer();
    let backend = Arc::new(ink_backend::MockBackend::new());
    let cfg = ServerConfig {
        prompt_tokens_details: true,
        eos_token_id: u32::try_from(special(&tok, "<|content_model_end_sampling|>")).expect("id"),
        ..ServerConfig::default()
    };
    let fe = Frontend::new(Arc::clone(&tok), cfg, backend.clone(), common::fixed_hooks()).expect("front end");
    backend.push(reply(&tok, "R.", "A.", 4));
    let j = json_body(fe.chat(&body(""), &RequestMeta::default(), |_| true).await);
    assert_eq!(j["usage"]["prompt_tokens_details"], json!({"cached_tokens": 4}));
    assert_eq!(
        keys(&j["usage"]),
        [
            "prompt_tokens",
            "total_tokens",
            "completion_tokens",
            "completion_tokens_details",
            "prompt_tokens_details"
        ]
    );
}

#[tokio::test]
async fn api_session_header_reaches_the_back_end() {
    let Fixture { tok, backend, fe } = fixture();
    backend.push(reply(&tok, "R.", "A.", 8));
    let meta = RequestMeta {
        session_header: "s1".to_owned(),
    };
    fe.chat(&body(r#","user":"someone""#), &meta, |_| true).await;
    assert_eq!(backend.submitted()[0].session, "s1");
}

#[tokio::test]
async fn api_a_writer_that_refuses_cancels_the_request() {
    let Fixture { tok, backend, fe } = fixture();
    backend.push(Script {
        step_delay: Duration::from_millis(5),
        ..reply(&tok, "A long thought that keeps going.", &"z".repeat(300), 1)
    });
    let mut frames = 0;
    let r = fe
        .chat(
            &body(r#","stream":true,"max_tokens":2000"#),
            &RequestMeta::default(),
            |_| {
                frames += 1;
                frames < 4
            },
        )
        .await;
    assert_eq!(r, Response::ClientGone);
    assert_eq!(backend.cancels(), 1);
    assert!(
        fe.metrics_text()
            .contains("ink:requests_cancelled_total{model_name=\"inkling-2-mini\"} 1")
    );
}

#[tokio::test]
async fn api_dropping_the_future_cancels_the_request() {
    let Fixture { tok, backend, fe } = fixture();
    backend.push(Script {
        step_delay: Duration::from_millis(50),
        ..reply(&tok, "Slow thought.", "Slow answer.", 1)
    });
    let seen = AtomicUsize::new(0);
    let text = body(r#","stream":true"#);
    let meta = RequestMeta::default();
    let run = fe.chat(&text, &meta, |_| {
        seen.fetch_add(1, Ordering::SeqCst);
        true
    });
    // give up while the request is in flight: the select drops `run` once two frames were written
    tokio::select! {
        _ = run => panic!("the request finished before it was dropped"),
        () = async { while seen.load(Ordering::SeqCst) < 2 { tokio::task::yield_now().await; } } => {}
    }
    assert_eq!(backend.cancels(), 1);
}

#[tokio::test]
async fn api_back_end_errors_are_http_500() {
    let Fixture { backend, fe, .. } = fixture();
    backend.push(Script {
        ending: Ending::Error,
        ..Script::default()
    });
    match fe.chat(&body(""), &RequestMeta::default(), |_| true).await {
        Response::Json { status: 500, body } => {
            assert!(body.contains("mock back end error") && body.contains("InternalServerError"))
        }
        other => panic!("expected 500, got {other:?}"),
    }
}

#[tokio::test]
async fn api_models_and_metrics() {
    let Fixture { fe, .. } = fixture();
    let models: Value = serde_json::from_str(&fe.models_json()).expect("JSON");
    assert_eq!(models["data"][0]["id"], "inkling-2-mini");
    assert_eq!(
        keys(&models["data"][0]),
        ["id", "object", "created", "owned_by", "root", "parent", "max_model_len"]
    );
    let metrics = fe.metrics_text();
    assert!(metrics.contains("vllm:num_requests_running{model_name=\"inkling-2-mini\"} 0\n"));
    assert!(metrics.contains("vllm:time_to_first_token_seconds_bucket{le=\"+Inf\",model_name=\"inkling-2-mini\"} 0\n"));
}

#[test]
fn execute_futures_are_send() {
    fn assert_send<T: Send>(_: &T) {}
    let Fixture { fe, .. } = fixture();
    let meta = RequestMeta::default();
    assert_send(&fe.chat("{}", &meta, |_| true));
}

/// Stop ids, `min_tokens` and the ids masked before it that the back end saw, per request.
type SeenStops = Arc<std::sync::Mutex<Vec<(Vec<i32>, u32, Vec<i32>)>>>;

/// The back end gets the token ids that end the request (EOS unless ignored, then `stop_token_ids`) and `min_tokens`,
/// so a back end that emits several tokens per step can stop where the front end does, and the ids it masks before
/// `min_tokens` (vLLM's `all_stop_token_ids`: EOS also when ignored, and `stop_token_ids`).
#[tokio::test]
async fn api_stop_token_ids_reach_the_back_end() {
    let Fixture { tok, backend, fe } = fixture();
    let seen: SeenStops = Arc::default();
    let eos = special(&tok, "<|content_model_end_sampling|>");
    {
        let seen = Arc::clone(&seen);
        backend.set_generator(move |req| {
            seen.lock().expect("lock").push((
                req.stop_token_ids.clone(),
                req.min_new_tokens,
                req.min_tokens_masked.clone(),
            ));
            Script {
                groups: vec![vec![eos]],
                ..Script::default()
            }
        });
    }
    for extra in [
        r#","stop_token_ids":[7,9],"min_tokens":2"#,
        r#","stop_token_ids":[7],"ignore_eos":true"#,
    ] {
        let _ = fe.chat(&body(extra), &RequestMeta::default(), |_| true).await;
    }
    assert_eq!(
        *seen.lock().expect("lock"),
        [(vec![eos, 7, 9], 2, vec![7, 9, eos]), (vec![7], 0, vec![7, eos])]
    );
}

/// `min_tokens` below 0 or above the request's `max_tokens` is refused with vLLM's `SamplingParams` messages; nothing
/// reaches the back end.
#[tokio::test]
async fn api_min_tokens_out_of_range_is_refused() {
    let Fixture { backend, fe, .. } = fixture();
    for (extra, want) in [
        (
            r#","min_tokens":-1"#,
            "min_tokens must be greater than or equal to 0, got -1.",
        ),
        (
            r#","min_tokens":101"#,
            "min_tokens must be less than or equal to max_tokens=100, got 101.",
        ),
    ] {
        match fe.chat(&body(extra), &RequestMeta::default(), |_| true).await {
            Response::Json { status, body } => {
                assert_eq!(status, 400, "{body}");
                let j: Value = serde_json::from_str(&body).expect("JSON");
                assert_eq!(j["error"]["message"], want, "{j}");
            }
            other => panic!("expected a 400, got {other:?}"),
        }
    }
    assert!(backend.submitted().is_empty());
}
