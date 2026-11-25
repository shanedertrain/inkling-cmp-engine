//! Gate G-FE4, socket level: the router on `127.0.0.1:0` and a raw TCP client. Ported from `api_http_stream_cancel_metrics`
//! and `api_http_concurrent_streams` of `engine/server/tests/test_api.cpp`, plus the checks the plan adds (cancel of
//! a non-streaming request, the body limit, the loopback refusal, a dead back end).

mod common;

use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use common::client::{Req, fetch, read_until_count, send};
use common::{Fixture, fixed_hooks, fixture, mini_tokenizer, reply, special};
use ink_api::http::{BODY_LIMIT_BYTES, BindError, bind_loopback, router};
use ink_api::{Frontend, RequestMeta, Response, ServerConfig};
use ink_backend::{Backend, BackendMetrics, GenRequest, MockBackend, StepFn};
use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

const VERSION: &str = r#"{"version":"test"}"#;

const CHAT: &str = r#"{"model":"inkling-2-mini","messages":[{"role":"user","content":"Hi"}],"max_tokens":100"#;

fn chat_body(extra: &str) -> String {
    format!("{CHAT}{extra}}}")
}

/// Serves `fe` on an ephemeral loopback port.
async fn serve(fe: Frontend) -> SocketAddr {
    let listener = bind_loopback("127.0.0.1", 0).await.expect("bind");
    let addr = listener.local_addr().expect("address");
    tokio::spawn(async move { axum::serve(listener, router(Arc::new(fe), VERSION.to_owned())).await });
    addr
}

/// The text of the `content` deltas of an event stream.
fn content_of(sse: &str) -> String {
    sse.split("\n\n")
        .filter_map(|f| f.strip_prefix("data: "))
        .filter_map(|d| serde_json::from_str::<Value>(d).ok())
        .filter_map(|j| j["choices"][0]["delta"]["content"].as_str().map(str::to_owned))
        .collect()
}

async fn wait_for(what: &str, limit: Duration, mut done: impl FnMut() -> bool) -> Duration {
    let t0 = Instant::now();
    while !done() {
        assert!(t0.elapsed() < limit, "timed out waiting for {what}");
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    t0.elapsed()
}

#[tokio::test]
async fn http_basic_routes() {
    let Fixture { fe, .. } = fixture();
    let addr = serve(fe).await;
    let health = fetch(addr, Req::get("/health")).await;
    assert_eq!((health.status, health.body.len()), (200, 0));
    let models = fetch(addr, Req::get("/v1/models")).await;
    assert_eq!(models.status, 200);
    assert_eq!(models.header("content-type"), Some("application/json"));
    assert!(models.text().contains("inkling-2-mini"));
    let version = fetch(addr, Req::get("/version")).await;
    assert_eq!((version.status, version.text()), (200, VERSION.to_owned()));
    let metrics = fetch(addr, Req::get("/metrics")).await;
    assert_eq!(metrics.status, 200);
    assert_eq!(
        metrics.header("content-type"),
        Some("text/plain; version=0.0.4; charset=utf-8")
    );
    assert!(metrics.text().contains("vllm:num_requests_running"));
    assert_eq!(fetch(addr, Req::get("/nope")).await.status, 404);
    assert_eq!(fetch(addr, Req::get("/v1/chat/completions")).await.status, 405);
}

/// The frames over HTTP are the frames of the in-process call, byte for byte, with HTTP/1.1 chunking and with the
/// close-delimited HTTP/1.0 body.
#[tokio::test]
async fn http_sse_equals_the_in_process_frames() {
    let Fixture { tok, backend, fe } = fixture();
    let script = reply(&tok, "Think over HTTP.", "Over HTTP.", 2);
    backend.push(script.clone());
    let mut want = String::new();
    let r = fe
        .chat(&chat_body(r#","stream":true"#), &RequestMeta::default(), |f| {
            want.push_str(f);
            true
        })
        .await;
    assert_eq!(r, Response::Streamed);
    assert_eq!(content_of(&want), "Over HTTP.");
    assert!(want.ends_with("data: [DONE]\n\n"));

    let Fixture { backend, fe, .. } = fixture();
    let addr = serve(fe).await;
    for minor10 in [false, true] {
        backend.push(script.clone());
        let body = chat_body(r#","stream":true"#);
        let mut req = Req::post("/v1/chat/completions", body.as_bytes());
        if minor10 {
            req = req.http10();
        }
        let got = fetch(addr, req).await;
        assert_eq!(got.status, 200);
        assert_eq!(got.header("content-type"), Some("text/event-stream"));
        assert_eq!(got.header("cache-control"), Some("no-cache"));
        assert_eq!(
            got.header("transfer-encoding").is_some(),
            !minor10,
            "HTTP/1.1 streams chunked, HTTP/1.0 by close"
        );
        assert_eq!(got.text(), want, "HTTP/1.{}", u8::from(!minor10));
    }
}

#[tokio::test]
async fn http_non_streaming_and_errors() {
    let Fixture { tok, backend, fe } = fixture();
    let addr = serve(fe).await;
    backend.push(reply(&tok, "R.", "Full over HTTP.", 3));
    let body = chat_body("");
    let ok = fetch(addr, Req::post("/v1/chat/completions", body.as_bytes())).await;
    assert_eq!(ok.status, 200);
    let j: Value = serde_json::from_str(&ok.text()).expect("JSON");
    assert_eq!(j["choices"][0]["message"]["content"], "Full over HTTP.");
    // a rejected streaming request is a JSON error, not an event stream
    let bad = fetch(
        addr,
        Req::post("/v1/chat/completions", br#"{"messages":[],"stream":true}"#),
    )
    .await;
    assert_eq!(bad.status, 400);
    assert_eq!(bad.header("content-type"), Some("application/json"));
    let unknown = chat_body("").replace("inkling-2-mini", "other");
    assert_eq!(
        fetch(addr, Req::post("/v1/chat/completions", unknown.as_bytes()))
            .await
            .status,
        404
    );
    // not JSON, not UTF-8
    assert_eq!(fetch(addr, Req::post("/v1/completions", b"{")).await.status, 400);
    let r = fetch(addr, Req::post("/v1/completions", &[0xff, 0xfe])).await;
    assert_eq!(r.status, 400);
    assert!(r.text().contains("not UTF-8"));
}

/// Port of `api_http_stream_cancel_metrics`: a client that drops the stream cancels the request, within one step
/// even while the back end is silent (the step here takes 300 ms).
#[tokio::test]
async fn http_dropped_stream_cancels_within_one_step() {
    const STEP: Duration = Duration::from_millis(300);
    let Fixture { tok, backend, fe } = fixture();
    let addr = serve(fe).await;
    let mut long = reply(&tok, "Long thought that keeps going and going.", &"z".repeat(400), 1);
    long.step_delay = STEP;
    backend.push(long);
    let body = chat_body(r#","stream":true,"max_tokens":2000"#);
    let mut s = send(addr, &Req::post("/v1/chat/completions", body.as_bytes())).await;
    // headers and two data frames, then the client goes away while the back end sleeps
    let got = read_until_count(&mut s, "data: ", 2, Duration::from_secs(5)).await;
    assert!(String::from_utf8_lossy(&got).contains("200 OK"));
    let t_drop = Instant::now();
    drop(s);
    let latency = wait_for("the cancel", Duration::from_secs(5), || backend.cancels() == 1).await;
    eprintln!("cancel {latency:?} after the client dropped (back end step {STEP:?})");
    assert!(
        t_drop.elapsed() < STEP,
        "cancelled within one step, took {:?}",
        t_drop.elapsed()
    );
    assert_eq!(backend.cancels(), 1);
    let metrics = wait_metric(addr, "ink:requests_cancelled_total{model_name=\"inkling-2-mini\"} 1").await;
    assert!(metrics.contains("vllm:num_requests_running"));
}

async fn wait_metric(addr: SocketAddr, line: &str) -> String {
    let t0 = Instant::now();
    loop {
        let text = fetch(addr, Req::get("/metrics")).await.text();
        if text.contains(line) {
            return text;
        }
        assert!(
            t0.elapsed() < Duration::from_secs(5),
            "metric line {line:?} never appeared:\n{text}"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// A non-streaming request is cancelled when its connection closes before the answer.
#[tokio::test]
async fn http_dropped_non_streaming_request_cancels() {
    const STEP: Duration = Duration::from_millis(300);
    let Fixture { tok, backend, fe } = fixture();
    let addr = serve(fe).await;
    let mut long = reply(&tok, "Long.", &"z".repeat(400), 1);
    long.step_delay = STEP;
    backend.push(long);
    let body = chat_body(r#","max_tokens":2000"#);
    let s = send(addr, &Req::post("/v1/chat/completions", body.as_bytes())).await;
    wait_for("the request reaching the back end", Duration::from_secs(5), || {
        !backend.submitted().is_empty()
    })
    .await;
    let t_drop = Instant::now();
    drop(s);
    wait_for("the cancel", Duration::from_secs(5), || backend.cancels() == 1).await;
    assert!(t_drop.elapsed() < STEP, "took {:?}", t_drop.elapsed());
    wait_metric(addr, "ink:requests_cancelled_total{model_name=\"inkling-2-mini\"} 1").await;
}

/// Port of `api_http_concurrent_streams`: eight streams at once, each with its `x-session-id` header.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn http_concurrent_streams_carry_the_session_header() {
    let Fixture { tok, backend, fe } = fixture();
    backend.set_generator({
        let tok = Arc::clone(&tok);
        move |_| {
            let mut s = reply(&tok, "Thinking in parallel.", "Answer for this session.", 1);
            s.step_delay = Duration::from_millis(2);
            s
        }
    });
    let addr = serve(fe).await;
    let body = chat_body(r#","stream":true"#);
    let mut tasks = Vec::new();
    for i in 0..8 {
        let body = body.clone();
        tasks.push(tokio::spawn(async move {
            let req = Req::post("/v1/chat/completions", body.as_bytes()).header("x-session-id", format!("s{i}"));
            fetch(addr, req).await
        }));
    }
    for t in tasks {
        let r = t.await.expect("task");
        assert_eq!(r.status, 200);
        let text = r.text();
        assert!(text.ends_with("data: [DONE]\n\n"), "the stream closed properly");
        assert_eq!(content_of(&text), "Answer for this session.");
    }
    let subs = backend.submitted();
    assert_eq!(subs.len(), 8);
    let mut sessions: Vec<&str> = subs.iter().map(|r| r.session.as_str()).collect();
    sessions.sort_unstable();
    assert_eq!(sessions, ["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7"]);
}

/// 64 MiB of JSON goes through; a declared length above the limit is refused with 413 before the body is read.
#[tokio::test]
async fn http_body_limit() {
    let Fixture { tok, backend, fe } = fixture();
    let addr = serve(fe).await;
    backend.push(reply(&tok, "R.", "Big body.", 3));
    let mut body = br#"{"prompt":"Hi","max_tokens":4,"pad":""#.to_vec();
    body.resize(64 << 20, b'x');
    body.extend_from_slice(br#""}"#);
    assert!(body.len() > 64 << 20);
    let r = fetch(addr, Req::post("/v1/completions", &body)).await;
    assert_eq!(r.status, 200, "{}", r.text().chars().take(300).collect::<String>());
    let j: Value = serde_json::from_str(&r.text()).expect("JSON");
    assert!(j["choices"][0]["text"].as_str().is_some_and(|t| !t.is_empty()), "{j}");

    // one byte over the limit, sent in 1 MiB pieces: the answer is 413 and the server stops reading
    let mut req = Req::post("/v1/completions", b"");
    req.declared_length = Some(BODY_LIMIT_BYTES + 1);
    let (mut rd, mut wr) = send(addr, &req).await.into_split();
    let writer = tokio::spawn(async move {
        let piece = vec![b' '; 1 << 20];
        let mut left = BODY_LIMIT_BYTES + 1;
        while left > 0 {
            let n = left.min(piece.len());
            if wr.write_all(&piece[..n]).await.is_err() {
                break; // the server answered and closed
            }
            left -= n;
        }
        wr
    });
    let mut buf = Vec::new();
    tokio::time::timeout(Duration::from_secs(60), async {
        let mut chunk = [0u8; 1024];
        while !buf.windows(4).any(|w| w == b"\r\n\r\n") {
            let k = rd.read(&mut chunk).await.expect("read");
            assert!(k > 0, "the server closed without answering");
            buf.extend_from_slice(&chunk[..k]);
        }
    })
    .await
    .expect("an answer");
    assert!(String::from_utf8_lossy(&buf).starts_with("HTTP/1.1 413"), "{buf:?}");
    drop(writer);
}

#[tokio::test]
async fn bind_refuses_non_loopback_addresses() {
    for host in ["0.0.0.0", "::", "192.168.1.5", "example.com", ""] {
        let e = bind_loopback(host, 0).await.expect_err(host);
        assert!(matches!(&e, BindError::NotLoopback(h) if h == host), "{host:?}: {e}");
    }
    for host in ["127.0.0.1", "localhost"] {
        let l = bind_loopback(host, 0).await.expect(host);
        assert!(l.local_addr().expect("address").ip().is_loopback());
    }
}

/// A back end wrapper whose health can be switched off.
struct Switch {
    inner: MockBackend,
    up: AtomicBool,
}

impl Backend for Switch {
    fn submit(&self, req: GenRequest, on_step: StepFn) {
        self.inner.submit(req, on_step);
    }
    fn finish(&self, id: u64) {
        self.inner.finish(id);
    }
    fn cancel(&self, id: u64) {
        self.inner.cancel(id);
    }
    fn metrics(&self) -> BackendMetrics {
        self.inner.metrics()
    }
    fn healthy(&self) -> bool {
        self.up.load(Ordering::SeqCst)
    }
}

#[tokio::test]
async fn http_unhealthy_back_end_refuses_requests() {
    let tok = mini_tokenizer();
    let backend = Arc::new(Switch {
        inner: MockBackend::new(),
        up: AtomicBool::new(true),
    });
    let cfg = ServerConfig {
        eos_token_id: u32::try_from(special(&tok, "<|content_model_end_sampling|>")).expect("small id"),
        ..ServerConfig::default()
    };
    let fe = Frontend::new(Arc::clone(&tok), cfg, backend.clone(), fixed_hooks()).expect("front end");
    let addr = serve(fe).await;
    assert_eq!(fetch(addr, Req::get("/health")).await.status, 200);
    backend.up.store(false, Ordering::SeqCst);
    assert_eq!(fetch(addr, Req::get("/health")).await.status, 503);
    let body = chat_body("");
    let r = fetch(addr, Req::post("/v1/chat/completions", body.as_bytes())).await;
    assert_eq!(r.status, 503);
    let j: Value = serde_json::from_str(&r.text()).expect("JSON");
    assert_eq!(j["error"]["code"], 503);
    assert!(backend.inner.submitted().is_empty(), "nothing reached the back end");
    // the read-only routes stay up
    assert_eq!(fetch(addr, Req::get("/metrics")).await.status, 200);
}

/// The scheduler's panic policy seen from HTTP: a panic on the loop thread ends the request in flight with an error
/// frame, turns `/health` to 503 and refuses new requests.
#[tokio::test]
async fn http_loop_panic_is_fatal() {
    use ink_sched::{Config, EngineBackend, ModelRuntime, RuntimeCaps, SegResult, StepError, StepPlan};
    use std::convert::Infallible;

    struct Panics;
    impl ModelRuntime for Panics {
        fn caps(&self) -> RuntimeCaps {
            RuntimeCaps::default()
        }
        fn step(&mut self, _: &StepPlan) -> Result<Vec<SegResult>, StepError> {
            panic!("runtime bug (expected by the test)");
        }
        fn now_ms(&self) -> f64 {
            0.0
        }
    }
    let tok = mini_tokenizer();
    let backend = Arc::new(EngineBackend::spawn(Config::default(), || Ok::<_, Infallible>(Panics)).expect("spawns"));
    let cfg = ServerConfig {
        eos_token_id: u32::try_from(special(&tok, "<|content_model_end_sampling|>")).expect("small id"),
        ..ServerConfig::default()
    };
    let fe = Frontend::new(tok, cfg, backend, fixed_hooks()).expect("front end");
    let addr = serve(fe).await;
    assert_eq!(fetch(addr, Req::get("/health")).await.status, 200);
    let body = chat_body(r#","stream":true"#);
    let r = fetch(addr, Req::post("/v1/chat/completions", body.as_bytes())).await;
    assert_eq!(r.status, 200);
    let text = r.text();
    assert!(text.contains(r#""type":"InternalServerError""#), "{text}");
    assert!(text.ends_with("data: [DONE]\n\n"));
    let t0 = Instant::now();
    while fetch(addr, Req::get("/health")).await.status == 200 {
        assert!(t0.elapsed() < Duration::from_secs(5), "still healthy");
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    assert_eq!(fetch(addr, Req::get("/health")).await.status, 503);
    let again = fetch(addr, Req::post("/v1/chat/completions", chat_body("").as_bytes())).await;
    assert_eq!(again.status, 503);
}

/// A device failure seen from HTTP (review V1): the runtime's `StepError::Device` fails the request in flight with an
/// error frame, turns `/health` to 503 and refuses new requests (`ink-server` then exits).
#[tokio::test]
async fn http_device_failure_is_fatal() {
    use ink_sched::{Config, EngineBackend, ModelRuntime, RuntimeCaps, SegResult, StepError, StepPlan};
    use std::convert::Infallible;

    struct DeadDevice;
    impl ModelRuntime for DeadDevice {
        fn caps(&self) -> RuntimeCaps {
            RuntimeCaps::default()
        }
        fn step(&mut self, _: &StepPlan) -> Result<Vec<SegResult>, StepError> {
            Err(StepError::Device(
                "runtime device error: cuStreamSynchronize: CUDA_ERROR_ILLEGAL_ADDRESS".to_owned(),
            ))
        }
        fn now_ms(&self) -> f64 {
            0.0
        }
    }
    let tok = mini_tokenizer();
    let backend =
        Arc::new(EngineBackend::spawn(Config::default(), || Ok::<_, Infallible>(DeadDevice)).expect("spawns"));
    let cfg = ServerConfig {
        eos_token_id: u32::try_from(special(&tok, "<|content_model_end_sampling|>")).expect("small id"),
        ..ServerConfig::default()
    };
    let fe = Frontend::new(tok, cfg, backend, fixed_hooks()).expect("front end");
    let addr = serve(fe).await;
    assert_eq!(fetch(addr, Req::get("/health")).await.status, 200);
    let r = fetch(addr, Req::post("/v1/chat/completions", chat_body("").as_bytes())).await;
    assert_eq!(r.status, 500);
    let text = r.text();
    assert!(
        text.contains("device failure") && text.contains("ILLEGAL_ADDRESS"),
        "{text}"
    );
    let t0 = Instant::now();
    while fetch(addr, Req::get("/health")).await.status == 200 {
        assert!(t0.elapsed() < Duration::from_secs(5), "still healthy");
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    let again = fetch(addr, Req::post("/v1/chat/completions", chat_body("").as_bytes())).await;
    assert_eq!(again.status, 503);
}

/// A prompt over the engine's per-session cap is a client error (review V13): HTTP 400 `BadRequestError` before any
/// stream starts, as config B answers a prompt over its limits, streaming or not; the back end keeps serving.
#[tokio::test]
async fn http_prompt_over_the_session_cap_is_a_bad_request() {
    use ink_sched::{Config, EngineBackend, FakeRuntime};
    use std::convert::Infallible;

    let tok = mini_tokenizer();
    let sched = Config {
        session_cap_tokens: 4,
        ..Config::default()
    };
    let backend =
        Arc::new(EngineBackend::spawn(sched, || Ok::<_, Infallible>(FakeRuntime::default())).expect("spawns"));
    let cfg = ServerConfig {
        eos_token_id: u32::try_from(special(&tok, "<|content_model_end_sampling|>")).expect("small id"),
        ..ServerConfig::default()
    };
    let fe = Frontend::new(tok, cfg, backend, fixed_hooks()).expect("front end");
    let addr = serve(fe).await;
    for extra in ["", r#","stream":true"#] {
        let r = fetch(addr, Req::post("/v1/chat/completions", chat_body(extra).as_bytes())).await;
        assert_eq!(r.status, 400, "stream {extra:?}: {}", r.text());
        let j: Value = serde_json::from_str(&r.text()).expect("a JSON error body");
        assert_eq!(j["error"]["type"], "BadRequestError");
        assert_eq!(j["error"]["code"], 400);
        assert!(
            j["error"]["message"]
                .as_str()
                .is_some_and(|m| m.contains("exceeds the per-session cap of 4 tokens")),
            "{j}"
        );
    }
    let completion = r#"{"model":"inkling-2-mini","prompt":[1,2,3,4,5],"max_tokens":1}"#;
    let r = fetch(addr, Req::post("/v1/completions", completion.as_bytes())).await;
    assert_eq!(r.status, 400, "{}", r.text());
    assert_eq!(fetch(addr, Req::get("/health")).await.status, 200);
}

/// A client that stops reading (but keeps the connection open) is cut off and its request cancelled once the socket
/// buffers and the frame queue are full; a client that reads as fast as it can gets the whole stream of the same
/// size. The back end emits one token per step with no delay, the fastest a back end can produce frames.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn http_stalled_reader_is_cut_off_and_a_fast_reader_is_not() {
    const TOKENS: usize = 200_000;
    let body = chat_body(r#","stream":true,"max_tokens":250000"#);

    let Fixture { tok, backend, fe } = fixture();
    let addr = serve(fe).await;
    backend.push(reply(&tok, "R.", &"z".repeat(TOKENS), 1));
    let mut s = send(addr, &Req::post("/v1/chat/completions", body.as_bytes())).await;
    let got = read_until_count(&mut s, "data: ", 1, Duration::from_secs(5)).await;
    assert!(String::from_utf8_lossy(&got).contains("200 OK"));
    // the client stalls: it holds the socket and reads nothing more
    let stalled = wait_for("the stalled client to be cut off", Duration::from_secs(60), || {
        backend.cancels() == 1
    })
    .await;
    eprintln!("stalled reader cut off {stalled:?} after its first frame");
    wait_metric(addr, "ink:requests_cancelled_total{model_name=\"inkling-2-mini\"} 1").await;
    // the server still serves others while the stalled socket is open
    assert_eq!(fetch(addr, Req::get("/health")).await.status, 200);
    drop(s);

    let Fixture { tok, backend, fe } = fixture();
    let addr = serve(fe).await;
    backend.push(reply(&tok, "R.", &"z".repeat(TOKENS), 1));
    let t0 = Instant::now();
    let r = fetch(addr, Req::post("/v1/chat/completions", body.as_bytes())).await;
    eprintln!("fast reader: {} bytes in {:?}", r.body.len(), t0.elapsed());
    assert_eq!(r.status, 200);
    let text = r.text();
    assert!(text.ends_with("data: [DONE]\n\n"), "the stream is complete");
    assert_eq!(content_of(&text).len(), TOKENS);
    assert_eq!(backend.cancels(), 0, "a reading client is never cut off");
}

/// Malformed and hostile bodies on both generation endpoints: each gets a 4xx JSON error (or a normal answer), never
/// a 5xx, a hang or a dropped connection, and the server stays healthy.
#[tokio::test]
async fn http_malformed_bodies_are_client_errors() {
    let Fixture { tok, backend, fe } = fixture();
    backend.set_generator({
        let tok = Arc::clone(&tok);
        move |_| reply(&tok, "R.", "ok", 2)
    });
    let addr = serve(fe).await;
    let deep = format!("{}{}", "[".repeat(100_000), "]".repeat(100_000));
    // nesting inside a tool schema, which the renderer walks recursively: just under serde_json's depth limit of
    // 128 it is rendered, above it the body is refused
    let deep_tool = |depth: usize| {
        format!(
            r#"{{"messages":[{{"role":"user","content":"Hi"}}],"tools":[{{"type":"function","function":{{"name":"f","parameters":{{"type":"object","x":{}1{}}}}}}}]}}"#,
            "[".repeat(depth),
            "]".repeat(depth)
        )
    };
    let chat = "/v1/chat/completions";
    let cmpl = "/v1/completions";
    let msg = |extra: &str| format!(r#"{{"messages":[{{"role":"user","content":"Hi"}}]{extra}}}"#);
    let prompt = |p: &str, extra: &str| format!(r#"{{"prompt":{p}{extra}}}"#);
    let mut cases: Vec<(&str, Vec<u8>, u16)> = vec![
        (chat, b"".to_vec(), 400),
        (chat, b"{".to_vec(), 400),
        (chat, br#"{"messages":[{"role":"user","content":"Hi"}]"#.to_vec(), 400),
        (chat, b"[]".to_vec(), 400),
        (chat, b"null".to_vec(), 400),
        (chat, b"123".to_vec(), 400),
        (chat, b"\"text\"".to_vec(), 400),
        (chat, deep.clone().into_bytes(), 400),
        (cmpl, deep.into_bytes(), 400),
        (chat, deep_tool(118).into_bytes(), 200),
        (chat, deep_tool(200).into_bytes(), 400),
        (
            chat,
            b"{\"messages\":[{\"role\":\"user\",\"content\":\"\xff\xfe\"}]}".to_vec(),
            400,
        ),
        (
            chat,
            b"{\"messages\":[{\"role\":\"user\",\"content\":\"a\x00b\"}]}".to_vec(),
            400,
        ),
        (chat, msg("").replace("Hi", r"a\u0000b").into_bytes(), 200),
        (chat, msg("").replace("Hi", r"\ud800").into_bytes(), 400),
        (chat, msg("").replace("Hi", r"\udc00\ud800").into_bytes(), 400),
        (chat, msg("").replace("Hi", r"😀").into_bytes(), 200),
        (chat, msg(r#","max_tokens":1e400"#).into_bytes(), 400),
        (chat, msg(r#","max_tokens":18446744073709551616"#).into_bytes(), 400),
        (
            chat,
            msg(r#","max_tokens":99999999999999999999999999999999"#).into_bytes(),
            400,
        ),
        (chat, msg(r#","max_tokens":-1"#).into_bytes(), 400),
        (chat, msg(r#","max_tokens":0"#).into_bytes(), 400),
        (chat, msg(r#","max_tokens":"10""#).into_bytes(), 400),
        (chat, msg(r#","max_tokens":true"#).into_bytes(), 400),
        (
            chat,
            msg(r#","max_completion_tokens":-9223372036854775808"#).into_bytes(),
            400,
        ),
        (chat, msg(r#","temperature":"hot""#).into_bytes(), 400),
        (chat, msg(r#","top_k":-1,"top_p":1e308,"seed":-1"#).into_bytes(), 200),
        (chat, msg(r#","seed":18446744073709551615"#).into_bytes(), 200),
        (chat, msg(r#","logprobs":true,"top_logprobs":1e30"#).into_bytes(), 400),
        (chat, msg(r#","logprobs":true,"top_logprobs":"5""#).into_bytes(), 400),
        (chat, msg(r#","top_logprobs":5"#).into_bytes(), 400),
        (chat, msg(r#","n":2"#).into_bytes(), 400),
        (chat, msg(r#","stop":5"#).into_bytes(), 400),
        (chat, msg(r#","stop":[1]"#).into_bytes(), 400),
        (chat, msg(r#","stop_token_ids":"x""#).into_bytes(), 400),
        (chat, msg(r#","stop_token_ids":[-1,1e20,4294967296]"#).into_bytes(), 200),
        (
            chat,
            msg(r#","stream_options":{"include_usage":true}"#).into_bytes(),
            400,
        ),
        (chat, msg(r#","model":5"#).into_bytes(), 400),
        (chat, msg(r#","model":"other""#).into_bytes(), 404),
        (chat, msg(r#","user":5"#).into_bytes(), 200),
        (chat, msg(r#","cache_salt":5"#).into_bytes(), 400),
        // lenient where vLLM's pydantic models answer 400 (documented deviations, no recorded request has them)
        (chat, msg(r#","tools":"x""#).into_bytes(), 200),
        (chat, msg(r#","tools":[1]"#).into_bytes(), 400),
        (chat, msg(r#","tools":[{"type":"function"}]"#).into_bytes(), 400),
        (
            chat,
            msg(r#","tools":[{"type":"function","function":{"name":"f"}}],"tool_choice":5"#).into_bytes(),
            400,
        ),
        (
            chat,
            msg(r#","tool_choice":{"type":"function","function":{}}"#).into_bytes(),
            400,
        ),
        (chat, msg(r#","reasoning_effort":5"#).into_bytes(), 400),
        (
            chat,
            msg(r#","chat_template_kwargs":{"reasoning_effort":"nan"}"#).into_bytes(),
            400,
        ),
        (
            chat,
            msg(r#","chat_template_kwargs":{"reasoning_effort":1e308}"#).into_bytes(),
            400,
        ),
        (chat, msg(r#","chat_template_kwargs":5"#).into_bytes(), 200),
        (chat, br#"{"messages":"x"}"#.to_vec(), 400),
        (chat, br#"{"messages":[1]}"#.to_vec(), 400),
        (chat, br#"{"messages":[{"role":5,"content":"x"}]}"#.to_vec(), 400),
        (
            chat,
            br#"{"messages":[{"role":"user","content":{"a":1}}]}"#.to_vec(),
            400,
        ),
        (
            chat,
            br#"{"messages":[{"role":"user","content":[{"type":"image_url"}]}]}"#.to_vec(),
            200,
        ),
        (chat, br#"{"messages":[{"role":"user","content":[5]}]}"#.to_vec(), 400),
        (
            chat,
            br#"{"messages":[{"role":"tool","content":[{"type":"image"}]}]}"#.to_vec(),
            400,
        ),
        (
            chat,
            br#"{"messages":[{"role":"assistant","tool_calls":[{"function":5}]}]}"#.to_vec(),
            400,
        ),
        (cmpl, b"".to_vec(), 400),
        (cmpl, prompt("[]", "").into_bytes(), 400),
        (cmpl, prompt(r#""""#, "").into_bytes(), 400),
        (cmpl, prompt("[-1]", "").into_bytes(), 400),
        (cmpl, prompt("[4294967296]", "").into_bytes(), 400),
        (cmpl, prompt("[1.5]", "").into_bytes(), 400),
        (cmpl, prompt("[1,\"a\"]", "").into_bytes(), 400),
        (cmpl, prompt("[[1,2]]", "").into_bytes(), 400),
        (cmpl, prompt(r#"{"a":1}"#, "").into_bytes(), 400),
        (cmpl, prompt(r#"["a","b"]"#, "").into_bytes(), 400),
        (cmpl, prompt(r#""Hi""#, r#","max_tokens":0"#).into_bytes(), 400),
        (cmpl, prompt(r#""Hi""#, r#","max_tokens":-5"#).into_bytes(), 400),
        (cmpl, prompt(r#""Hi""#, r#","max_tokens":1e400"#).into_bytes(), 400),
        (cmpl, prompt(r#""Hi""#, r#","logprobs":-3"#).into_bytes(), 200),
        (cmpl, prompt(r#""Hi""#, r#","logprobs":"x""#).into_bytes(), 400),
        (cmpl, prompt(r#""Hi""#, r#","echo":true"#).into_bytes(), 400),
        (cmpl, prompt(r#""Hi\u0000""#, "").into_bytes(), 200),
        (cmpl, prompt(r#""<|end_message|>\ud800""#, "").into_bytes(), 400),
    ];
    // the same rejections must not depend on the stream flag
    let streamed: Vec<(&str, Vec<u8>, u16)> = cases
        .iter()
        .filter(|(_, b, status)| {
            *status == 400
                && b.starts_with(b"{\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],")
                && !b.windows(14).any(|w| w == b"stream_options")
        })
        .map(|(p, b, s)| {
            let mut b = b.clone();
            b.truncate(b.len() - 1);
            b.extend_from_slice(br#","stream":true}"#);
            (*p, b, *s)
        })
        .collect();
    cases.extend(streamed);
    let mut table = Vec::new();
    for (path, body, want) in &cases {
        let r = fetch(addr, Req::post(path, body)).await;
        let shown: String = String::from_utf8_lossy(body).chars().take(70).collect();
        table.push(format!("{} {path} {shown}", r.status));
        assert_eq!(
            r.status,
            *want,
            "{path} {shown:?}: {}",
            r.text().chars().take(300).collect::<String>()
        );
        if r.status == 200 {
            assert!(!r.body.is_empty());
        } else {
            assert_eq!(r.header("content-type"), Some("application/json"), "{shown:?}");
            let j: Value = serde_json::from_slice(&r.body).expect("a JSON error body");
            assert_eq!(j["error"]["code"], r.status, "{shown:?}");
        }
    }
    // a header that is not visible ASCII is ignored, not an error
    let r = fetch(
        addr,
        Req::post(chat, msg("").as_bytes()).header("x-session-id", "s\u{e9}x"),
    )
    .await;
    table.push(format!("{} {chat} x-session-id with non-ASCII bytes", r.status));
    assert_eq!(r.status, 200);
    eprintln!("malformed-input table ({} cases):\n{}", table.len(), table.join("\n"));
    assert_eq!(fetch(addr, Req::get("/health")).await.status, 200);
}
