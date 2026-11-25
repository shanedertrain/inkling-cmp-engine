# ink-server: HTTP server (package F5, gate G-FE4)

`ink-server --model-dir DIR [--host 127.0.0.1] [--port 8407] [--backend mock|engine] ...` serves the OpenAI front end
of `ink-api` over a back end. The transport itself is `ink-api::http` (`router`, `bind_loopback`); this crate is the
process: flags, back end construction, `/version` identity, signal handling.

| Back end | What it is |
|---|---|
| `mock` (default) | scripted reply: a reasoning block and a text block naming the prompt length, 3 tokens per step, `--mock-delay-ms` per step. With `--mock-tool-call NAME=ARGS` (ARGS a JSON object) a prompt without a tool result is answered with a call of that tool instead of the text block: one tool round trip for a client such as `pi` (F6, `bench/raw/2026-11-23-rust-fe-gates/`) |
| `fake` | `ink-sched::EngineBackend::spawn` over `FakeRuntime` (hash tokens on a virtual clock, no model): scheduler, page pool and front end end to end. Flags `--max-running`, `--session-cap`, `--kv-pool-tokens` override the scheduler defaults |
| `engine` | configuration `M3` (package I1): `EngineBackend::spawn` over `ink-runtime::RuntimeAdapter` over `CapiRuntime`, the C++ engine through `ink_rt.h` (`--rt-lib libinkrt_c.so`, `--gemm-algos` pin file, `--slots`, `--kv-pages`; weights from `--model-dir`). Library and engine are created on the scheduler's loop thread (one engine per process, bound to its thread). Chunked prefill, device argmax or host sampling, exact-resume prefix reuse; with `--draft-fold FOLD` (package I2) the runtime's DSpark drafter (int8 head, bf16 K/V, `--draft-ctx` positions per slot, default 65,536): greedy requests without logprobs decode speculatively (8-row verify, greedy accept on the device), every other request one row per step; `--spec-trace FILE` writes one JSON line per round. Without it, plain decode. An engine fault (`ink::fatal()`) ends the process. `--step-log FILE`: one JSON line per step and phase with host timings (`ink_step` events). Records: `bench/raw/2026-11-23-rust-I1-served/`, `bench/raw/2026-11-23-rust-I2-spec-served/` |

Behaviour: loopback only (`localhost` or a loopback IP literal; anything else is refused before the model is loaded
or a socket opened, AGENTS hard rule 5); `--port 0` picks a free port and the log line `listening on http://ADDR`
says which; the first log line is the build line; SIGINT/SIGTERM stop accepting and give open streams 10 s; request
body limit 256 MiB; `/health` is 503 and generation requests are 503 once the back end's loop has died (a panic on
the loop thread is fatal, see `ink-sched` README).

`/version` body: `{"version": <build line>, "backend": ..., "identity": <ink-record identity: engine, version, commit,
dirty, diff, config, lock and vendor hashes, components frontend / scheduler / runtime>}`. `engine`: config `M3`, the
`runtime` entry is the C++ library's identity (`engine=native`, its version and commit, `abi: 1`); `mock` and `fake`:
config `M1`, the `runtime` entry is this build. Prompt token ids of `/v1/completions` are checked against the
vocabulary the runtime reports (`RuntimeCaps::vocab`; fake and engine back ends).

## Gate G-FE4

| Test | Where |
|---|---|
| router on `127.0.0.1:0` with a raw TCP client: routes, SSE bytes equal the in-process frames (HTTP/1.1 chunked and HTTP/1.0), JSON and error bodies, dropped stream and dropped non-streaming request cancel within one back-end step, 8 concurrent streams with `x-session-id`, 64 MiB body, 413 above 256 MiB, loopback refusal, unhealthy back end, loop panic | `crates/ink-api/tests/http.rs` (no model needed: mini tokenizer) |
| the binary as a process: `--version`, non-loopback refusal, mock and fake back ends over the real tokenizer, `/version` identity, the vocabulary check, the engine back end's refusals without its library or pin file, SIGTERM | `crates/ink-server/tests/server.rs` (`INK_MODEL_DIR`) |
| 20 sampled recorded requests per mode (non-streaming, SSE, `prod`) over HTTP equal the direct replay byte for byte | `crates/ink-tools/tests/replay_gate.rs` (`ink-replay --http --sample 20`; `INK_MODEL_DIR`, `INK_REPLAY_DIR`) |
| `curl` against the mock server inside the build container | `bench/raw/2026-11-23-rust-F5-http-gate/curl-session.txt`; framing cross-check (curl's de-chunking = a strict de-chunker on `--raw`) in `bench/raw/2026-11-23-rust-fe-gates/` |
| `pi` 1.6.0 tool round trip against `--backend mock --mock-tool-call` | `bench/raw/2026-11-23-rust-fe-gates/` (F6) |
| a client that stops reading is cut off and cancelled, a fast reader of the same 48 MB stream is not; 103 malformed or hostile bodies give 4xx or a normal answer, never 5xx or a hang | `crates/ink-api/tests/http.rs` (F6) |

PORT-MAP rows (source commit `4171493`, state: ported, gate G-FE4 passed): `src/api/server.{h,cpp}` ->
`ink-api::http`; `tools/ink_server.cpp` -> `ink-server`; `tests/test_api.cpp` `api_http_*` ->
`ink-api/tests/http.rs`.

## Port notes

| C++ | Rust |
|---|---|
| cpp-httplib, a thread pool of `--threads` connections | axum on tokio; `--threads` is gone |
| `RequestMeta.alive` (1 s liveness probe while waiting for a step) | the response body being dropped: hyper drops it when the connection closes, `Sender::closed` wakes the request's task at once (measured 2-3 ms, not up to 1 s) |
| 499 for a vanished non-streaming client | the handler future is dropped by hyper; nothing is sent |
| `is_loopback_host`: exactly `127.0.0.1`, `localhost`, `::1` | `localhost` and any loopback IP literal (`127.0.0.2` too); `::ffff:127.0.0.1` is refused |
| `/health` always 200 | 503 once the back end is unhealthy (S2 review: the back end can die) |
| `/version` fixed string | the build line and the `ink-record` identity |
| write timeout 30 s | a client that stops reading fills the socket buffers and a 1,024-frame queue and is treated as gone (cancel); measured with the mock at full speed: cut off 40 ms after it stops, while a client that reads takes a 48 MB stream whole (F6 test) |
| invalid UTF-8 body | HTTP 400 `JSON decode error` |
