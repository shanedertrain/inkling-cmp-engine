# ink-api (packages F4, F5)

OpenAI request handling with config B's wire behaviour (`/v1/chat/completions`, `/v1/completions`, `/v1/models`,
`/metrics`) over a `Backend`, and the HTTP transport. Ported from `engine/server/src/api/{openai,server}.*` @ `4171493`
(the C++ was deleted at cut-over U2), itself a port of vLLM's OpenAI serving layer (Apache-2.0).

| Module | Content |
|---|---|
| `frontend` | `Frontend`: `prepare_chat` / `prepare_completions` (validate, render, 4xx before anything streams), `execute` (back-end steps to frames or a body) |
| `gen_params` | sampling, stop and stream options of both endpoints (`parse_gen`) |
| `chat`, `completions` | chunk and response assembly, usage, finish reasons |
| `logprobs` | logprob rows, partial-UTF-8 repair, the two wire forms |
| `run` | one submitted request; dropping it cancels the request in the back end |
| `stats` | counters and the Prometheus text under vLLM's metric names |
| `http` | axum router, SSE body from a bounded channel, cancel on a dropped stream, `bind_loopback` (loopback only) |

Gates: G-FE3 (`ink-tools` `replay_gate`), G-FE4 (`tests/http.rs`, `ink-server`), records
`bench/raw/2026-11-23-rust-F4-api-gate/`, `-F5-http-gate/`, `-fe-gates/` (F6 re-run).

## Limits and deviations from config B

None of these can change a gate: every recorded request is valid, and the gates re-run after each change.

| Input | Here | vLLM (config B) |
|---|---|---|
| Malformed JSON, invalid UTF-8, lone surrogate escapes, nesting deeper than 128, numbers beyond f64 | 400 `JSON decode error: ...` (serde_json's text) | 400 with Python's text |
| Integer fields given as floats (`max_tokens: 1.5`) | truncated (`get<int>` of the C++ port) | 400 (pydantic) |
| Fields of the wrong type that the renderer does not read (`tools: "x"`, `user: 5`, an `image_url` part without a URL) | ignored, 200 | 400 (pydantic) |
| Out-of-range sampling values (negative temperature, `top_p` > 1, `min_tokens` > `max_tokens`, negative completions `logprobs`) | passed on (negative `logprobs`: off) | 400 (`SamplingParams._verify_args`) |
| Prompt token ids of `/v1/completions` | 400 `Token id N is out of vocabulary` (N the largest id) at or above `ServerConfig::vocab_size`, the vocabulary the runtime reports (`RuntimeCaps::vocab`; I1). The mock back end has no model and sets none: any `u32` id passes | 400 with the same text above the tokenizer's largest id (`validate_token_ids`) |
| Request body | 256 MiB limit (413 above); the parsed `serde_json::Value` can take several times the body in memory | no limit |
| A client that stops reading | cut off once the socket buffers and a 1,024-frame queue are full (measured 40 ms at the mock's full speed), then cancelled | 30 s write timeout in the C++ port |
| `max_tokens` < 1 on `/v1/completions` | 400 (F6; the C++ port let it through to the back end) | 400 |
