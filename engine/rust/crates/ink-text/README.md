# ink-text (packages F1, F2)

Tokenizer wrapper over `tokenizers` (`Tokenizer`: encode with added tokens recognised or literal, decode, ids) and the
Python-compatible formatting of config B (`pyfmt`, `pyjson`). Ported from `engine/server/src/text/{tokenizer,pyjson,pyfmt}.*`.

| Part | Provided by | Own code |
|---|---|---|
| Byte-level BPE, split regex, added-token matching, decode | `tokenizers` 0.23.2 | wrapper only (two instances: added tokens recognised / literal) |
| `json.dumps`, `_tml_json_dumps`, `repr`, `float()`, `int()`, `round()`, `%g` | none exists | `pyfmt.rs`, `pyjson.rs` (std for parsing, rounding, digit generation) |
| `json.loads` | `serde_json` | not ported |

Package F2 adds the chat renderer, the streaming detokeniser with stop handling, and the `ink-textcli` bin (crate
`ink-tools`: `render MODEL_DIR REQUESTS.json OUT.jsonl`, `encode MODEL_DIR TEXT_FILE`; `-` reads stdin).

| File (C++ in `engine/server/`, deleted at cut-over U2) | Rust |
|---|---|
| `src/text/render.{h,cpp}` (`resolve_chat_params`, `ChatRenderer`, `resolve_effort`, `effort_text`, `RequestError`) | `src/render.rs` |
| `src/text/detok.{h,cpp}` (`DecodeStream`, `OutputState`, `StopParams`) | `src/detok.rs` |
| `tools/ink_textcli.cpp` | `crates/ink-tools/src/bin/ink-textcli.rs` |
| `tests/test_text.cpp` (`text_render_*`, `text_effort_and_floats`, `text_decode_stream_partial_utf8`, `text_output_state_stops`) | `src/render.rs` tests, `tests/render.rs`, `tests/detok.rs`, `src/pyfmt.rs` tests |
| `unicode.{h,cpp}` | not ported: `char`, `str` and `pyfmt::py_strip` |

| Part | Provided by | Own code |
|---|---|---|
| Streaming decode step (partial UTF-8 held back, special tokens skipped) | `tokenizers::DecodeStream` via `Tokenizer::inner()` | wrapper that recreates the stream after `InvalidPrefix` (vLLM does) |
| Chat rendering | none: config B renders with Python code, not the model's Jinja template (no template crate adopted) | `render.rs`, direct port |
| EOS / stop id / `max_tokens` check, stop strings, hold-back, `bisect_left` truncation | none | `detok.rs` |
| `json.loads` of tool-call arguments | `serde_json` | the Python error text is not reproduced |

Parity tests (need the model's tokenizer files and the config B recordings; both stay out of git):

```sh
# once: tokenizer files from the server (no weights) and the recordings
mkdir -p engine/rust/testdata-local/inkling/tiktoken
cp $MODEL_DIR/{tokenizer.json,special_tokens_map.json,tokenizer_config.json} engine/rust/testdata-local/inkling/
cp $MODEL_DIR/tiktoken/tokenizer.model engine/rust/testdata-local/inkling/tiktoken/
python3 -I engine/rust/crates/ink-text/tests/prepare_fixtures.py        # -> engine/rust/testdata-local/parity/
# from engine/rust/
scripts/build.sh env INK_MODEL_DIR=/w/testdata-local/inkling INK_PARITY_DIR=/w/testdata-local/parity \
  cargo test -p ink-text --offline --locked -- --nocapture
```

`prepare_fixtures.py` also writes `render_requests.jsonl` (445 chat requests of `requests.json.zst` with config B's recorded
prompt ids) and `render_s0.jsonl` (29 S0 requests); `tests/render.rs` is gate G-FE1. The record of the CLI run is
`bench/raw/2026-11-23-rust-F2-render-gate/`.
