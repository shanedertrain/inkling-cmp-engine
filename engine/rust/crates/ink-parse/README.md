# ink-parse (package F3)

Config B's `--reasoning-parser inkling --tool-call-parser inkling`: reasoning, content and tool calls out of the text and
token ids of Inkling-2-Mini's output, streaming ([`InklingParser::parse_delta`]) and non-streaming (`parse`), with the
tool schema's type coercion of the arguments. Ported from `engine/server/src/parse/inkling_parser.*` @ `6100ab8`, itself
a port of vLLM's Inkling parser (Apache-2.0, Copyright contributors to the vLLM project).

| C++ (`engine/server/src/parse/`, deleted at cut-over U2) | Rust (`src/`) |
|---|---|
| `Lexer`, terminals | `lexer.rs` |
| `Scanner` (token-id scanner) | `scanner.rs` |
| grammar, `StreamingParserEngine`, `on_terminal`, argument streaming | `machine.rs` |
| `ParserEngine` events to deltas, tool slots, extraction | `assemble.rs` |
| `ParserEngine` entry points | `engine.rs` |
| `InklingParser` (`parse_delta`, `parse`, helpers) | `parser.rs` |
| `ParserTables`, `load_all_special_tokens` | `tables.rs` (`special_tokens_from_dir`) |
| schema coercion, `fix_arg_types`, `find_tool_properties` | `coerce.rs` |
| `safe_arg_prefix`, `extract_json_member_object`, `search_name` | `args.rs` |
| `DeltaMessage`, `FunctionCall`, `Parsed` | `delta.rs` |
| `ParseRequest` | `request.rs` |
| the C++ JSON helpers (`json_loads`, `dumps`) | not ported: `serde_json` and `ink-text`'s `pyfmt` |
| `ink_parsecli` | not a binary: the differential test runs the same loop in process (the `ink-parsecli` stub was deleted at cut-over U2) |

Behaviours kept from the C++ port (`bench/raw/2026-11-23-S5-server-parity/README.md`) that belong to the parser: the
last tool call's `arguments` become `""` at stream end when unset; an empty `tool_calls` list is never serialised
(`DeltaMessage::to_value`); a field assigned `None` is written as `null`, an unset one is omitted. EOS before the length
cap, first-chunk key order and the 400 for non-text tool/system parts belong to detokenisation and the API layer.

## Tests

| Test | Needs |
|---|---|
| unit tests in `src/` (lexer, argument scanning, coercion) and `tests/behaviour.rs` (the five tests of `test_parser.cpp` and the pinned behaviours, on a small byte-level tokenizer built in the test) | nothing |
| `tests/differential.rs`: gate G-FE2, the recorded token streams re-chunked four ways through the parser against config B's own parser output | the model's tokenizer files and the recordings, both out of git; skips with a message when `INK_MODEL_DIR` or `INK_PARSER_DIR` is unset |

```sh
python3 -I engine/rust/crates/ink-parse/tests/prepare_fixtures.py          # -> engine/rust/testdata-local/parser/
# tokenizer files as in crates/ink-text/README.md (engine/rust/testdata-local/inkling/)
# from engine/rust/
scripts/build.sh env INK_MODEL_DIR=/w/testdata-local/inkling INK_PARSER_DIR=/w/testdata-local/parser \
  cargo test -p ink-parse --offline --locked --test differential -- --nocapture
```

The step text of each recorded stream (EOS text excluded, partial UTF-8 held back) is made in the test by
`tests/common/mod.rs::steps_of` over `tokenizers`' `DecodeStream`; the front end's `OutputState` replaces it when it
lands in `ink-text`. A reference id that was captured more than once (8 ids, 3 captures) is compared row by row, in order.

## Limits (shared with the C++ port)

| Limit | Effect |
|---|---|
| `serde_json` without `arbitrary_precision` | an integer beyond `[-2^63, 2^64)` in generated arguments becomes a float when coercion rewrites the arguments (Python keeps the digits); `ink-text`'s `parse_int` rejects such text, so a string coerced to `integer` or `number` falls through to the next type |
| `serde_json` does not read `NaN`, `Infinity` or lone surrogate escapes | such arguments are left uncoerced (Python reads them) |
| Python's non-ASCII decimal digits | not accepted by `parse_int` / `parse_float` (`ink-text`) |
