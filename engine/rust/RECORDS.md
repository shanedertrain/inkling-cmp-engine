# Record schema `ink-record/1`

Written by `ink-record` (`crates/ink-record`); read by the Python analysis. One JSON object per line in
`<name>.jsonl` (`<name>.dirty.jsonl` for a dirty tree). Never overwritten (`create_new`). Rule: docs/RUST-PLAN.md section 6.

| Key | Content |
|---|---|
| `schema` | `"ink-record/1"` |
| `time` | UTC, RFC 3339 |
| `identity` | below |
| `context` | `argv` (list), `config` (object of the run), `gpu` (`{index, uuid, vbios}` or null), `power_cap_w` (int or null), `baseline` (what the numbers are compared with), `label` |
| `data` | the measurement, free-form per tool |

`identity` (key order as written):

| Key | Content |
|---|---|
| `engine` | `"native"` or `"rust"`: implementation of the **runtime** (it produces the numbers) |
| `version`, `commit`, `dirty`, `diff` | of the runtime; `commit` = 40 lowercase hex; `diff` = sha256 hex, present exactly when `dirty` |
| `config` | mixed configuration `M0`-`M5` |
| `lock_sha256`, `vendor_sha256` | sha256 of `Cargo.lock` and the tree hash of `vendor.sha256` (vendor tree + lock + `.cargo/config.toml`) of the Rust build that wrote the record; 64 lowercase hex, from `BUILD_ID.json` `lock` / `vendor`; validated like `commit` |
| `components` | keys `frontend`, `scheduler`, `tools`, `runtime`, `drafter`, `kpack`; a component not in the process is omitted; `runtime` is always present and equals the top-level fields. Value: `{impl, version, commit, dirty, [diff], [abi]}`, or `{id}` (sha256 hex) for `kpack` |

Writer rules (`RecordWriter::create`): the identity must validate (well-formed commit, `dirty` iff `diff`, `runtime`
equal to the top level) else `RecordError::Identity`; any dirty component gives `RecordError::Dirty` and no file unless
`DirtyPolicy::Allow`, which writes `<name>.dirty.jsonl` (never quote such a record). Names with a path separator, empty
or starting with `.` are refused.

Build identity: `scripts/build-id.sh` writes `BUILD_ID.json` (`lock`, `vendor` as above, `commit`, `dirty` = `git status --porcelain -- engine`
non-empty, `diff` = sha256 of `git diff HEAD -- engine` plus name and content of every untracked file, else null).
`ink-record/build.rs` reads it and fails the build when it is missing or malformed (including `lock`/`vendor`). `ink_record::BUILD_LINE`
(`rust version=X commit=<40 hex> dirty=<0|1> diff=<hex|->`) is what each binary prints for `--version`.
