# engine/rust: Rust rewrite workspace (docs/RUST-PLAN.md)

State (2026-11-23): the front end (`ink-text`, `ink-parse`, `ink-api`, `ink-backend`, bin `ink-server`; tools
`ink-textcli`, `ink-replay`) and the scheduler (`ink-sched`) are cut over: the C++ `engine/server/` is deleted
(cut-over U1 `bench/raw/2026-11-23-rust-sched-gate/`, U2 `bench/raw/2026-11-23-rust-fe-gates/`). `ink-record` and
`ink-rt-api::capi` (client of the interim C ABI) are done; `ink-runtime` has `RuntimeAdapter` (package I1:
`ink-server --backend engine` serves the C++ engine through `ink_rt.h`, configuration `M3`; record
`bench/raw/2026-11-23-rust-I1-served/`); `ink-cuda` (R1), `ink-kernels`, `ink-model`, the Rust `Engine` and the other
`ink-tools` bins are in progress or stubs. Each crate's README has its gate and port notes.

Everything that compiles runs in docker, `--network none`, workspace and `vendor/` read-only (AGENTS hard rule 6),
`--cap-drop ALL --security-opt no-new-privileges --pids-limit 4096 --memory 32g` (`INK_MEM`), root filesystem read-only,
`CARGO_HOME` and `/tmp` per-run tmpfs (nothing a build script writes outlives its container; `target/` is the only
writable mount). Cargo flags are `--offline --locked` everywhere (`--frozen` is the same pair).

| Step | Command (from `engine/rust/`) |
|---|---|
| Build image | Root `Dockerfile` (`docker build -t inkling-cmp-engine:v1.0.0 .`) |
| Fetch crates (once per dependency change, network, runs no crate code) | `scripts/vendor.sh`: fresh tmpfs cargo home, workspace read-only, output to `target/vendor-new`, checked against `Cargo.lock` (`scripts/lock-check.py`), then swapped into `vendor/`. A new `Cargo.lock` is its own step: `rm Cargo.lock; scripts/vendor.sh --generate-lock`, review the diff |
| Verify the vendor tree | `scripts/vendor-check.sh` (`build.sh` and `rt.sh` do it first): refuses symlinks and special files, hashes file modes, contents, `Cargo.lock` and `.cargo/config.toml` |
| Write the build identity | `scripts/build-id.sh` (optional; `ink-record/build.rs` auto-derives from git + `Cargo.lock` when absent) |
| Build / test / lint | `scripts/build.sh cargo build --workspace --offline --locked` (same for `test`, `clippy --all-targets ... -- -D warnings`) |
| Format | `scripts/fmt.sh` (`--check` to verify) |
| V1 acceptance from a clean checkout | `scripts/check.sh` |
| Run server / tools | `./start.sh` or `cargo build --release --locked -p ink-server -p ink-tools` |

Not committed: `vendor/`, `target/`, `BUILD_ID.json`. Committed: `Cargo.lock`, `vendor.sha256`
(`<sum>  <n> files`; `sum` = sha256 of the lines `modes:<h>`, `files:<h>`, `lock:<h>`, `config:<h>`: sorted `mode path` list
of every vendor file, sorted `sha256sum` list of them, `sha256(Cargo.lock)`, `sha256(.cargo/config.toml)`),
`docker/IMAGE.lock`. Record schema: `RECORDS.md`.
