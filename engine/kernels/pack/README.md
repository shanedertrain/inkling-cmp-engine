# engine/kernels/pack: the kernel pack of the Rust runtime

The CUDA kernels stay CUDA and have one source (`engine/kernels/`). This directory compiles the kernel headers (in place,
no copies) into two cubins the Rust runtime embeds (`ink-kernels`), and generates its typed launchers. Gate E0 checks the
pack's per-kernel `.text` SASS hashes against the committed reference `e0-ref.jsonl` (seeded from the final C++ cut-over
oracle at `bench/raw/2026-11-24-rust-cutover/`). Plan: docs/RUST-PLAN.md 3.2, package R2.

## After changing any kernel or launch struct: one command

```
engine/kernels/pack/regen.sh                 # verify SASS unchanged against e0-ref.jsonl
engine/kernels/pack/regen.sh --update-ref    # after an intentional kernel change; updates e0-ref.jsonl
git add engine/rust/crates/ink-kernels engine/kernels/pack/e0-ref.jsonl
```

It builds the pack with nvcc (CUDA 13.x), runs `xtask gen-kernels`, runs E0 against `e0-ref.jsonl`, and only on E0 PASS (or
`--update-ref`) writes the regenerated files into your tree. Records: `engine/rust/target/kpack-regen/` (or the
directory given as argument).

`engine/rust/scripts/check.sh` starts with `check_fresh.py` (no nvcc, a few ms): it fails, naming the changed files,
when any file the pack is compiled from differs from what `ink-kernels` was generated from. Run it alone with
`python3 -I engine/kernels/pack/check_fresh.py`.

| You changed | Also edit | Then |
|---|---|---|
| a kernel body or a header it includes | nothing | `regen.sh` |
| a kernel's parameters | nothing (the Rust launcher's signature follows; Rust call sites stop compiling until updated) | `regen.sh` |
| a struct passed to a kernel (by value or behind a pointer) | its field list in `structs.def` if fields were added, removed or renamed | `regen.sh` (`layout_emit` fails on a missing field) |
| added a kernel or a template instance the engine launches | `keys.toml`: `key = "C++ instance"` in `[engine]` or `[drafter]` (key: lowercase, `::`/`<`/`, ` -> `_`) | `regen.sh` (E0 lists a kernel the pack lacks; the generator lists one without key) |
| removed a kernel | its `keys.toml` line | `regen.sh` |
| a kernel now needs a cooperative launch | `cooperative = [...]` in `keys.toml` | `regen.sh` |

## Files

| File | What |
|---|---|
| `pack_engine.cu`, `pack_drafter.cu` | The two TUs: the headers of `engine.cu` / `draft.cu`, plus `keep_<tu>.inc` (generated from `keys.toml`) taking each instance's address, so exactly those instances are compiled |
| `keys.toml` | Stable key -> kernel instance (C++ name as the demangler prints it, no parameters, no `(anonymous namespace)::`), per TU; the cooperative list |
| `structs.def` | Every struct a kernel takes, with its fields (X-macro for `layout_emit.cpp`) |
| `layout_emit.cpp` | Host program: size, alignment, field offsets and types from the compiler -> `layout.json`; fails when the fields do not tile the struct |
| `build.sh OUT` | The pack: keep files, kpack id (sha256 of every input file from `nvcc -M`, nvcc version, flags), cubins, `layout.json`, `manifest.json`. No device symbol is added to the cubins (an extra global changed `k_cos_sin`'s SASS: `bench/raw/2026-11-23-rust-R2-pack/pack1/`) |
| `e0.sh LIB PACK OUT`, `e0_compare.py` | Gate E0: per-kernel `.text` sha256 of the pack vs the fatbin of `libinkrt_c.so` |
| `regen.sh` | The one command above |
| `check_fresh.py` | Staleness check without nvcc (manifest inputs vs this tree, kpack id vs `gen.rs`) |

Generator: `engine/rust/crates/xtask/src/kernels/` (`xtask gen-kernels PACK OUT`): matches keys to cubin kernels (one
each, both ways), checks every parameter's offset and size against nvcc's `KPARAM_INFO` and every struct field's
offset against the compiler's, writes `gen.rs` (kernel table, `#[repr(C)]` mirrors with layout assertions, one launcher
per key). Launchers are `unsafe fn` (R3 decision; the contract is in the `ink-kernels` crate docs, "Safety of the
launchers"). `layout_emit` also counts each struct's members by aggregate initialisation, so a member hidden in padding
cannot be left out of `structs.def` (R3 found `projs::Job::cta0` missing that way). `gen.rs` records the FNV-1a 64 of the generator's sources; `check_fresh.py` fails when the generator changed after the
last regeneration. Cubins are byte-reproducible (`build.sh` stages kernel sources and `keep_*.inc` into a fixed path
with zeroed mtimes before `nvcc -lineinfo`); `ink-kernels`' `build.rs` verifies both `kpack_id` and the cubin sizes /
FNV-1a 64 checksums against `gen.rs`.
