# ink-tools

Binaries that drive the engine or the front end from the command line. Library part (`src/lib.rs`): the helpers the
golden-set tools share.

| Binary | What | Package, gate |
|---|---|---|
| `ink-gold` | Golden-set runner (port of the C++ `ink_gold`) | T1, G-T1 |
| `ink-inv` | Batch-, chunk- and mixed-call invariance tests (port of the C++ `ink_inv`) | T1, G-T2 |
| `ink-optest` | Op-by-op comparison against the S1 reference's dumped tensors, `chain` and `iso` (port of the C++ `ink_optest`) | T1, G-T3 |
| `ink-spec` | Speculative decoding with the DSpark drafter, the modes of `engine/native/tools/spec.cpp`: `greedy` (path check, `--trace`: drafts and accept results per round), `time` (free-running rounds: ms per round, draft / verify split, tokens/s, NVML), `golden` (timed golden prompts, tokens/s and tokens/J); every session verifies 7 drafts (`--tau off`) | A2 (greedy), T2 (time, golden); gates G5, G-T4 |
| `ink-oracle` | Oracle record set of the runtime (E1-E4) | A1 |
| `ink-load` | E1 of the Rust loader (`ink-model`): weight sums and arena map | R4 |
| `ink-textcli`, `ink-parsecli`, `ink-replay` | Front-end tools | U2 |
| `ink-bench` | Timing tool (port of the C++ `ink_bench`): decode-step cells with NVML window and per-region split, `--ablate` cells, `--mode prefill\|pmix\|mixed`; same argv and JSON lines (`summ_bench.py`, `summ_ablate.py` read them) | T2, G-T4 |

## Golden-set tools

All run over `dyn ink_rt_api::Runtime`; today that is `CapiRuntime`, the C++ engine through `ink_rt.h` v2
(`libinkrt_c.so`, config `M2`). Output on stdout is the C++ tool's JSON line format, byte for byte (numbers are
printed with C's `%g`, `src/fmt.rs`), so `engine/native/tests/{gold_post,summ_optest}.py` and
`engine/tools/goldens/compare.py` read it unchanged. Lines that carry the identity start with
`"engine","engine_version","commit","dirty"` of the runtime library (`engine::identity_fields`). Wall-clock fields
(`prefill_s`, `decode_s`, `seconds`) are the only fields that differ between two runs of the same engine.

Engine flags (`engine::EngineArgs`, the C++ tools' flags plus the library): `--rt-lib` (the library), `--model`,
`--gemm-algos` (the cuBLASLt pin file: loaded when it exists, written after start-up otherwise), `--slots`, `--tune-m`
(default 32), `--glue`, `--graphs` (`--moe`, `--gate-tail`, `--hum-cache`, `--hum-weights` deleted in delta 1: one expert path,
one gate tail); the production pin file is `engine/pins/gemm-algos.json`. Per tool: `--cases --only --skip --tf-batch
--greedy-sessions --pages` (`ink-gold`), `--cases --max-chunk` (`ink-inv`; 4096 / 8192 add the long-chunk tests),
`--dump --mode --save --out-dir --pages` (`ink-optest`), `--mode greedy --cases --sessions --only --path --fold --kv8
--ctx --draft-ctx --pages --tie-margin --trace` (`ink-spec`; `--slots` must equal `--sessions`). Record flags:
`--record-dir`, optionally `--gpu --uuid --vbios --power-cap-w` and `--allow-dirty` (debugging: writes
`<tool>.dirty.jsonl`).

```sh
ink-gold   --rt-lib libinkrt_c.so --gemm-algos pins.json --record-dir rec/ --model /model --cases cases.txt > gold.jsonl
ink-inv    --rt-lib libinkrt_c.so --gemm-algos pins.json --record-dir rec/ --model /model --cases cases.txt > inv.jsonl
ink-optest --rt-lib libinkrt_c.so --gemm-algos pins.json --record-dir rec/ --dump dumps/short --mode iso > out.jsonl
ink-spec   --rt-lib libinkrt_c.so --gemm-algos pins.json --record-dir rec/ --slots 4 --sessions 4 --cases cases.txt \
           --fold /s2f/fold.bin --path 128 --trace trace.jsonl > spec.jsonl
```

`ink-optest --mode iso` uses the tap write-back of `ink_rt.h` v2: after each compared op the reference's rows replace
the engine's (when the dump covers the tensor from its first row), so the next op sees the reference's inputs.

### Record (`--record-dir`)

Every run writes `<tool>.jsonl` there (`ink-record/1`): the identity of this build (`components.tools`) and of the
library (`components.runtime`, `abi`), `argv`, the engine flags, GPU, power cap, and a closing `done` line with counts
and engine counters. The writer refuses a dirty component. `ink-gold` also writes, when its row order is the
oracle's (`--tf-batch 1`, `--greedy-sessions 4`), the E4 files of the oracle record set: `e4-logits.jsonl`,
`e4-logits-full.f32`, `e4-greedy-tokens.json` (`src/e4.rs`, shared with `ink-oracle record`), so the run can be
compared with `cmp` against `bench/raw/2026-11-23-oracle-native-v2/oracle/`.

### Tests

`tests/golden_tools.rs` runs the binaries over the C stub of `ink-rt-api` (`tests/c/ink_rt_stub.c`, no GPU): expected
rows, E4 files, record identity, refusal of an existing record, zero differing bits on a batch-invariant engine, a
deliberately wrong dump row, the iso write-back. Unit tests: `%g` formatting against C's printf, top-k and hashes,
`cases.txt` parsing, the E4 writer.

## Timing tools (`ink-bench`, `ink-spec --mode time|golden`)

Same flags as the C++ tools plus the engine and record flags above; output lines in the C++ format on stdout (`src/line.rs`:
keys in call order, `%.7g`), each also written to the record as `{"line": ...}`. NVML window fields come from
`ink_cuda::Sampler` (`src/tele.rs`: power, SM and memory clock, temperature, clock-event reasons, energy counter at 10 Hz).
Differences from the C++ tools, all from what `ink_rt.h` v2 offers:

| Item | C++ tool | Rust tool |
|---|---|---|
| Stream sync after a step | `cudaStreamSynchronize` | a 4-byte device read (`SpecRuntime::download`) |
| Per-component split | events between the engine's marks, `_gpu_span` = first to last mark | `Runtime::marks` (same events, summed per name); `_gpu_span` = sum of the intervals |
| `--ablate` | `set_ablation` before the first graph | the seam names regions of captured graphs: one step that keeps nothing captures the graph first |
| `--fine`, graph-region count on stderr | engine fields | refused / dropped: need `ink_rt_set("profile_fine")` and a region-count call |
| `spec --mode time`: draft / verify split | cudaEvents in every 10th round | two device waits in every 10th round; those rounds are left out of the step statistics |
| `spec --gate`, `--tau`, `--call-rows` | gated verify width, the verify call's rows rounded up to `S m` | the same (`SpecRuntime::verify_widths`, `Drafter::conf`, `CallRows::fill`) |
| `spec --conf-trace` | per-round probabilities and survivors | refused: needs `ink_draft_survivors` |
| `--slots` | `max(sessions + 1, 4)` (bench), `sessions` (spec) | the same, derived from `--sessions` in `ink-bench`; `ink-spec` requires `--slots == --sessions` |
