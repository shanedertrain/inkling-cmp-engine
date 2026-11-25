# Baseline: vLLM config B (what the engine must beat)

Frozen since the first Inkling project; production fallback and source of goldens. Launch script:
the vLLM Config B single-card replica launcher.

| Setting | Value |
|---|---|
| Image | `vllm/vllm-openai@sha256:16721463e5ae0c81623c65c50f5b8d4e37710d9bc4f6f4f7512ec1939096272f` |
| Overlay | Hiutaky/vllm Ampere port (`6e9d1cce`) mounted over `dist-packages/vllm` |
| Model | `thinkingmachines/Inkling-2-Mini` rev `ed40e7da`, native weights (NVFP4 experts layers 0-19, MXFP8 20-23, via the Humming kernel), bf16 |
| Layout | TP1, one card per replica, `--network host`, `--cpus 12`, `--shm-size 16g`, 127.0.0.1 |
| Flags | `--dtype bfloat16 --limit-mm-per-prompt.image=0 --enable-auto-tool-choice --tool-call-parser inkling --reasoning-parser inkling --max-model-len 131072 --max-num-seqs 8 --compilation-config.cudagraph_mode=FULL_DECODE_ONLY --speculative-config.method=dspark --speculative-config.num_speculative_tokens=7` |
| Env | `HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1` |
| Untouched defaults | api-server-count 1, renderer workers 1, prefix-cache retention 0, default batched-token budget (2,048), no depth schedule, no CPU pinning |
| Load | 4 agent sessions per card (owner target); 140 W cap, stock clocks/fans |

## Numbers to beat (cite the source, not this table, in reports)

| Metric | Value | Source |
|---|---|---|
| Decode step and tok/s per (n,k) and context, % of byte ceiling | 72 cells; e.g. (4,7) 45K: 45.2 ms, 416 tok/s | `cmp170hx-engine-opt/THEORY.md` §5 + Correction (floors at 970-1,130 GB/s) |
| Phase shares of the step (experts, attention, dense, glue, drafter, host) | experts ~48 %; host 0.3-1.3 % | `wave0/job-P-decode-profile.md` (+ Correction) |
| Hardware rates at 140 W (6D vs 67 cards) | HBM read, FFMA 13.7 TFLOPS, bf16 mma ~100 TFLOPS, thin GEMV 560-740 GB/s | `wave0/hardware-rates.md` |
| Expert kernel stand-ins vs Humming; weight formats | exact NVFP4 power-bound ~790 GB/s; int8 path 1.6x but lossy | `wave0/job-K0-decode-precheck.md`, `wave0/job-W-weight-format.md` |
| TTFT and load (agent replay) | TTFT p99 5.41-5.75 s at N = 4 on the panel; live tail 18 s | `wave0/job-F-ttft-and-load.md:90-92,186` |
| Prefill and API path | 0.245 ms/token at N = 4; 61 ms + 0.217 ms/token at 32K | `wave0/job-F…:44`, `wave0/job-Q-prefill-and-api.md:20` |
| Agent suites | Hard suite pooled 58.8 %; easy 54-task suite | internal agent benchmark suite |
| Run-to-run noise (A/A, 4 cards, several boots) | Pending at handoff | `wave0/job-AA-noise.md` (when written), `raw/2026-11-22-job-AA/` |

Baseline summary recorded prior to
Supermicro chassis: one A/A re-baseline period before comparing anything to pre-move numbers.
