"""System architecture diagram for inkling-cmp-engine (README)."""
from open_diagram import Diagram, Note, Panel, Tile, Tiles
from od_themes import dark

CLIENT_TILES = [
    "/v1/chat/completions",
    "/v1/completions",
    "Streaming SSE",
    "Tool calls & thinking",
]
FRONTEND_TILES = [
    "ink-server + ink-api",
    "ink-text renderer",
    "ink-parse tools/think",
    "Prometheus /metrics",
]
SCHED_TILES = [
    "Paged KV pool",
    "Prefix checkpoints",
    "150 ms steal grace",
    "Chunked prefill/decode",
]
TARGET_TILES = [
    "18.98 GB NVFP4_N1S1",
    "40 SWA + 8 global",
    "16-row decode attn",
    "TC prefill rel-bias",
]
SPEC_TILES = [
    "DSpark Markov (default)",
    "Verbatim lookup (8)",
    "DFlash2 (opt-in)",
    "128-bit Leviathan",
]


def build():
    d = Diagram(
        "inkling-cmp-engine Architecture",
        subtitle="Pure-Rust + CUDA sm_80 single-model serving stack for 1-8x NVIDIA CMP 170HX (64 GB, 140 W cap)",
        tag="v1.0.0",
        note=Note("Scaling", "One card per replica (127.0.0.1:8400..8407); 4.13M BF16 or 8.26M+ FP8 KV tokens across 8 cards."),
        theme=dark(),
    )

    (clients,), (frontend,), (sched,), (target, spec) = d.columns(
        [
            [
                Panel(
                    "OpenAI Clients",
                    Tiles(CLIENT_TILES, cols=1),
                    tone="neutral",
                    sub="Loopback HTTP / load balancer",
                )
            ],
            [
                Panel(
                    "Pure-Rust Front End",
                    Tiles(FRONTEND_TILES, cols=1),
                    sub="Zero Python, Tokio async",
                )
            ],
            [
                Panel(
                    "Prefix Scheduler",
                    Tiles(SCHED_TILES, cols=1),
                    sub="ink-sched (up to 8 slots)",
                )
            ],
            [
                Panel(
                    "Target Looped MoE",
                    Tiles(TARGET_TILES, cols=2),
                    tone="dark",
                    sub="kpack (367 kernels, 0 spills)",
                ),
                Panel(
                    "Exact Speculative Path",
                    Tiles(SPEC_TILES, cols=2),
                    tone="accent",
                    sub="Ptok3 gated verify (1..16 rows)",
                ),
            ],
        ],
        gap=88,
        row_gap=52,
        valign="top",
    )

    d.group("CUDA sm_80 Runtime (ink-runtime, 140 W)", [target, spec], sub="Embedded byte-reproducible cubins")

    d.link(clients, frontend, label="HTTP / SSE")
    d.link(frontend, sched, label="token IDs")
    d.link(sched, target, label="prefill / verify")
    d.link(sched, spec, start="bottom", end="left", label="draft round")
    d.link(spec, target, start="top", end="bottom", label="<= 15 drafts")
    return d
