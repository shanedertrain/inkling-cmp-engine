"""Single speculative decode round diagram for inkling-cmp-engine (README)."""
from open_diagram import Diagram, Note, Panel, Tiles
from od_themes import dark

BATCH_TILES = [
    "Up to 8 active slots",
    "Anchor token + pos",
    "Paged global KV",
    "SWA-1024 rings",
]
DSPARK_TILES = [
    "3-layer draft model",
    "7-step Markov loop",
    "top_p = 0.90 filter",
    "58-node CUDA graph",
]
LOOKUP_TILES = [
    "Verbatim n-gram match",
    "--lookup 8 --lookup-min 8",
    "Ptok3 gated width",
    "Zero VRAM overhead",
]
VERIFY_TILES = [
    "1..16 rows per slot",
    "NVFP4_N1S1 MoE GEMM",
    "Single-pass 16-row attn",
    "Fused router + AttnRes",
]
ACCEPT_TILES = [
    "128-bit Leviathan",
    "Rejection + residual",
    "Commit accepted KV",
    "Instant draft rollback",
]


def build():
    d = Diagram(
        "Speculative Decode Round (140 W Budget)",
        subtitle="CUDA-graphed draft, target verify, and exact Leviathan acceptance on 1x CMP 170HX",
        tag="LOSSLESS",
        note=Note("Verified exact", "0/8,000 greedy argmax mismatches; 13/13 sdist tests (425,984 draws, 0 zero-prob draws)."),
        theme=dark(),
    )

    (batch,), (dspark, lookup), (verify,), (accept,) = d.columns(
        [
            [
                Panel(
                    "Active Batch",
                    Tiles(BATCH_TILES, cols=1),
                    tone="neutral",
                    sub="N <= 8 sessions / card",
                )
            ],
            [
                Panel(
                    "DSpark Drafter",
                    Tiles(DSPARK_TILES, cols=1),
                    tone="dark",
                    sub="Default (or DFlash2 opt-in)",
                ),
                Panel(
                    "Verbatim Lookup",
                    Tiles(LOOKUP_TILES, cols=1),
                    sub="Appended suffix drafts",
                ),
            ],
            [
                Panel(
                    "Target Verify",
                    Tiles(VERIFY_TILES, cols=1),
                    tone="dark",
                    sub="Bit-invariant vs 1-row",
                )
            ],
            [
                Panel(
                    "Exact Acceptance",
                    Tiles(ACCEPT_TILES, cols=1),
                    tone="accent",
                    sub="3.00-3.14 tok/rd mean",
                )
            ],
        ],
        gap=108,
        row_gap=44,
        valign="top",
    )

    d.group("Proposal (up to 15 drafts)", [dspark, lookup])

    d.link(batch, dspark, label="anchor")
    d.link(batch, lookup, start="bottom", end="left", label="token history")
    d.link(dspark, verify, label="<= 7 drafts")
    d.link(lookup, verify, start="right", end="bottom", label="<= 8 suffix")
    d.link(verify, accept, label="target logits")

    lane_y = max(p.y + p.h for p in d.placed) + 72
    d.link(accept, batch, start="bottom", end="bottom", via_y=lane_y, label="accepted tokens -> next anchor")
    return d
