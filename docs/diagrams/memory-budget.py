"""64 GB CMP 170HX VRAM budget comparison diagram: BF16 default vs --kv-fp8 (README)."""
from open_diagram import Diagram, Note, Panel, Tiles
from od_themes import dark

SHARED_TILES = [
    "Target weights: 18.98 GB",
    "NVFP4_N1S1 (-1.27 GB)",
    "DSpark + fold: ~3.8 GB",
    "SWA rings + CUDA graphs",
]
BF16_TILES = [
    "504 global KV pages",
    "516,096 tokens / card",
    "4.13M tokens (8 cards)",
    "KL 0.00477 (98.04% top-1)",
]
FP8_TILES = [
    "1,008+ global KV pages",
    "1.03M+ tokens / card",
    "8.26M+ tokens (8 cards)",
    "4x8 120K decode: -16.6%",
]


def build():
    d = Diagram(
        "64 GB CMP 170HX Memory & KV Modes",
        subtitle="Fixed 18.98 GB weight footprint leaves room for 516K lossless BF16 or 1.03M+ opt-in FP8 KV tokens per card",
        tag="64 GB HBM2e",
        note=Note("DFlash2 opt-in", "Replacing DSpark with --draft-dflash2 frees an extra ~3.3 GiB VRAM per card."),
        theme=dark(),
    )

    (shared,), (bf16, fp8) = d.columns(
        [
            [
                Panel(
                    "Shared Card Footprint",
                    Tiles(SHARED_TILES, cols=1),
                    tone="neutral",
                    sub="Native NVFP4 + MXFP8 (no re-quant)",
                )
            ],
            [
                Panel(
                    "Default: Lossless BF16 KV",
                    Tiles(BF16_TILES, cols=2),
                    tone="dark",
                    sub="Best N=4 M1: 323.8-360.6 tok/s (+25% vs Config B)",
                ),
                Panel(
                    "Opt-In: --kv-fp8 (E4M3 Global KV)",
                    Tiles(FP8_TILES, cols=2),
                    tone="accent",
                    sub="Fits 4 x 262K contexts / card (lossy, KL 0.0262)",
                ),
            ],
        ],
        gap=132,
        row_gap=48,
        valign="top",
    )

    d.link(shared, bf16, label="default")
    d.link(shared, fp8, start="bottom", end="left", label="KV_FP8=1")
    return d
