//! Inkling-2-Mini dimensions and the layer visit plan (port of `engine/native/src/model.hpp` @ 66fe353).

/// Hidden size.
pub const H: usize = 2304;
/// Query heads.
pub const NH: usize = 24;
/// Key/value heads.
pub const KVH: usize = 8;
/// Head dimension.
pub const HD: usize = 128;
/// Relative-position projection width per head.
pub const DREL: usize = 16;
/// Relative-logit extent.
pub const REL_EXT: usize = 1024;
/// Sliding window of the local layers.
pub const WINDOW: usize = 1024;
/// Expert intermediate size.
pub const INTER: usize = 1152;
/// Routed experts per layer.
pub const NEXP: usize = 128;
/// Experts per token.
pub const TOPK: usize = 8;
/// Short-convolution taps.
pub const CONVK: usize = 4;
/// Checkpoint layers.
pub const NLAYERS: usize = 24;
/// Layers per block.
pub const BLOCK: usize = 6;
/// First layer of the looped middle.
pub const LOOP_START: usize = 6;
/// One past the last layer of the looped middle.
pub const LOOP_END: usize = 18;
/// Passes over the looped middle.
pub const PASSES: usize = 3;
/// Padded vocabulary (rows of `embed` / `unembed`).
pub const VOCAB_PAD: usize = 201_024;
/// Vocabulary.
pub const VOCAB: usize = 200_058;
/// Rank of the loop controllers' gates.
pub const GATE_RANK: usize = 64;
/// Query projection width (3072).
pub const QD: usize = NH * HD;
/// Key/value projection width (1024).
pub const KD: usize = KVH * HD;
/// Layer visits of one forward pass (48).
pub const NVISITS: usize = LOOP_START + PASSES * (LOOP_END - LOOP_START) + (NLAYERS - LOOP_END);

/// RMS-norm epsilon.
pub const EPS: f32 = 1e-6;
/// Query/key L2 clip.
pub const QK_CLIP: f32 = 22.627_417;
/// Relative-logit clip.
pub const REL_CLIP: f32 = 8.0;
/// Log-length scale alpha.
pub const LOG_ALPHA: f32 = 0.1;
/// Log-length floor.
pub const LOG_FLOOR: f32 = 8192.0;
/// Output logit soft cap.
pub const LOGIT_SOFTCAP: f32 = 30.0;
/// Router scale.
pub const ROUTE_SCALE: f32 = 8.0;

/// Layers with full (global) attention; the others use the sliding window.
pub fn is_full_layer(l: usize) -> bool {
    matches!(l, 5 | 11 | 17 | 23)
}

/// One step of the forward pass: a checkpoint layer and the pass over it (0 outside the loop).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Visit {
    /// Checkpoint layer.
    pub layer: usize,
    /// Pass (0..[`PASSES`] inside the loop, else 0).
    pub pass: usize,
}

/// The 48 visits in order: layers 0-5, three passes over 6-17, then 18-23.
pub fn visit_plan() -> Vec<Visit> {
    let head = (0..LOOP_START).map(|layer| Visit { layer, pass: 0 });
    let middle = (0..PASSES).flat_map(|pass| (LOOP_START..LOOP_END).map(move |layer| Visit { layer, pass }));
    let tail = (LOOP_END..NLAYERS).map(|layer| Visit { layer, pass: 0 });
    head.chain(middle).chain(tail).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derived_sizes_match_model_hpp() {
        // model.hpp comments: QD 3072, KD 1024, NVISITS 48.
        assert_eq!((QD, KD, NVISITS), (3072, 1024, 48));
        // QK_CLIP = 22.627416997969522f (sqrt(512)), the same f32.
        assert_eq!(QK_CLIP, 22.627_416_997_969_522_f64 as f32);
    }

    #[test]
    fn visit_plan_order() {
        let v = visit_plan();
        assert_eq!(v.len(), NVISITS);
        let pairs: Vec<(usize, usize)> = v.iter().map(|x| (x.layer, x.pass)).collect();
        let mut want: Vec<(usize, usize)> = (0..6).map(|l| (l, 0)).collect();
        for p in 0..3 {
            want.extend((6..18).map(|l| (l, p)));
        }
        want.extend((18..24).map(|l| (l, 0)));
        assert_eq!(pairs, want);
        assert_eq!(pairs[6], (6, 0));
        assert_eq!(pairs[18], (6, 1));
        assert_eq!(pairs[41], (17, 2));
        assert_eq!(pairs[42], (18, 0));
    }

    #[test]
    fn eight_full_and_forty_sliding_visits() {
        // engine.cu Engine::Engine: INK_REQUIRE(nf == 8 && nr == 40).
        let full = visit_plan().iter().filter(|v| is_full_layer(v.layer)).count();
        assert_eq!((full, NVISITS - full), (8, 40));
        let layers: Vec<usize> = (0..NLAYERS).filter(|&l| is_full_layer(l)).collect();
        assert_eq!(layers, [5, 11, 17, 23]);
    }
}
