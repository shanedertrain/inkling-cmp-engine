// Ported from engine/server/src/backend/native_adapter.{h,cpp} @ 4171493 (FakeRuntime, FakeCost).
//! Deterministic runtime on a virtual clock: tokens from a hash, step time from the cost model.

use ink_backend::TokenLogprobs;

use crate::plan::{SegKind, SegResult, StepPlan};
use crate::runtime::{Drafting, ModelRuntime, RuntimeCaps, StepError};

/// Step-cost model of [`FakeRuntime`]; defaults are config B measurements (`SCHEDULER.md` "Interleave rule").
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FakeCost {
    /// Fixed cost per step, ms (P10 README:41).
    pub base_ms: f64,
    /// Per decode row, ms (P10 README:41).
    pub row_ms: f64,
    /// Per prefill token, ms (P9 README:96-97).
    pub prefill_tok_ms: f64,
    /// Per prefill token per 1,000 cached keys, ms (P9 README:97-98).
    pub attn_ms_per_tok_kkey: f64,
    /// Multiplies everything (robustness: the scheduler's estimate against a slower truth).
    pub scale: f64,
}

impl Default for FakeCost {
    fn default() -> Self {
        Self {
            base_ms: 25.4,
            row_ms: 0.617,
            prefill_tok_ms: 0.171,
            attn_ms_per_tok_kkey: 0.00245,
            scale: 1.0,
        }
    }
}

/// Deterministic tokens and the step-cost model on a virtual clock (tests, policy simulation).
///
/// Token at position `pos` of sequence `seq` is [`FakeRuntime::token_at`]; a decode segment of width `w` accepts
/// `1 + token_at(seq, pos0) % w` tokens. Each step advances the clock by [`FakeRuntime::step_cost_ms`].
#[derive(Debug, Clone)]
pub struct FakeRuntime {
    cost: FakeCost,
    caps: RuntimeCaps,
    clock_ms: f64,
    steps: u64,
    last_step_ms: f64,
}

impl Default for FakeRuntime {
    fn default() -> Self {
        Self::new(FakeCost::default())
    }
}

impl FakeRuntime {
    /// A fake runtime with every capability (external pages, drafter, conv checkpoints) and `engine/native` limits.
    #[must_use]
    pub fn new(cost: FakeCost) -> Self {
        Self {
            cost,
            caps: RuntimeCaps {
                external_pages: true,
                drafter: Drafting::Any,
                conv_checkpoints: true,
                ..RuntimeCaps::default()
            },
            clock_ms: 0.0,
            steps: 0,
            last_step_ms: 0.0,
        }
    }

    /// The deterministic token of sequence `seq` at position `pos`, in `[1000, 151000)`.
    #[must_use]
    pub fn token_at(seq: u64, pos: i64) -> i32 {
        let mut h = seq.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (pos as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
        h ^= h >> 29;
        i32::try_from(h % 150_000).expect("below 150000") + 1000
    }

    /// Lets virtual time pass without a step.
    pub fn advance(&mut self, ms: f64) {
        self.clock_ms += ms;
    }

    /// Steps run so far.
    #[must_use]
    pub fn steps(&self) -> u64 {
        self.steps
    }

    /// Virtual duration of the last step, ms.
    #[must_use]
    pub fn last_step_ms(&self) -> f64 {
        self.last_step_ms
    }

    /// Virtual duration of `plan`, ms.
    #[must_use]
    pub fn step_cost_ms(&self, plan: &StepPlan) -> f64 {
        let c = &self.cost;
        let mut ms = c.base_ms + c.row_ms * f64::from(plan.decode_rows);
        for g in plan.segs.iter().filter(|g| g.kind == SegKind::Prefill) {
            let keys_k = (f64::from(g.pos0) + f64::from(g.width) / 2.0) / 1000.0;
            ms += f64::from(g.width) * (c.prefill_tok_ms + c.attn_ms_per_tok_kkey * keys_k);
        }
        ms * c.scale
    }
}

impl ModelRuntime for FakeRuntime {
    fn caps(&self) -> RuntimeCaps {
        self.caps
    }

    fn step(&mut self, plan: &StepPlan) -> Result<Vec<SegResult>, StepError> {
        let mut out = Vec::new();
        for g in plan.segs.iter().filter(|g| g.samples) {
            let (n, first_pos) = match g.kind {
                // decode: the tokens after the input token's position
                SegKind::Decode => {
                    let accepted =
                        1 + u64::from(Self::token_at(g.seq, i64::from(g.pos0)).unsigned_abs()) % u64::from(g.width);
                    (accepted.clamp(1, u64::from(g.width)), i64::from(g.pos0) + 1)
                }
                SegKind::Prefill => (1, i64::from(g.pos0) + i64::from(g.width)),
            };
            let tokens: Vec<i32> = (0..n as i64).map(|i| Self::token_at(g.seq, first_pos + i)).collect();
            let logprobs = if g.sampling.num_logprobs.is_some() {
                tokens
                    .iter()
                    .map(|&t| TokenLogprobs {
                        logprob: -0.5,
                        rank: 1,
                        top_ids: vec![t],
                        top_logprobs: vec![-0.5],
                    })
                    .collect()
            } else {
                Vec::new()
            };
            out.push(SegResult {
                seq: g.seq,
                tokens,
                logprobs,
            });
        }
        self.last_step_ms = self.step_cost_ms(plan);
        self.clock_ms += self.last_step_ms;
        self.steps += 1;
        Ok(out)
    }

    fn now_ms(&self) -> f64 {
        self.clock_ms
    }
}
