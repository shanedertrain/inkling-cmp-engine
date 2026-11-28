// Ported from engine/server/src/sched/scheduler.h @ 4171493 (plan and event types).
//! Step plans, runtime results and the events the scheduler reports to the front end.

use ink_backend::{SamplingParams, TokenLogprobs};

/// What a step segment does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegKind {
    /// One chunk of a prompt.
    Prefill,
    /// One sampled token, plus verify rows when speculating.
    Decode,
}

/// One sequence's share of a step.
#[derive(Debug, Clone, PartialEq)]
pub struct Segment {
    /// Sequence id.
    pub seq: u64,
    /// Engine slot the sequence occupies.
    pub slot: u32,
    /// Prefill chunk or decode.
    pub kind: SegKind,
    /// Position of the first row, which is the number of tokens already cached.
    pub pos0: u32,
    /// Prefill: the chunk. Decode: the last accepted token (drafts belong to the runtime).
    pub tokens: Vec<i32>,
    /// Decode: 1 plus drafted tokens. Prefill: chunk length.
    pub width: u32,
    /// Whether the segment returns sampled tokens: every decode, the last prefill chunk.
    pub samples: bool,
    /// The slot has a new occupant or a stale ring: the runtime resets conv state and ring.
    pub reset_slot: bool,
    /// Page ids covering `[0, pos0 + width)`.
    pub pages: Vec<u32>,
    /// Sampling controls of the sequence.
    pub sampling: SamplingParams,
    /// Where the segment's emitted tokens end.
    pub stop: Stop,
}

/// Where the tokens a segment emits end, checked token by token (vLLM's `check_stop`): after the first stop token
/// once at least `min_tokens` are out, or at `max_tokens`. A runtime commits the rows of the kept tokens only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stop {
    /// Token ids that end the request.
    pub ids: Vec<i32>,
    /// Output tokens before the segment.
    pub generated: u32,
    /// Stop ids count from this many output tokens on.
    pub min_tokens: u32,
    /// Output token limit.
    pub max_tokens: u32,
    /// Token ids whose logits the runtime sets to -inf in rows that yield one of the first `min_tokens` output
    /// tokens (vLLM's min-tokens logits processor), so no stop id is emitted before `min_tokens`.
    pub masked: Vec<i32>,
}

impl Default for Stop {
    /// No stop token and no limit.
    fn default() -> Self {
        Self {
            ids: Vec::new(),
            generated: 0,
            min_tokens: 0,
            max_tokens: u32::MAX,
            masked: Vec::new(),
        }
    }
}

impl Stop {
    /// How many of `tokens`, emitted in order by the segment, the request keeps (at least one when there is one).
    #[must_use]
    pub fn keep(&self, tokens: &[i32]) -> usize {
        for (i, t) in tokens.iter().enumerate() {
            let out = u64::from(self.generated) + i as u64 + 1;
            if out >= u64::from(self.max_tokens) || (out >= u64::from(self.min_tokens) && self.ids.contains(t)) {
                return i + 1;
            }
        }
        tokens.len()
    }

    /// How many of the first `rows` logit rows of the segment the runtime masks: row `j` yields output token number
    /// `generated + j + 1`, which may not be a masked id while `generated + j < min_tokens` (vLLM masks a row whose
    /// position + 1 is below prompt length + `min_tokens`).
    #[must_use]
    pub fn masked_rows(&self, rows: usize) -> usize {
        if self.masked.is_empty() {
            return 0;
        }
        (self.min_tokens.saturating_sub(self.generated) as usize).min(rows)
    }

    /// Whether `token`, as output token number `out` (1-based), ends the request on a stop id.
    #[must_use]
    pub fn stops_at(&self, token: i32, out: usize) -> bool {
        out as u64 >= u64::from(self.min_tokens) && self.ids.contains(&token)
    }
}

/// One step's work: one decode segment per decoding session and packed prefill chunks up to the step budget.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct StepPlan {
    /// Segments in execution order (decode segments first).
    pub segs: Vec<Segment>,
    /// Decode rows in the step.
    pub decode_rows: u32,
    /// Prefill rows in the step.
    pub prefill_rows: u32,
    /// The scheduler's own cost estimate, in milliseconds.
    pub predicted_ms: f64,
    /// Slots whose runtime state no sequence will continue from, when the plan was made: no sequence holds them
    /// (running or parked), or their sequence starts over in them (`reset_slot` pending, possibly in a later step). A
    /// runtime with its own page tables frees their state before the step, so its pages track the scheduler's pool.
    pub stale_slots: Vec<u32>,
}

impl StepPlan {
    /// Whether the plan has no segment (nothing to run).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.segs.is_empty()
    }
}

/// What the runtime returns for one sampling segment.
#[derive(Debug, Clone, PartialEq)]
pub struct SegResult {
    /// Sequence id.
    pub seq: u64,
    /// Accepted tokens: 1 to `width` for decode, 1 for the last prefill chunk.
    pub tokens: Vec<i32>,
    /// One entry per token when logprobs were requested.
    pub logprobs: Vec<TokenLogprobs>,
}

/// Why a sequence ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Finish {
    /// Finished normally: `max_new_tokens` reached, or `finish`/`cancel` from the front end.
    Done,
    /// The per-session cap (prompt + output) was reached.
    SessionCap,
    /// The step failed; the message names the cause.
    Error(String),
}

/// What the back end reports to the front end for one sequence after a step, `finish` or `cancel`.
#[derive(Debug, Clone, PartialEq)]
pub struct SeqEvent {
    /// Sequence id.
    pub seq: u64,
    /// Tokens accepted this step.
    pub tokens: Vec<i32>,
    /// Logprobs of the runtime's result, when requested.
    pub logprobs: Vec<TokenLogprobs>,
    /// Prompt tokens served from resident KV; set on the first event with tokens only.
    pub num_cached_tokens: Option<u32>,
    /// Set on the last event of the sequence.
    pub finish: Option<Finish>,
}

impl SeqEvent {
    pub(crate) fn finished(seq: u64, finish: Finish) -> Self {
        Self {
            seq,
            tokens: Vec::new(),
            logprobs: Vec::new(),
            num_cached_tokens: None,
            finish: Some(finish),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn masked_rows_cover_the_outputs_before_min_tokens() {
        let stop = |generated, min_tokens, masked: &[i32]| Stop {
            ids: vec![9],
            generated,
            min_tokens,
            max_tokens: 100,
            masked: masked.to_vec(),
        };
        // row j yields output generated + j + 1: masked while generated + j < min_tokens
        assert_eq!(stop(0, 3, &[9]).masked_rows(8), 3);
        assert_eq!(stop(2, 3, &[9]).masked_rows(8), 1);
        assert_eq!(stop(3, 3, &[9]).masked_rows(8), 0);
        assert_eq!(stop(0, 20, &[9]).masked_rows(8), 8);
        assert_eq!(stop(0, 20, &[9]).masked_rows(1), 1);
        // nothing to mask without ids, or without min_tokens
        assert_eq!(stop(0, 20, &[]).masked_rows(8), 0);
        assert_eq!(stop(0, 0, &[9]).masked_rows(8), 0);
    }
}
