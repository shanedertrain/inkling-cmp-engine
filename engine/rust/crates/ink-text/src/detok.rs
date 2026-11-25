// Ported from engine/server/src/text/detok.{h,cpp} @ 4171493, which ports vllm/v1/engine/detokenizer.py
// (BaseIncrementalDetokenizer.update, get_next_output_text, check_stop_strings) and vllm/v1/core/sched/utils.py
// check_stop. Apache-2.0, Copyright contributors to the vLLM project. The byte-level step algorithm is the
// `tokenizers` crate's `DecodeStream` (Apache-2.0, Copyright Hugging Face).
//! Streaming detokenisation and stop handling, identical to config B's output processor.
//!
//! [`DecodeStream`] turns token ids into text as they arrive, holding back an incomplete UTF-8 sequence.
//! [`OutputState`] applies the engine core's stop checks (EOS, stop ids, `max_tokens`) and the detokeniser's stop
//! strings to every step of one request and hands out the text that is safe to send.

use std::error::Error;
use std::fmt;

use tokenizers::{
    DecoderWrapper, ModelWrapper, NormalizerWrapper, PostProcessorWrapper, PreTokenizerWrapper,
    tokenizer::DecodeStreamError,
};

use crate::Tokenizer;

type HfStream<'t> = tokenizers::DecodeStream<
    't,
    ModelWrapper,
    NormalizerWrapper,
    PreTokenizerWrapper,
    PostProcessorWrapper,
    DecoderWrapper,
>;

/// The tokenizer's decoder failed (not observed on valid ids).
#[derive(Debug)]
pub struct DetokError(Box<dyn Error + Send + Sync>);

impl fmt::Display for DetokError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "detokenizer: {}", self.0)
    }
}

impl Error for DetokError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(&*self.0)
    }
}

/// Incremental decoder: the text of each new token, with an incomplete UTF-8 sequence held back until it completes.
///
/// The step algorithm is `tokenizers`' own. Added here, as in vLLM's `FastIncrementalDetokenizer`: when the stream
/// reports an invalid prefix it is recreated and the token is stepped again.
pub struct DecodeStream<'t> {
    tok: &'t Tokenizer,
    skip_special: bool,
    inner: HfStream<'t>,
}

impl fmt::Debug for DecodeStream<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DecodeStream")
            .field("skip_special", &self.skip_special)
            .finish_non_exhaustive()
    }
}

impl<'t> DecodeStream<'t> {
    /// A stream at the start of a sequence; with `skip_special` added special tokens produce no text.
    #[must_use]
    pub fn new(tok: &'t Tokenizer, skip_special: bool) -> Self {
        Self {
            tok,
            skip_special,
            inner: tok.inner().decode_stream(skip_special),
        }
    }

    /// The text `id` completes: empty while the bytes seen so far end inside a character.
    ///
    /// # Errors
    /// The decoder fails.
    pub fn step(&mut self, id: u32) -> Result<String, DetokError> {
        match self.inner.step(id) {
            Ok(text) => Ok(text.unwrap_or_default()),
            Err(e)
                if matches!(
                    e.downcast_ref::<DecodeStreamError>(),
                    Some(DecodeStreamError::InvalidPrefix { .. })
                ) =>
            {
                self.inner = self.tok.inner().decode_stream(self.skip_special);
                Ok(self.inner.step(id).map_err(DetokError)?.unwrap_or_default())
            }
            Err(e) => Err(DetokError(e)),
        }
    }
}

/// Why a request finished.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinishReason {
    /// EOS, a stop token id or a stop string.
    Stop,
    /// `max_tokens` reached.
    Length,
}

impl FinishReason {
    /// The OpenAI spelling, `"stop"` or `"length"`.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Stop => "stop",
            Self::Length => "length",
        }
    }
}

/// What ended a request with [`FinishReason::Stop`], when it is not EOS.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StopReason {
    /// A token id of `stop_token_ids`.
    TokenId(u32),
    /// A stop string.
    Text(String),
}

/// Stop and length parameters of one request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StopParams {
    /// Stop strings.
    pub stop: Vec<String>,
    /// Extra stop token ids (EOS is separate).
    pub stop_token_ids: Vec<u32>,
    /// The end-of-sequence id, when the request stops on it.
    pub eos_token_id: Option<u32>,
    /// Do not stop on EOS.
    pub ignore_eos: bool,
    /// Keep the stop string in the output text.
    pub include_stop_str_in_output: bool,
    /// Stop checks apply only from this many output tokens.
    pub min_tokens: usize,
    /// Output token cap.
    pub max_tokens: usize,
    /// Added special tokens produce no text.
    pub skip_special_tokens: bool,
}

impl Default for StopParams {
    fn default() -> Self {
        Self {
            stop: Vec::new(),
            stop_token_ids: Vec::new(),
            eos_token_id: None,
            ignore_eos: false,
            include_stop_str_in_output: false,
            min_tokens: 0,
            max_tokens: 16,
            skip_special_tokens: true,
        }
    }
}

/// The outcome of one [`OutputState::append`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StepResult {
    /// Tokens accepted this step, after truncation at a stop.
    pub new_ids: Vec<u32>,
    /// Set when the request finished with this step.
    pub finish: Option<FinishReason>,
    /// The stop token id or stop string, when one ended the request (not for EOS or length).
    pub stop_reason: Option<StopReason>,
}

/// Per-request output state: the engine core's stop check applied token by token (so speculative overshoot is
/// dropped), the detokeniser and the stop strings, as vLLM's `EngineCore` and `OutputProcessor` do.
#[derive(Debug)]
pub struct OutputState<'t> {
    params: StopParams,
    stream: DecodeStream<'t>,
    ids: Vec<u32>,
    text: String,
    /// `max(len(stop)) - 1` characters held back while running.
    stop_buffer_chars: usize,
    /// Byte offset of the text already handed out.
    last_offset: usize,
    finished: bool,
}

impl<'t> OutputState<'t> {
    /// A fresh request state.
    #[must_use]
    pub fn new(tok: &'t Tokenizer, params: StopParams) -> Self {
        let stop_buffer_chars = if params.include_stop_str_in_output {
            0
        } else {
            params
                .stop
                .iter()
                .map(|s| s.chars().count())
                .max()
                .map_or(0, |n| n.saturating_sub(1))
        };
        Self {
            stream: DecodeStream::new(tok, params.skip_special_tokens),
            params,
            ids: Vec::new(),
            text: String::new(),
            stop_buffer_chars,
            last_offset: 0,
            finished: false,
        }
    }

    /// Output ids so far, truncated at a stop.
    #[must_use]
    pub fn output_ids(&self) -> &[u32] {
        &self.ids
    }

    /// Whether the request has finished.
    #[must_use]
    pub fn finished(&self) -> bool {
        self.finished
    }

    /// Appends the tokens one model step produced (several for speculative decoding).
    ///
    /// # Errors
    /// The decoder fails.
    pub fn append(&mut self, step_ids: &[u32]) -> Result<StepResult, DetokError> {
        let mut result = StepResult::default();
        if self.finished {
            return Ok(result);
        }
        // 1) engine core: token by token, check_stop after each (EOS, stop ids, then the length cap)
        let mut accepted = 0;
        let mut stop_by_token = false;
        for &id in step_ids {
            accepted += 1;
            let n_out = self.ids.len() + accepted;
            if n_out >= self.params.min_tokens {
                if !self.params.ignore_eos && Some(id) == self.params.eos_token_id {
                    result.finish = Some(FinishReason::Stop);
                    stop_by_token = true;
                    break;
                }
                if self.params.stop_token_ids.contains(&id) {
                    result.finish = Some(FinishReason::Stop);
                    result.stop_reason = Some(StopReason::TokenId(id));
                    stop_by_token = true;
                    break;
                }
            }
            if n_out >= self.params.max_tokens {
                result.finish = Some(FinishReason::Length);
                break;
            }
        }
        let accepted = &step_ids[..accepted];
        if accepted.is_empty() {
            return Ok(result);
        }
        // 2) detokenizer update(new_token_ids, stop_terminated)
        let num_prev = self.ids.len();
        let first_new = self.ids.len();
        // a stop token is not decoded (unless its text is kept) but still counts as an output token
        let (to_decode, skipped) = match accepted.split_last() {
            Some((&last, rest)) if stop_by_token && !self.params.include_stop_str_in_output => (rest, Some(last)),
            _ => (accepted, None),
        };
        let track = !self.params.stop.is_empty();
        let mut offsets: Vec<usize> = Vec::new();
        if track {
            offsets.push(self.text.len());
        }
        for &id in to_decode {
            self.ids.push(id);
            let piece = self.stream.step(id)?;
            self.text.push_str(&piece);
            if track {
                offsets.push(self.text.len());
            }
        }
        if let Some(id) = skipped {
            self.ids.push(id);
            if track {
                offsets.push(self.text.len());
            }
        }
        if track && self.ids.len() > self.params.min_tokens {
            let idx = self.params.min_tokens.saturating_sub(num_prev);
            let check_from = offsets[idx.min(offsets.len() - 1)];
            if let Some((stop, truncate_to)) = self.check_stop_strings(self.text.len() - check_from) {
                let mut stop_end = truncate_to.unwrap_or(self.text.len());
                if !self.params.include_stop_str_in_output {
                    stop_end += stop.len();
                }
                // bisect_left(offsets, stop_end, lo=1)
                let num_kept = 1 + offsets[1..].partition_point(|&o| o < stop_end);
                self.ids.truncate(self.ids.len().min(first_new + num_kept));
                if let Some(cut) = truncate_to {
                    self.text.truncate(cut);
                }
                result.finish = Some(FinishReason::Stop);
                result.stop_reason = Some(StopReason::Text(stop));
            }
        }
        result.new_ids = self.ids[first_new..].to_vec();
        self.finished = result.finish.is_some();
        Ok(result)
    }

    /// The earliest-ending stop string found in the text added by the last `new_bytes` bytes, with the byte
    /// offset to truncate the text at (`None`: keep all of it, for `include_stop_str_in_output` at the end).
    fn check_stop_strings(&self, new_bytes: usize) -> Option<(String, Option<usize>)> {
        if new_bytes == 0 {
            return None;
        }
        let mut best: Option<(&str, usize, usize)> = None; // (stop, index, end)
        for stop in &self.params.stop {
            let start = (self.text.len() + 1).saturating_sub(new_bytes + stop.len());
            let start = self.text.ceil_char_boundary(start);
            let Some(found) = self.text[start..].find(stop.as_str()) else {
                continue;
            };
            let index = start + found;
            let end = index + stop.len();
            if best.is_none_or(|(_, _, best_end)| end < best_end) {
                best = Some((stop, index, end));
            }
        }
        let (stop, index, end) = best?;
        let truncate_to = if !self.params.include_stop_str_in_output {
            Some(index)
        } else if end >= self.text.len() {
            None
        } else {
            Some(end)
        };
        Some((stop.to_owned(), truncate_to))
    }

    /// The text since the previous call, holding back `max(len(stop)) - 1` characters while the request runs.
    pub fn next_text(&mut self, finished: bool) -> String {
        let length = self.safe_len(finished);
        if self.last_offset < length {
            let out = self.text[self.last_offset..length].to_owned();
            self.last_offset = length;
            out
        } else {
            String::new()
        }
    }

    /// All text that is safe to send (`get_next_output_text(finished, delta=False)`).
    #[must_use]
    pub fn full_text(&self, finished: bool) -> &str {
        &self.text[..self.safe_len(finished)]
    }

    /// Byte length of the text that may be sent: everything, or all but the held-back characters.
    fn safe_len(&self, finished: bool) -> usize {
        let held = if finished { 0 } else { self.stop_buffer_chars };
        if held == 0 {
            return self.text.len();
        }
        self.text.char_indices().rev().nth(held - 1).map_or(0, |(i, _)| i)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The decode-stream and stop-handling tests that need a tokenizer are in tests/detok.rs.

    #[test]
    fn finish_reason_spelling() {
        assert_eq!(FinishReason::Stop.as_str(), "stop");
        assert_eq!(FinishReason::Length.as_str(), "length");
    }
}
