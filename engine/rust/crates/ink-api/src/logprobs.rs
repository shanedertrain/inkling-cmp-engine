// Ported from engine/server/src/api/openai.{h,cpp} @ 4171493, which ports vLLM's v1/engine/logprobs.py and the
// logprobs wire forms of entrypoints/openai/ (Apache-2.0, Copyright contributors to the vLLM
// project; config B overlay Hiutaky/vllm @ 6e9d1cce).
//! Logprob rows and their two wire forms (`vllm/v1/engine/logprobs.py` `_decode_rows` and
//! `_correct_decoded_token`; chat `ChatCompletionLogProbs`, completions `CompletionLogProbs`).

use ink_backend::TokenLogprobs;
use ink_text::Tokenizer;
use serde_json::{Map, Value, json};

use crate::error::RunError;

/// One candidate of a position: its id, logprob and decoded text.
#[derive(Debug)]
pub(crate) struct Entry {
    pub logprob: f64,
    pub decoded: String,
    id: u32,
}

/// One generated position: the sampled token first, then the top-k candidates (a candidate that is the sampled
/// token keeps its first position and takes the later value).
#[derive(Debug)]
pub(crate) struct Row {
    entries: Vec<Entry>,
}

impl Row {
    /// The sampled token's entry.
    fn sampled(&self) -> &Entry {
        &self.entries[0]
    }
}

/// Decodes logprob rows, repairing tokens that decode to a partial UTF-8 sequence from the preceding ones.
pub(crate) struct LogprobDecoder<'t> {
    tok: &'t Tokenizer,
    history: Vec<u32>,
}

fn ends_with_replacement(s: &str) -> bool {
    s.ends_with('\u{FFFD}')
}

fn token_id(id: i32) -> Result<u32, RunError> {
    u32::try_from(id).map_err(|_| RunError(format!("the back end reported the negative token id {id}")))
}

/// The ids of one step as the text side takes them.
pub(crate) fn token_ids(ids: &[i32]) -> Result<Vec<u32>, RunError> {
    ids.iter().map(|&id| token_id(id)).collect()
}

/// What a step without logprobs for a token stands in with.
pub(crate) fn no_logprobs() -> TokenLogprobs {
    TokenLogprobs {
        logprob: 0.0,
        rank: 1,
        top_ids: Vec::new(),
        top_logprobs: Vec::new(),
    }
}

impl<'t> LogprobDecoder<'t> {
    pub(crate) fn new(tok: &'t Tokenizer) -> Self {
        Self {
            tok,
            history: Vec::new(),
        }
    }

    fn decode(&self, ids: &[u32]) -> Result<String, RunError> {
        Ok(self.tok.decode(ids, false)?)
    }

    /// `_correct_decoded_token`: the text `id` adds to up to four preceding tokens, when it decodes to U+FFFD alone.
    fn correct(&self, id: u32, context: &[u32]) -> Result<String, RunError> {
        const MAX_CONTEXT: usize = 4;
        for n in 1..=context.len().min(MAX_CONTEXT) {
            let ctx = &context[context.len() - n..];
            let mut with_id = ctx.to_vec();
            with_id.push(id);
            let full = self.decode(&with_id)?;
            if ends_with_replacement(&full) {
                continue;
            }
            let mut clean_end = ctx.len();
            for (j, &c) in ctx.iter().enumerate().rev() {
                if ends_with_replacement(&self.decode(&[c])?) {
                    clean_end = j;
                } else {
                    break;
                }
            }
            let clean = if clean_end > 0 {
                self.decode(&ctx[..clean_end])?
            } else {
                String::new()
            };
            if let Some(rest) = full.strip_prefix(clean.as_str()) {
                return Ok(rest.to_owned());
            }
            // the longest common prefix, cut back to a character boundary of `full`
            let mut i = clean.bytes().zip(full.bytes()).take_while(|(a, b)| a == b).count();
            while i > 0 && !full.is_char_boundary(i) {
                i -= 1;
            }
            return Ok(full.get(i..).unwrap_or_default().to_owned());
        }
        Ok(String::new())
    }

    /// Rows for the sampled tokens of one step; `width` is the number of candidates per row (`top_k + 1`).
    ///
    /// # Errors
    /// A negative token id or a decoder failure.
    pub(crate) fn rows(
        &mut self,
        sampled: &[u32],
        logprobs: &[TokenLogprobs],
        width: usize,
    ) -> Result<Vec<Row>, RunError> {
        let mut out = Vec::with_capacity(sampled.len());
        for (&sampled_id, lp) in sampled.iter().zip(logprobs) {
            let mut ids = vec![sampled_id];
            let mut vals = vec![f64::from(lp.logprob)];
            for (k, (&id, &val)) in lp.top_ids.iter().zip(&lp.top_logprobs).enumerate() {
                if k + 1 >= width {
                    break;
                }
                ids.push(token_id(id)?);
                vals.push(f64::from(val));
            }
            let mut decoded = ids
                .iter()
                .map(|&id| self.decode(&[id]))
                .collect::<Result<Vec<_>, _>>()?;
            if decoded.iter().any(|d| ends_with_replacement(d)) {
                let from = self.history.len().saturating_sub(4);
                let context = &self.history[from..];
                for (d, &id) in decoded.iter_mut().zip(&ids) {
                    if ends_with_replacement(d) {
                        *d = self.correct(id, context)?;
                    }
                }
            }
            self.history.push(sampled_id);
            let mut entries: Vec<Entry> = Vec::with_capacity(ids.len());
            for ((id, logprob), decoded) in ids.into_iter().zip(vals).zip(decoded) {
                match entries.iter_mut().find(|e| e.id == id) {
                    Some(e) => {
                        e.logprob = logprob;
                        e.decoded = decoded;
                    }
                    None => entries.push(Entry { logprob, decoded, id }),
                }
            }
            out.push(Row { entries });
        }
        Ok(out)
    }
}

fn clamp(logprob: f64) -> Value {
    Value::from(if logprob < -9999.0 { -9999.0 } else { logprob })
}

fn bytes_of(s: &str) -> Value {
    Value::Array(s.bytes().map(Value::from).collect())
}

/// `ChatCompletionLogProbs` of consecutive positions: each token's logprob and its first `top_n` candidates.
pub(crate) fn chat_logprobs(rows: &[Row], top_n: u32) -> Value {
    let limit = usize::try_from(top_n).unwrap_or(usize::MAX);
    let content: Vec<Value> = rows
        .iter()
        .map(|row| {
            let tok = row.sampled();
            let tops: Vec<Value> = row
                .entries
                .iter()
                .take(limit)
                .map(|e| json!({"token": e.decoded, "logprob": clamp(e.logprob), "bytes": bytes_of(&e.decoded)}))
                .collect();
            json!({"token": tok.decoded, "logprob": clamp(tok.logprob), "bytes": bytes_of(&tok.decoded),
                   "top_logprobs": tops})
        })
        .collect();
    json!({ "content": content })
}

/// `CompletionLogProbs` of consecutive positions; `offset` is the character offset of the first token in the
/// text so far and is advanced past these.
pub(crate) fn completion_logprobs(rows: &[Row], offset: &mut usize) -> Value {
    let (mut offsets, mut token_logprobs, mut tokens, mut tops) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for row in rows {
        let tok = row.sampled();
        offsets.push(Value::from(*offset));
        token_logprobs.push(clamp(tok.logprob));
        tokens.push(Value::from(tok.decoded.as_str()));
        let mut top = Map::new();
        for e in &row.entries {
            top.insert(e.decoded.clone(), clamp(e.logprob));
        }
        tops.push(Value::Object(top));
        *offset += tok.decoded.chars().count();
    }
    json!({"text_offset": offsets, "token_logprobs": token_logprobs, "tokens": tokens, "top_logprobs": tops})
}
