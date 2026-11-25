// Ported from engine/server/src/parse/inkling_parser.cpp (`ParserTables`, `load_all_special_tokens`) @ 6100ab8
// (Apache-2.0, Copyright contributors to the vLLM project).
//! Special-token tables of the Inkling grammar, built once from the tokenizer.

use std::collections::HashSet;
use std::error::Error;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ink_text::Tokenizer;
use serde_json::Value;

use crate::lexer::TERMINALS;

/// The token ids of the Inkling grammar and the special tokens outside it (dropped when seen by id).
#[derive(Debug)]
pub struct ParserTables {
    tokenizer: Arc<Tokenizer>,
    pub(crate) thinking: Option<u32>,
    pub(crate) text: Option<u32>,
    pub(crate) model: Option<u32>,
    user: Option<u32>,
    system: Option<u32>,
    tool: Option<u32>,
    end_sampling: Option<u32>,
    end_message: Option<u32>,
    /// Ids of the grammar's terminals that exist in the vocabulary.
    pub(crate) special_ids: Vec<u32>,
    /// Ids of special tokens outside the grammar.
    pub(crate) drop_ids: HashSet<u32>,
    /// The same tokens as text, in `all_special_tokens` order.
    pub(crate) drop_literals: Vec<String>,
}

impl ParserTables {
    /// Builds the tables. `all_special_tokens` is the tokenizer's `all_special_tokens` (see
    /// [`special_tokens_from_dir`]).
    #[must_use]
    pub fn new(tokenizer: Arc<Tokenizer>, all_special_tokens: &[String]) -> Arc<Self> {
        let id = |s: &str| tokenizer.token_id(s);
        let special_ids = TERMINALS.iter().filter_map(|(lit, _)| id(lit)).collect();
        let mut drop_ids = HashSet::new();
        let mut drop_literals = Vec::new();
        for s in all_special_tokens {
            if TERMINALS.iter().any(|(lit, _)| lit == s) {
                continue;
            }
            if let Some(i) = id(s) {
                drop_ids.insert(i);
                drop_literals.push(s.clone());
            }
        }
        Arc::new(Self {
            thinking: id("<|content_thinking|>"),
            text: id("<|content_text|>"),
            model: id("<|message_model|>"),
            user: id("<|message_user|>"),
            system: id("<|message_system|>"),
            tool: id("<|message_tool|>"),
            end_sampling: id("<|content_model_end_sampling|>"),
            end_message: id("<|end_message|>"),
            special_ids,
            drop_ids,
            drop_literals,
            tokenizer,
        })
    }

    /// Text of one token, as the tokenizer's lossy decode returns it (unknown ids give the empty string).
    pub(crate) fn decode_token(&self, id: u32) -> String {
        self.tokenizer.decode(&[id], false).unwrap_or_default()
    }

    /// `InklingParser._inkling2_is_reasoning_end`: whether the last thing the prompt closed was reasoning.
    pub(crate) fn is_reasoning_end(&self, ids: &[u32]) -> bool {
        let mut latest = None;
        for &id in ids.iter().rev() {
            let id = Some(id);
            if id == self.model || id == self.user || id == self.system || id == self.tool {
                return latest == self.text && latest.is_some() && id == self.model;
            }
            if latest.is_none() && (id == self.text || id == self.thinking || id == self.end_sampling) {
                latest = id;
            }
        }
        latest.is_some() && latest == self.text
    }

    /// `InklingParser.count_reasoning_tokens`: ids between a thinking start and the next end of message or sampling.
    pub(crate) fn count_reasoning_tokens(&self, ids: &[u32]) -> usize {
        let mut in_reasoning = false;
        let mut count = 0;
        for &id in ids {
            let id = Some(id);
            if id == self.thinking {
                in_reasoning = true;
            } else if id == self.end_message || id == self.end_sampling {
                in_reasoning = false;
            } else if in_reasoning {
                count += 1;
            }
        }
        count
    }
}

/// `tokenizer.all_special_tokens` of an HF model directory: the bos, eos, unk, sep, pad, cls and mask tokens and
/// `additional_special_tokens` of `special_tokens_map.json`, in that order, without repeats.
///
/// # Errors
/// The file cannot be read or is not JSON.
pub fn special_tokens_from_dir(model_dir: impl AsRef<Path>) -> Result<Vec<String>, SpecialTokensError> {
    let path = model_dir.as_ref().join("special_tokens_map.json");
    let fail = |source: Box<dyn Error + Send + Sync>| SpecialTokensError {
        path: path.clone(),
        source,
    };
    let text = std::fs::read_to_string(&path).map_err(|e| fail(e.into()))?;
    let map: Value = serde_json::from_str(&text).map_err(|e| fail(e.into()))?;
    let mut out: Vec<String> = Vec::new();
    let mut add = |v: &Value| {
        let token = match v {
            Value::String(s) => Some(s.as_str()),
            Value::Object(o) => o.get("content").and_then(Value::as_str),
            _ => None,
        };
        if let Some(s) = token
            && !out.iter().any(|seen| seen == s)
        {
            out.push(s.to_owned());
        }
    };
    for key in [
        "bos_token",
        "eos_token",
        "unk_token",
        "sep_token",
        "pad_token",
        "cls_token",
        "mask_token",
    ] {
        if let Some(v) = map.get(key) {
            add(v);
        }
    }
    if let Some(Value::Array(items)) = map.get("additional_special_tokens") {
        items.iter().for_each(&mut add);
    }
    Ok(out)
}

/// `special_tokens_map.json` could not be read.
#[derive(Debug)]
pub struct SpecialTokensError {
    path: PathBuf,
    source: Box<dyn Error + Send + Sync>,
}

impl fmt::Display for SpecialTokensError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "cannot read special tokens from {}: {}",
            self.path.display(),
            self.source
        )
    }
}

impl Error for SpecialTokensError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(self.source.as_ref())
    }
}
