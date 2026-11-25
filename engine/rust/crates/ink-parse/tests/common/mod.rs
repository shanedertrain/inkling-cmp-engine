//! Shared test support: a small byte-level tokenizer with the Inkling special tokens, and the per-step text a serving
//! layer would hand the parser (through `ink-text`'s `OutputState`).
#![allow(dead_code, reason = "each test binary uses a part of this module")]

use std::sync::Arc;

use ink_parse::{IdFactory, ParserTables};
use ink_text::{OutputState, Specials, StopParams, Tokenizer};
use serde_json::{Value, json};

pub const SPECIALS: [&str; 14] = [
    "<|message_model|>",
    "<|message_user|>",
    "<|message_system|>",
    "<|message_tool|>",
    "<|content_text|>",
    "<|content_thinking|>",
    "<|end_message|>",
    "<|content_model_end_sampling|>",
    "<|content_invoke_tool_json|>",
    "<|content_invoke_tool_text|>",
    "<|content_tool_error|>",
    "<|content_image|>",
    "<|eos|>",
    "<|pad|>",
];

/// GPT-2's byte-to-unicode table.
fn byte_chars() -> Vec<char> {
    let printable: Vec<u32> = (33..=126).chain(161..=172).chain(174..=255).collect();
    let mut extra = 0;
    (0..=255u32)
        .map(|b| {
            if printable.contains(&b) {
                char::from_u32(b).expect("latin-1 scalar")
            } else {
                extra += 1;
                char::from_u32(255 + extra).expect("scalar")
            }
        })
        .collect()
}

/// A byte-level BPE without merges (one token per byte) plus the special tokens as added tokens, ids 256 and up.
pub fn mini_tokenizer() -> Arc<Tokenizer> {
    let vocab: serde_json::Map<String, Value> = byte_chars()
        .into_iter()
        .zip(0u32..)
        .map(|(c, i)| (c.to_string(), json!(i)))
        .collect();
    let added: Vec<Value> = SPECIALS
        .iter()
        .zip(256u32..)
        .map(|(content, id)| {
            json!({"id": id, "content": content, "single_word": false, "lstrip": false, "rstrip": false,
                   "normalized": false, "special": true})
        })
        .collect();
    let def = json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": added, "normalizer": null,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false},
        "post_processor": null,
        "decoder": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true},
        "model": {"type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": null,
                  "end_of_word_suffix": null, "fuse_unk": false, "byte_fallback": false, "ignore_merges": false,
                  "vocab": vocab, "merges": []}
    });
    Arc::new(Tokenizer::from_json(def.to_string().as_bytes()).expect("mini tokenizer definition"))
}

pub fn mini_tables() -> (Arc<Tokenizer>, Arc<ParserTables>) {
    let tok = mini_tokenizer();
    let specials: Vec<String> = SPECIALS.iter().map(|s| (*s).to_owned()).collect();
    let tables = ParserTables::new(Arc::clone(&tok), &specials);
    (tok, tables)
}

pub fn fixed_ids() -> IdFactory {
    Arc::new(|| "chatcmpl-tool-0000000000000000".to_owned())
}

/// One model step as the parser sees it.
#[derive(Debug)]
pub struct Step {
    pub text: String,
    pub ids: Vec<u32>,
    pub finished: bool,
}

/// The steps config B's engine core and detokeniser make of token groups: `OutputState` with the EOS id and the
/// length cap, special tokens kept in the text, no stop strings. Stops at the step that finishes.
pub fn steps_of(tok: &Tokenizer, eos: u32, max_tokens: usize, groups: &[Vec<u32>]) -> Vec<Step> {
    let mut state = OutputState::new(
        tok,
        StopParams {
            eos_token_id: Some(eos),
            max_tokens,
            skip_special_tokens: false,
            ..StopParams::default()
        },
    );
    let mut steps = Vec::new();
    for group in groups {
        let result = state.append(group).expect("the decode stream accepts the ids");
        let finished = result.finish.is_some();
        steps.push(Step {
            text: state.next_text(finished),
            ids: result.new_ids,
            finished,
        });
        if finished {
            break;
        }
    }
    steps
}

pub fn encode(tok: &Tokenizer, text: &str) -> Vec<u32> {
    tok.encode(text, Specials::Recognize).expect("encode")
}
