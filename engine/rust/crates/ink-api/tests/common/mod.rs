//! Shared test support: a small byte-level tokenizer with the Inkling special tokens, scripted replies and a
//! front end over a [`MockBackend`] with deterministic hooks.
#![allow(dead_code, reason = "each test binary uses a part of this module")]

use std::sync::Arc;

use ink_api::{Frontend, Hooks, ServerConfig};
use ink_backend::{MockBackend, Script};
use ink_text::{Specials, Tokenizer};
use serde_json::{Value, json};

pub const SPECIALS: [&str; 15] = [
    "<|endoftext|>",
    "<|begin_of_text|>",
    "<|message_user|>",
    "<|message_model|>",
    "<|message_system|>",
    "<|message_tool|>",
    "<|content_text|>",
    "<|content_thinking|>",
    "<|content_image|>",
    "<|content_xml|>",
    "<|content_tool_error|>",
    "<|content_invoke_tool_json|>",
    "<|content_invoke_tool_text|>",
    "<|end_message|>",
    "<|content_model_end_sampling|>",
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

pub fn special(tok: &Tokenizer, name: &str) -> i32 {
    i32::try_from(tok.token_id(name).expect("a special token of the mini tokenizer")).expect("small id")
}

pub fn encode(tok: &Tokenizer, text: &str) -> Vec<i32> {
    tok.encode(text, Specials::Recognize)
        .expect("encodes")
        .into_iter()
        .map(|id| i32::try_from(id).expect("small id"))
        .collect()
}

/// A reasoning message and a text message, then end of sampling, in groups of `group` tokens.
pub fn reply(tok: &Tokenizer, reasoning: &str, content: &str, group: usize) -> Script {
    let mut all = vec![special(tok, "<|message_model|>"), special(tok, "<|content_thinking|>")];
    all.extend(encode(tok, reasoning));
    all.push(special(tok, "<|end_message|>"));
    all.push(special(tok, "<|message_model|>"));
    all.push(special(tok, "<|content_text|>"));
    all.extend(encode(tok, content));
    all.push(special(tok, "<|end_message|>"));
    all.push(special(tok, "<|content_model_end_sampling|>"));
    Script {
        groups: all.chunks(group).map(<[i32]>::to_vec).collect(),
        num_cached_tokens: Some(4),
        ..Script::default()
    }
}

/// Fixed ids and clock: `chatcmpl-0000000000000000`, tool calls `chatcmpl-tool-000000000000000N`.
pub fn fixed_hooks() -> Hooks {
    let counter = Arc::new(std::sync::atomic::AtomicU32::new(0));
    Hooks {
        request_hex: Arc::new(|| "0000000000000000".to_owned()),
        tool_call_id: Arc::new(move || {
            format!(
                "chatcmpl-tool-{:016x}",
                counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1
            )
        }),
        now: Arc::new(|| 0),
    }
}

pub struct Fixture {
    pub tok: Arc<Tokenizer>,
    pub backend: Arc<MockBackend>,
    pub fe: Frontend,
}

pub fn fixture() -> Fixture {
    let tok = mini_tokenizer();
    let backend = Arc::new(MockBackend::new());
    let cfg = ServerConfig {
        eos_token_id: u32::try_from(special(&tok, "<|content_model_end_sampling|>")).expect("small id"),
        ..ServerConfig::default()
    };
    let fe = Frontend::new(Arc::clone(&tok), cfg, backend.clone(), fixed_hooks())
        .expect("front end over the mini tokenizer");
    Fixture { tok, backend, fe }
}

pub mod client;
