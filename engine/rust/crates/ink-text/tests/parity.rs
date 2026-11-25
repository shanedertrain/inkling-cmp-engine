//! Token-id and text equality of the Rust tokenizer against config B's recorded outputs (package F1).
//!
//! Needs the model's tokenizer files (`INK_MODEL_DIR`) and the recordings extracted by `tests/prepare_fixtures.py`
//! (`INK_PARITY_DIR`); see `README.md` of the crate. Without them a test prints `SKIPPED` and passes.

use std::fs;
use std::path::PathBuf;

use ink_text::{Specials, Tokenizer};
use serde_json::Value;

fn tokenizer() -> Option<Tokenizer> {
    let Some(dir) = std::env::var_os("INK_MODEL_DIR") else {
        eprintln!("SKIPPED: INK_MODEL_DIR is not set (directory holding the model's tokenizer.json)");
        return None;
    };
    Some(Tokenizer::load(PathBuf::from(dir)).expect("tokenizer.json in INK_MODEL_DIR loads"))
}

fn rows(name: &str) -> Option<Vec<Value>> {
    let Some(dir) = std::env::var_os("INK_PARITY_DIR") else {
        eprintln!("SKIPPED: INK_PARITY_DIR is not set (output of tests/prepare_fixtures.py)");
        return None;
    };
    let text = fs::read_to_string(PathBuf::from(dir).join(name)).expect("fixture file is readable");
    Some(
        text.lines()
            .map(|l| serde_json::from_str(l).expect("fixture line is JSON"))
            .collect(),
    )
}

fn ids(v: &Value) -> Vec<u32> {
    v.as_array()
        .expect("ids are an array")
        .iter()
        .map(|n| u32::try_from(n.as_u64().expect("id is an unsigned integer")).expect("id fits u32"))
        .collect()
}

/// Prints the equality line the report quotes and fails when anything differs.
fn report(name: &str, differing: &[String], total: usize) {
    println!("PARITY {name}: {} identical / {total} total", total - differing.len());
    assert!(differing.is_empty(), "{name}: differing cases: {differing:?}");
}

#[test]
fn tokenizer_roundtrip() {
    // Port of test_text.cpp `text_tokenizer_roundtrip`.
    let Some(tok) = tokenizer() else { return };
    let s = "naïve café 漢字 😀 <|end_message|> code: if (a < b) { return x; }\n\ttabs";
    let literal = tok.encode(s, Specials::Literal).expect("encodes");
    assert_eq!(tok.decode(&literal, false).expect("decodes"), s);
    let recognised = tok.encode(s, Specials::Recognize).expect("encodes");
    assert!(
        recognised.len() < literal.len(),
        "the marker is one token when recognised"
    );
    assert_eq!(tok.token_id("<|content_model_end_sampling|>"), Some(200_006));
    assert!(tok.is_special(200_010) && !tok.is_special(13));
    assert_eq!(tok.token_id("<|end_message|>"), Some(200_010));
}

#[test]
fn raw_texts_encode_to_config_b_ids() {
    // Recorded ids are config B's /tokenize of the text cut to `raw_tokens`; `full_count` is the uncut length.
    let (Some(tok), Some(cases)) = (tokenizer(), rows("raw_text.jsonl")) else {
        return;
    };
    let mut bad = Vec::new();
    for c in &cases {
        let got = tok
            .encode(c["text"].as_str().expect("text"), Specials::Recognize)
            .expect("encodes");
        let want = ids(&c["ids"]);
        let full = c["full_count"].as_u64().map(|n| usize::try_from(n).expect("fits"));
        if got.get(..want.len()) != Some(&want[..]) || full.is_some_and(|n| n != got.len()) {
            bad.push(c["src"].to_string());
        }
    }
    report("raw text -> ids (prefix and full count)", &bad, cases.len());
}

#[test]
fn completion_prompts_encode_and_outputs_decode() {
    let (Some(tok), Some(cases)) = (tokenizer(), rows("completions.jsonl")) else {
        return;
    };
    let (mut bad_enc, mut bad_dec, mut n_enc) = (Vec::new(), Vec::new(), 0);
    for c in &cases {
        if let (Some(text), false) = (c["text"].as_str(), c["prompt_ids"].is_null()) {
            n_enc += 1;
            if tok.encode(text, Specials::Recognize).expect("encodes") != ids(&c["prompt_ids"]) {
                bad_enc.push(c["src"].to_string());
            }
        }
        let skip = c["skip_special"].as_bool().expect("flag");
        let got = tok.decode(&ids(&c["out_ids"]), skip).expect("decodes");
        let want = c["out_text"].as_str().expect("text");
        // a stop string ends the text before the tokens that spelled it
        let stopped = c["finish"] == "stop" && got.starts_with(want);
        if got != want && !stopped {
            bad_dec.push(c["src"].to_string());
        }
    }
    report("completions: prompt text -> ids", &bad_enc, n_enc);
    report("completions: output ids -> text", &bad_dec, cases.len());
}

#[test]
fn chat_prompts_roundtrip_through_text() {
    // The rendered prompt text is not recorded; the recorded ids are decoded and encoded again. Equality holds
    // because the renderer ends every ordinary piece at an added token, except where a message contains the spelling
    // of one: the renderer encodes that text as ordinary text, which decoding cannot tell from the real token. Such a
    // message is checked on its own: its ordinary encoding must appear in the recorded ids.
    let (Some(tok), Some(cases)) = (tokenizer(), rows("chat_prompts.jsonl")) else {
        return;
    };
    let (mut bad, mut n_literal) = (Vec::new(), 0);
    for c in &cases {
        let want = ids(&c["ids"]);
        let literal: Vec<&str> = c["literal_texts"]
            .as_array()
            .expect("list")
            .iter()
            .filter_map(Value::as_str)
            .collect();
        if literal.is_empty() {
            let text = tok.decode(&want, false).expect("decodes");
            if tok.encode(&text, Specials::Recognize).expect("encodes") != want {
                bad.push(c["src"].to_string());
            }
            continue;
        }
        n_literal += 1;
        for text in literal {
            let piece = tok.encode(text, Specials::Literal).expect("encodes");
            if !want.windows(piece.len()).any(|w| w == piece) {
                bad.push(c["src"].to_string());
            }
        }
    }
    println!("PARITY chat prompts with literal special-token text, checked by ordinary encoding: {n_literal}");
    report("chat prompt ids -> text -> ids", &bad, cases.len());
}

#[test]
fn chat_outputs_decode_to_config_b_text() {
    // The response text is reasoning + content: a prefix of the decoded ids (special tokens skipped). What follows is
    // one of: nothing, white space the parser strips, a stop string (finish "stop"), or tool-call JSON after the
    // `<|content_invoke_tool_json|>` token (also when a length cut left no complete call).
    const INVOKE_TOOL_JSON: u32 = 200_049;
    let (Some(tok), Some(cases)) = (tokenizer(), rows("chat_outputs.jsonl")) else {
        return;
    };
    let (mut bad, mut exact, mut ws, mut stop, mut tool) = (Vec::new(), 0, 0, 0, 0);
    for c in &cases {
        let out = ids(&c["out_ids"]);
        let want = format!(
            "{}{}",
            c["reasoning"].as_str().unwrap_or_default(),
            c["content"].as_str().unwrap_or_default()
        );
        let got = tok
            .decode(&out, c["skip_special"].as_bool().expect("flag"))
            .expect("decodes");
        let Some(rest) = got.strip_prefix(&want) else {
            bad.push(c["src"].to_string());
            continue;
        };
        if rest.is_empty() {
            exact += 1;
        } else if rest.trim().is_empty() {
            ws += 1;
        } else if c["finish"] == "stop" {
            stop += 1;
        } else if out.contains(&INVOKE_TOOL_JSON) {
            tool += 1;
        } else {
            bad.push(format!("{} (unexplained rest {rest:?})", c["src"]));
        }
    }
    println!("PARITY chat outputs: exact {exact}, trailing white space {ws}, stop string {stop}, tool call {tool}");
    report("chat outputs: ids -> reasoning+content prefix", &bad, cases.len());
}
