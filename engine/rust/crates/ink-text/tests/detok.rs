//! Streaming detokenisation and stop handling (package F2): ports of the `engine/server` tests, and the streamed
//! text of config B's recorded outputs against one-shot decoding.
//!
//! Needs the model's tokenizer files (`INK_MODEL_DIR`), and for the recordings the output of
//! `tests/prepare_fixtures.py` (`INK_PARITY_DIR`). Without them a test prints `SKIPPED` and passes.

use std::fs;
use std::path::PathBuf;

use ink_text::{DecodeStream, FinishReason, OutputState, Specials, StopParams, StopReason, Tokenizer};
use serde_json::Value;

/// `<|content_model_end_sampling|>`, used as EOS by the C++ tests.
const EOS: u32 = 200_006;
const END_MESSAGE: u32 = 200_010;

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

fn encode(tok: &Tokenizer, text: &str) -> Vec<u32> {
    tok.encode(text, Specials::Literal).expect("encodes")
}

/// Id of the single-byte vocabulary entry whose byte-level spelling is the Latin-1 character `c`.
fn single_byte_id(tok: &Tokenizer, c: char) -> u32 {
    *tok.inner()
        .get_vocab(false)
        .get(&c.to_string())
        .unwrap_or_else(|| panic!("vocabulary has the byte token {c:?}"))
}

#[test]
fn text_decode_stream_partial_utf8() {
    let Some(tok) = tokenizer() else { return };
    // bytes 0xC3 and 0xA9 are U+00C3 and U+00A9 in the byte-level alphabet; together they are "é"
    let (c3, a9) = (single_byte_id(&tok, '\u{c3}'), single_byte_id(&tok, '\u{a9}'));
    let mut ds = DecodeStream::new(&tok, false);
    assert_eq!(ds.step(encode(&tok, "x")[0]).expect("steps"), "x");
    assert_eq!(ds.step(c3).expect("steps"), "", "held: incomplete sequence");
    assert_eq!(ds.step(a9).expect("steps"), "é");
    assert_eq!(ds.step(END_MESSAGE).expect("steps"), "<|end_message|>");
    let mut skip = DecodeStream::new(&tok, true);
    assert_eq!(skip.step(END_MESSAGE).expect("steps"), "");
}

fn params(max_tokens: usize) -> StopParams {
    StopParams {
        eos_token_id: Some(EOS),
        max_tokens,
        ..StopParams::default()
    }
}

#[test]
fn text_output_state_stops() {
    let Some(tok) = tokenizer() else { return };
    // EOS stops, its text is not emitted, the token is counted
    {
        let mut os = OutputState::new(
            &tok,
            StopParams {
                skip_special_tokens: false,
                ..params(100)
            },
        );
        let mut step = encode(&tok, "Hello");
        step.push(EOS);
        step.push(encode(&tok, "x")[0]); // speculative overshoot after EOS: dropped
        let r = os.append(&step).expect("appends");
        assert_eq!(r.finish, Some(FinishReason::Stop));
        assert_eq!(r.new_ids.len(), step.len() - 1);
        assert_eq!(os.next_text(true), "Hello");
    }
    // EOS exactly at max_tokens: still "stop" (EOS is checked before the length cap)
    {
        let mut os = OutputState::new(&tok, params(2));
        let r = os.append(&[encode(&tok, "Hi")[0], EOS]).expect("appends");
        assert_eq!(r.finish, Some(FinishReason::Stop));
    }
    // max_tokens cuts a speculative step
    {
        let mut os = OutputState::new(&tok, params(3));
        let r = os.append(&encode(&tok, "one two three four five")).expect("appends");
        assert_eq!(r.finish, Some(FinishReason::Length));
        assert_eq!(r.new_ids.len(), 3);
    }
    // stop string: held back while streaming, truncated, reason = the string
    {
        let mut os = OutputState::new(
            &tok,
            StopParams {
                stop: vec!["STOP".to_owned()],
                ..params(100)
            },
        );
        let r1 = os.append(&encode(&tok, "abc ST")).expect("appends");
        assert_eq!(r1.finish, None);
        let t1 = os.next_text(false);
        assert!(
            t1.len() <= "abc ST".len() - 3,
            "the last 3 characters are held back: {t1:?}"
        );
        let r2 = os.append(&encode(&tok, "OP tail")).expect("appends");
        assert_eq!(r2.finish, Some(FinishReason::Stop));
        assert_eq!(r2.stop_reason, Some(StopReason::Text("STOP".to_owned())));
        assert_eq!(t1 + &os.next_text(true), "abc ");
        assert!(os.finished());
    }
}

#[test]
fn stop_token_id_and_included_stop_string() {
    let Some(tok) = tokenizer() else { return };
    let mut os = OutputState::new(
        &tok,
        StopParams {
            stop_token_ids: vec![END_MESSAGE],
            skip_special_tokens: false,
            ..params(100)
        },
    );
    let mut step = encode(&tok, "done");
    step.push(END_MESSAGE);
    let r = os.append(&step).expect("appends");
    assert_eq!(r.finish, Some(FinishReason::Stop));
    assert_eq!(r.stop_reason, Some(StopReason::TokenId(END_MESSAGE)));
    assert_eq!(os.next_text(true), "done");

    let mut os = OutputState::new(
        &tok,
        StopParams {
            stop: vec!["END".to_owned()],
            include_stop_str_in_output: true,
            ..params(100)
        },
    );
    let r = os.append(&encode(&tok, "the END then more")).expect("appends");
    assert_eq!(r.finish, Some(FinishReason::Stop));
    assert_eq!(os.full_text(true), "the END");
}

/// Streams `out` in steps of `chunk` ids and returns the text handed out and the ids kept.
fn stream(tok: &Tokenizer, out: &[u32], skip: bool, chunk: usize) -> String {
    let mut os = OutputState::new(
        tok,
        StopParams {
            max_tokens: usize::MAX,
            skip_special_tokens: skip,
            ..StopParams::default()
        },
    );
    let mut text = String::new();
    for step in out.chunks(chunk) {
        let r = os.append(step).expect("appends");
        assert_eq!(r.finish, None);
        text += &os.next_text(false);
    }
    assert_eq!(os.output_ids(), out);
    text + &os.next_text(true)
}

/// The streamed text of a recorded output equals decoding it at once, except where the stream ends inside a
/// character (decode writes U+FFFD, the stream writes nothing for the incomplete bytes).
fn streamed_equals_decoded(tok: &Tokenizer, out: &[u32], skip: bool) -> bool {
    let whole = tok.decode(out, skip).expect("decodes");
    [1, 3, out.len().max(1)].into_iter().all(|chunk| {
        let streamed = stream(tok, out, skip, chunk);
        streamed == whole || whole.strip_suffix('\u{fffd}').is_some_and(|w| streamed.starts_with(w))
    })
}

#[test]
fn recorded_outputs_stream_to_the_decoded_text() {
    let (Some(tok), Some(chat), Some(completions)) =
        (tokenizer(), rows("chat_outputs.jsonl"), rows("completions.jsonl"))
    else {
        return;
    };
    let mut bad = Vec::new();
    for c in chat.iter().chain(&completions) {
        let skip = c["skip_special"].as_bool().expect("flag");
        if !streamed_equals_decoded(&tok, &ids(&c["out_ids"]), skip) {
            bad.push(c["src"].to_string());
        }
    }
    let total = chat.len() + completions.len();
    println!(
        "PARITY streamed text == decoded text: {} identical / {total} total",
        total - bad.len()
    );
    assert!(bad.is_empty(), "differing outputs: {bad:?}");
}

#[test]
fn recorded_completions_stream_to_config_b_text() {
    let (Some(tok), Some(cases)) = (tokenizer(), rows("completions.jsonl")) else {
        return;
    };
    let mut bad = Vec::new();
    for c in &cases {
        let got = stream(&tok, &ids(&c["out_ids"]), c["skip_special"].as_bool().expect("flag"), 1);
        let want = c["out_text"].as_str().expect("text");
        // a stop string ends config B's text before the tokens that spelled it
        if got != want && !(c["finish"] == "stop" && got.starts_with(want)) {
            bad.push(c["src"].to_string());
        }
    }
    println!(
        "PARITY completions streamed text == config B text: {} identical / {} total",
        cases.len() - bad.len(),
        cases.len()
    );
    assert!(bad.is_empty(), "differing outputs: {bad:?}");
}
