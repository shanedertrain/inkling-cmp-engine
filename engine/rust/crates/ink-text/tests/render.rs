//! Gate G-FE1: prompt token ids of the Rust renderer against config B's recorded ids (package F2).
//!
//! Needs the model's tokenizer files (`INK_MODEL_DIR`) and the recordings extracted by `tests/prepare_fixtures.py`
//! (`INK_PARITY_DIR`); see `README.md` of the crate. Without them a test prints `SKIPPED` and passes.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use ink_text::{ChatRenderer, RequestError, Tokenizer, resolve_chat_params};
use serde_json::{Value, json};

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

fn render_body(tok: &Tokenizer, body: &Value) -> Result<Vec<u32>, RequestError> {
    ChatRenderer::new(tok)
        .expect("Inkling special tokens are present")
        .render(&resolve_chat_params(body)?)
}

fn ids(v: &Value) -> Vec<u32> {
    v.as_array()
        .expect("ids are an array")
        .iter()
        .map(|n| u32::try_from(n.as_u64().expect("id is an unsigned integer")).expect("id fits u32"))
        .collect()
}

/// Renders every request and compares with every reference config B recorded for it. A recorded error (HTTP 4xx)
/// must be an error here too. Prints the equality line the report quotes.
fn check_render_parity(file: &str, label: &str, expected_total: usize) {
    let (Some(tok), Some(cases)) = (tokenizer(), rows(file)) else {
        return;
    };
    let mut by_group: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    let mut differing = Vec::new();
    for c in &cases {
        let id = c["id"].as_str().expect("id");
        let got = render_body(&tok, &c["body"]);
        let refs = c["refs"].as_object().expect("refs");
        assert!(!refs.is_empty(), "{id}: no recorded reference");
        let ok = refs.values().all(|r| match (&got, r.get("ids")) {
            (Ok(g), Some(want)) => *g == ids(want),
            (Err(_), None) => r.get("error").is_some(),
            _ => false,
        });
        let entry = by_group
            .entry(c["group"].as_str().expect("group").to_owned())
            .or_default();
        entry.1 += 1;
        if ok {
            entry.0 += 1;
        } else {
            let detail = match &got {
                Ok(g) => format!("{} ids", g.len()),
                Err(e) => format!("error: {e}"),
            };
            differing.push(format!("{id} ({detail}; refs {:?})", refs.keys().collect::<Vec<_>>()));
        }
    }
    let total = cases.len();
    let same: usize = by_group.values().map(|g| g.0).sum();
    println!("PARITY {label}: {same} identical / {total} total");
    for (group, (s, t)) in &by_group {
        println!("PARITY   {group}: {s}/{t}");
    }
    assert_eq!(total, expected_total, "{label}: number of recorded requests");
    assert!(differing.is_empty(), "{label}: differing requests: {differing:#?}");
}

#[test]
fn g_fe1_render_requests_equal_config_b_ids() {
    // requests.json: 445 chat requests (351 agent-panel turns, 28 S0 cases, 45 render branches, 21 generations);
    // one of the render branches is a request config B rejects with HTTP 400 and must be rejected here too.
    check_render_parity("render_requests.jsonl", "requests.json chat requests", 445);
}

#[test]
fn g_fe1_s0_golden_requests_equal_recorded_ids() {
    // s0-golden-chat-requests.json: the 29 S0 cases, including the 125,001-token one, against the S0 capture.
    check_render_parity("render_s0.jsonl", "S0 golden chat requests", 29);
}

#[test]
fn text_render_matches_config_b() {
    // Port of test_text.cpp `text_render_matches_config_b`: the six cases of render_goldens.inc, by request id.
    let (Some(tok), Some(cases)) = (tokenizer(), rows("render_requests.jsonl")) else {
        return;
    };
    for id in [
        "syn-sys-user",
        "syn-tool-roundtrip",
        "syn-effort-kwarg-string-num",
        "syn-continue-final",
        "syn-tools-named-noparallel",
        "syn-tool-named-result",
    ] {
        let c = cases
            .iter()
            .find(|c| c["id"] == id)
            .unwrap_or_else(|| panic!("{id} is among the recorded requests"));
        let want = ids(&c["refs"]["render"]["ids"]);
        assert_eq!(render_body(&tok, &c["body"]).expect("renders"), want, "{id}");
    }
}

#[test]
fn text_render_errors() {
    // Port of test_text.cpp `text_render_errors`.
    let Some(tok) = tokenizer() else { return };
    let rejects = |body: Value| render_body(&tok, &body).is_err();
    assert!(rejects(json!({"messages": []})));
    assert!(rejects(
        json!({"messages": [{"role": "user", "content": "x"}], "tool_choice": "required"})
    ));
    assert!(rejects(
        json!({"messages": [{"role": "user", "content": "x"}], "continue_final_message": true})
    ));
    assert!(rejects(json!({"messages": [{"role": "robot", "content": "x"}]})));
    assert!(rejects(
        json!({"messages": [{"role": "user", "content": "x"}], "reasoning_effort": "huge"})
    ));
    assert!(rejects(json!({
        "messages": [{"role": "user", "content": "x"}],
        "chat_template_kwargs": {"reasoning_effort": 1.5}
    })));
    assert!(rejects(json!({
        "messages": [{"role": "user", "content": "x"}],
        "tools": [{"type": "function", "function": {"name": "a"}}],
        "tool_choice": {"type": "function", "function": {"name": "b"}}
    })));
    let err = render_body(&tok, &json!({"messages": []})).expect_err("rejected");
    assert_eq!(err.param(), Some("messages"));
}
