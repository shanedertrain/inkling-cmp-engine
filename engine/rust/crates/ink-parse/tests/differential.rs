//! Gate G-FE2: the parser against config B's own parser (vLLM's, run in config B's image), on the recorded token
//! streams re-chunked four ways, streaming and non-streaming.
//!
//! Needs the model's tokenizer files and the recordings, both kept out of git; the test says so and passes when the
//! environment is unset:
//!
//! ```sh
//! python3 -I engine/rust/crates/ink-parse/tests/prepare_fixtures.py          # -> engine/rust/testdata-local/parser/
//! # from engine/rust/
//! scripts/build.sh env INK_MODEL_DIR=/w/testdata-local/inkling INK_PARSER_DIR=/w/testdata-local/parser \
//!   cargo test -p ink-parse --offline --locked --release --test differential -- --nocapture
//! ```

mod common;

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use common::{fixed_ids, steps_of};
use ink_parse::{InklingParser, ParseRequest, ParserTables, ToolChoice, special_tokens_from_dir};
use ink_text::Tokenizer;
use serde_json::{Value, json};

/// The id of the end-of-sequence token of Inkling-2-Mini.
const EOS: u32 = 200_006;

fn read_lines(path: &PathBuf) -> Vec<Value> {
    fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .lines()
        .map(|l| serde_json::from_str(l).expect("a JSON line"))
        .collect()
}

fn env_dir(name: &str) -> Option<PathBuf> {
    let dir = std::env::var_os(name)
        .map(PathBuf::from)
        .filter(|p| !p.as_os_str().is_empty());
    if dir.is_none() {
        eprintln!("SKIPPED: {name} is not set (see the module documentation of tests/differential.rs)");
    }
    dir
}

/// Tool-call ids are random in config B and fixed here: compare with both replaced.
fn normalise(delta: &Value) -> Value {
    let mut v = delta.clone();
    if let Some(calls) = v.get_mut("tool_calls").and_then(Value::as_array_mut) {
        for call in calls {
            if let Some(id) = call.get_mut("id") {
                *id = json!("<ID>");
            }
        }
    }
    v
}

struct Outcome {
    deltas: Vec<Value>,
    full: Value,
}

fn run_case(tok: &Arc<Tokenizer>, tables: &Arc<ParserTables>, case: &Value) -> Outcome {
    let ids = |v: &Value| -> Vec<u32> {
        v.as_array()
            .expect("an id list")
            .iter()
            .map(|n| u32::try_from(n.as_u64().expect("an id")).expect("u32 id"))
            .collect()
    };
    let prompt = ids(&case["prompt_ids"]);
    let groups: Vec<Vec<u32>> = case["groups"].as_array().expect("groups").iter().map(ids).collect();
    let request = ParseRequest {
        tools: case["tools"].as_array().cloned().unwrap_or_default(),
        tool_choice: ToolChoice::from_value(&case["tool_choice"]),
        include_reasoning: case["include_reasoning"].as_bool().unwrap_or(true),
    };
    let max_tokens = case["max_tokens"]
        .as_u64()
        .map_or(usize::MAX, |n| usize::try_from(n).unwrap_or(usize::MAX));
    let steps = steps_of(tok, EOS, max_tokens, &groups);
    let mut streaming = InklingParser::new(Arc::clone(tables), request.clone(), Some(fixed_ids()));
    let deltas = steps
        .iter()
        .map(|s| {
            streaming
                .parse_delta(&s.text, &s.ids, &prompt, s.finished)
                .map_or(Value::Null, |d| d.to_value())
        })
        .collect();
    let full_text: String = steps.iter().map(|s| s.text.as_str()).collect();
    let parsed = InklingParser::new(Arc::clone(tables), request, Some(fixed_ids())).parse(&full_text);
    let tool_calls = parsed.tool_calls.map_or(Value::Null, |calls| {
        Value::Array(calls.iter().map(|c| json!([c.name, c.arguments])).collect())
    });
    Outcome {
        deltas,
        full: json!({"reasoning": parsed.reasoning, "content": parsed.content, "tool_calls": tool_calls}),
    }
}

#[test]
fn parser_differential_against_config_b() {
    let (Some(model_dir), Some(data_dir)) = (env_dir("INK_MODEL_DIR"), env_dir("INK_PARSER_DIR")) else {
        return;
    };
    let tok = Arc::new(Tokenizer::load(&model_dir).expect("tokenizer.json"));
    let specials = special_tokens_from_dir(&model_dir).expect("special_tokens_map.json");
    let tables = ParserTables::new(Arc::clone(&tok), &specials);

    // Ids repeat when a request was captured more than once (8 ids, 3 captures each): the k-th reference row of an
    // id belongs to the k-th case of that id.
    let mut cases: BTreeMap<String, Vec<Value>> = BTreeMap::new();
    for case in read_lines(&data_dir.join("cases.jsonl")) {
        cases
            .entry(case["id"].as_str().expect("case id").to_owned())
            .or_default()
            .push(case);
    }
    let rows = read_lines(&data_dir.join("ref.jsonl"));
    let mut occurrence: BTreeMap<String, usize> = BTreeMap::new();

    // (rows, streaming equal, non-streaming equal) per chunking; ids with at least one differing row
    let mut by_chunking: BTreeMap<String, [usize; 3]> = BTreeMap::new();
    let mut ids_seen = BTreeSet::new();
    let mut ids_failing = BTreeSet::new();
    let mut failures = Vec::new();
    for want in &rows {
        let id = want["id"].as_str().expect("reference id");
        let k = occurrence.entry(id.to_owned()).and_modify(|k| *k += 1).or_insert(0);
        let case = cases
            .get(id)
            .and_then(|v| v.get(*k))
            .unwrap_or_else(|| panic!("no case for reference row {id} #{k}"));
        let got = run_case(&tok, &tables, case);
        let want_deltas: Vec<Value> = want["deltas"]
            .as_array()
            .expect("deltas")
            .iter()
            .map(normalise)
            .collect();
        let got_deltas: Vec<Value> = got.deltas.iter().map(normalise).collect();
        let (stream_ok, full_ok) = (got_deltas == want_deltas, got.full == want["full"]);
        let tally = by_chunking
            .entry(id.rsplit('/').next().expect("chunking suffix").to_owned())
            .or_default();
        tally[0] += 1;
        tally[1] += usize::from(stream_ok);
        tally[2] += usize::from(full_ok);
        ids_seen.insert(id.to_owned());
        if !(stream_ok && full_ok) {
            ids_failing.insert(id.to_owned());
            let first = got_deltas.iter().zip(&want_deltas).position(|(a, b)| a != b);
            failures.push(format!(
                "{id}: stream_ok={stream_ok} full_ok={full_ok} steps ours/ref {}/{} first differing step {first:?}\n  ours {}\n  ref  {}",
                got_deltas.len(),
                want_deltas.len(),
                first.map_or_else(|| got.full.to_string(), |i| got_deltas[i].to_string()),
                first.map_or_else(|| want["full"].to_string(), |i| want_deltas[i].to_string()),
            ));
        }
    }
    let totals = by_chunking
        .values()
        .fold([0; 3], |a, t| [a[0] + t[0], a[1] + t[1], a[2] + t[2]]);
    for (chunking, [n, s, f]) in &by_chunking {
        eprintln!("{chunking}: {n} cases, streaming equal {s}/{n}, non-streaming equal {f}/{n}");
    }
    eprintln!(
        "total: {} rows, streaming equal {}/{}, non-streaming equal {}/{}",
        totals[0], totals[1], totals[0], totals[2], totals[0]
    );
    eprintln!(
        "distinct ids: {}, all rows equal for {} of them (the recorded gate counts distinct ids: 1,708)",
        ids_seen.len(),
        ids_seen.len() - ids_failing.len()
    );
    for f in failures.iter().take(10) {
        eprintln!("DIFF {f}");
    }
    assert_eq!(
        (totals[0], ids_seen.len()),
        (1724, 1708),
        "the recorded reference: 1,724 rows over 1,708 distinct ids"
    );
    assert!(failures.is_empty(), "{} of {} rows differ", failures.len(), totals[0]);
}
