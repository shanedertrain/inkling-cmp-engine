//! Gate G-FE3: config B's recorded token streams through the front end over the scripted back end, with the
//! numbers of the C++ replay (`bench/raw/2026-11-23-S5-server-parity/response-parity-cap{1,2,3}.txt`).
//!
//! Needs the model's tokenizer files and the captures, both kept out of git; the test says so and passes when the
//! environment is unset:
//!
//! ```sh
//! python3 -I engine/rust/crates/ink-tools/tests/prepare_fixtures.py          # -> engine/rust/testdata-local/replay/
//! # from engine/rust/
//! scripts/build.sh env INK_MODEL_DIR=/w/testdata-local/inkling INK_REPLAY_DIR=/w/testdata-local/replay \
//!   cargo test -p ink-tools --release --offline --locked --test replay_gate -- --nocapture
//! ```

use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;

fn env_dir(name: &str) -> Option<PathBuf> {
    let dir = std::env::var_os(name)
        .map(PathBuf::from)
        .filter(|p| !p.as_os_str().is_empty());
    if dir.is_none() {
        eprintln!("SKIPPED: {name} is not set (see the module documentation of tests/replay_gate.rs)");
    }
    dir
}

/// Gate G-FE4: 20 sampled requests of capture 1 (streaming, non-streaming and `prod` records) over a loopback socket
/// through the router give the same frames and bodies as the direct replay, byte for byte, and the same verdicts.
#[test]
fn http_replay_equals_the_direct_replay() {
    let (Some(model_dir), Some(data_dir)) = (env_dir("INK_MODEL_DIR"), env_dir("INK_REPLAY_DIR")) else {
        return;
    };
    let tmp = std::env::temp_dir().join(format!("ink-replay-http-{}", std::process::id()));
    let capture = data_dir.join("cap1");
    let mut dumps = Vec::new();
    let mut results = Vec::new();
    for http in [false, true] {
        let dump = tmp.join(if http { "http" } else { "direct" });
        let out = tmp.join(if http { "http.json" } else { "direct.json" });
        std::fs::create_dir_all(&dump).expect("dump directory");
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_ink-replay"));
        cmd.args([model_dir.as_path(), capture.as_path(), out.as_path()])
            .args(["--sample", "20", "--lpids"])
            .arg(&capture)
            .arg("--dump")
            .arg(&dump);
        if http {
            cmd.arg("--http");
        }
        assert!(cmd.status().expect("ink-replay starts").success(), "http = {http}");
        results.push(serde_json::from_str::<Value>(&std::fs::read_to_string(&out).expect("result")).expect("JSON"));
        dumps.push(dump);
    }
    let names = |dir: &Path| {
        let mut v: Vec<_> = std::fs::read_dir(dir)
            .expect("dump directory")
            .map(|e| e.expect("entry").file_name())
            .collect();
        v.sort();
        v
    };
    let files = names(&dumps[0]);
    assert_eq!(files, names(&dumps[1]), "the same requests were answered");
    assert!(files.len() >= 15, "about 20 samples, some skipped: {files:?}");
    for name in &files {
        assert_eq!(
            std::fs::read(dumps[0].join(name)).expect("direct"),
            std::fs::read(dumps[1].join(name)).expect("http"),
            "{name:?} over HTTP differs from the direct replay"
        );
    }
    for key in ["identical", "total", "skipped"] {
        assert_eq!(results[0][key], results[1][key], "{key}");
    }
    eprintln!(
        "HTTP replay: {} files byte-identical to the direct replay, {}/{} identical to config B",
        files.len(),
        results[1]["identical"],
        results[1]["total"]
    );
    std::fs::remove_dir_all(&tmp).expect("remove the temporary directory");
}

/// Runs `ink-replay` on one capture and returns its result file.
fn replay(model_dir: &Path, data_dir: &Path, cap: &str) -> Value {
    let out = std::env::temp_dir().join(format!("ink-replay-{cap}-{}.json", std::process::id()));
    let capture = data_dir.join(cap);
    let status = Command::new(env!("CARGO_BIN_EXE_ink-replay"))
        .args([model_dir, &capture, &out])
        .arg("--lpids")
        .arg(&capture)
        .status()
        .expect("ink-replay starts");
    assert!(status.success(), "ink-replay failed on {cap}");
    let text = std::fs::read_to_string(&out).expect("result file");
    std::fs::remove_file(&out).expect("remove the result file");
    serde_json::from_str(&text).expect("result JSON")
}

fn count(result: &Value, key: &str) -> (u64, u64) {
    let e = &result["by_endpoint"][key];
    (e["identical"].as_u64().unwrap_or(0), e["total"].as_u64().unwrap_or(0))
}

#[test]
fn replay_matches_config_b_as_the_cpp_replay_did() {
    let (Some(model_dir), Some(data_dir)) = (env_dir("INK_MODEL_DIR"), env_dir("INK_REPLAY_DIR")) else {
        return;
    };
    // (identical, total, skipped) per capture, as response-parity-cap{1,2,3}.txt
    let expected = [("cap1", 795, 798, 8), ("cap2", 81, 89, 2), ("cap3", 8, 8, 0)];
    let (mut chat_stream, mut chat_full, mut completions) = ((0, 0), (0, 0), (0, 0));
    let mut failing_non_prod = Vec::new();
    for (cap, identical, total, skipped) in expected {
        let r = replay(&model_dir, &data_dir, cap);
        let got = (r["identical"].as_u64(), r["total"].as_u64(), r["skipped"].as_u64());
        assert_eq!(got, (Some(identical), Some(total), Some(skipped)), "{cap}");
        assert_eq!(
            r["key_order_differences"], 0,
            "{cap}: identical rows also have config B's key order"
        );
        let add = |acc: &mut (u64, u64), keys: &[&str]| {
            for key in keys {
                let (same, n) = count(&r, key);
                acc.0 += same;
                acc.1 += n;
            }
        };
        add(&mut chat_stream, &["chat/stream"]);
        add(&mut chat_full, &["chat/nonstream"]);
        add(&mut completions, &["completions/stream", "completions/nonstream"]);
        for row in r["rows"].as_array().expect("rows") {
            if row["identical"] == false && row["mode"] != "prod" {
                failing_non_prod.push((
                    cap,
                    row["id"].as_str().unwrap_or_default().to_owned(),
                    row["mode"].clone(),
                    row["differs_only_in_logprobs"].clone(),
                ));
            }
        }
    }
    eprintln!("chat SSE {chat_stream:?}, chat JSON {chat_full:?}, completions {completions:?}");
    assert_eq!(chat_stream, (435, 435));
    assert_eq!(chat_full, (435, 435));
    assert_eq!(completions, (11, 12));
    // the one miss is config B differing from itself: the concurrent logprob pair of capture 2, in its logprobs field only
    assert_eq!(failing_non_prod.len(), 1, "{failing_non_prod:?}");
    let (cap, id, mode, logprobs_only) = &failing_non_prod[0];
    assert_eq!(
        (*cap, id.as_str(), mode.as_str(), logprobs_only.as_bool()),
        ("cap2", "cmp-ids-logprobs", Some("stream"), Some(true))
    );
}
