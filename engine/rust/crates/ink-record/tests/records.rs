//! Acceptance tests of package V1: identity validation and the writer's dirty refusal.

use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU32, Ordering},
};

use ink_record::{
    BUILD_LINE, Component, ComponentName, Context, DirtyPolicy, Engine, Identity, RecordError, RecordWriter, Source,
};
use serde_json::{Value, json};

const COMMIT: &str = "2bdcac0f3a1d5e7b9c4d6e8f0a2b4c6d8e0f1a3b";
const DIFF: &str = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08";

fn scratch() -> PathBuf {
    static N: AtomicU32 = AtomicU32::new(0);
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!(
        "records-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&dir).expect("scratch dir is creatable under the target directory");
    dir
}

fn source(engine: Engine, dirty: bool) -> Source {
    Source {
        implementation: engine,
        version: "0.2.0".to_owned(),
        commit: COMMIT.to_owned(),
        dirty,
        diff: dirty.then(|| DIFF.to_owned()),
        abi: None,
    }
}

fn clean() -> Identity {
    Identity::new(
        source(Engine::Native, false),
        "M2",
        [(ComponentName::Tools, Component::Source(source(Engine::Rust, false)))],
    )
}

fn context() -> Context {
    Context {
        argv: vec!["ink-bench".to_owned()],
        config: json!({"graphs": true}),
        label: "t".to_owned(),
        ..Context::default()
    }
}

fn lines(path: &std::path::Path) -> Vec<Value> {
    fs::read_to_string(path)
        .expect("record file is readable")
        .lines()
        .map(|l| serde_json::from_str(l).expect("every line is json"))
        .collect()
}

#[test]
fn clean_identity_writes_a_jsonl_record_with_identity() {
    let dir = scratch();
    let mut w =
        RecordWriter::create(&dir, "run", clean(), context(), DirtyPolicy::Refuse).expect("clean tree is accepted");
    w.append(&json!({"step_ms": 47.5})).expect("append");
    w.append(&json!({"step_ms": 47.6})).expect("append");
    assert_eq!(w.path(), dir.join("run.jsonl"));
    let rows = lines(w.path());
    assert_eq!(rows.len(), 2);
    let id = &rows[0]["identity"];
    assert_eq!(rows[0]["schema"], "ink-record/1");
    assert_eq!(id["engine"], "native");
    assert_eq!(id["commit"], COMMIT);
    assert_eq!(id["dirty"], false);
    for key in ["lock_sha256", "vendor_sha256"] {
        let h = id[key].as_str().expect("hash is a string");
        assert!(h.len() == 64 && h.bytes().all(|b| b.is_ascii_hexdigit()), "{key}: {h}");
    }
    assert_eq!(id["lock_sha256"], clean().lock_sha256);
    assert_eq!(id["vendor_sha256"], clean().vendor_sha256);
    assert_eq!(id["components"]["runtime"]["impl"], "native");
    assert_eq!(id["components"]["tools"]["impl"], "rust");
    assert_eq!(rows[1]["data"]["step_ms"], 47.6);
    assert_eq!(rows[0]["context"]["argv"][0], "ink-bench");
}

#[test]
fn dirty_runtime_is_refused_and_leaves_no_file() {
    let dir = scratch();
    let id = Identity::new(source(Engine::Native, true), "M0", []);
    let err = RecordWriter::create(&dir, "run", id, context(), DirtyPolicy::Refuse).expect_err("dirty must be refused");
    assert!(
        matches!(
            err,
            RecordError::Dirty {
                component: ComponentName::Runtime
            }
        ),
        "got {err}"
    );
    assert_eq!(fs::read_dir(&dir).expect("dir").count(), 0, "nothing may be written");
}

#[test]
fn dirty_other_component_is_refused() {
    let dir = scratch();
    let id = Identity::new(
        source(Engine::Native, false),
        "M2",
        [(ComponentName::Tools, Component::Source(source(Engine::Rust, true)))],
    );
    let err = RecordWriter::create(&dir, "run", id, context(), DirtyPolicy::Refuse).expect_err("dirty tools refused");
    assert!(
        matches!(
            err,
            RecordError::Dirty {
                component: ComponentName::Tools
            }
        ),
        "got {err}"
    );
}

#[test]
fn allowed_dirty_record_goes_to_dirty_file_and_carries_diff() {
    let dir = scratch();
    let id = Identity::new(source(Engine::Rust, true), "M5", []);
    let mut w = RecordWriter::create(&dir, "run", id, context(), DirtyPolicy::Allow).expect("allowed");
    w.append(&json!({})).expect("append");
    assert_eq!(w.path(), dir.join("run.dirty.jsonl"));
    let id = &lines(w.path())[0]["identity"];
    assert_eq!(id["dirty"], true);
    assert_eq!(id["diff"], DIFF);
}

#[test]
fn allowing_dirty_does_not_rename_a_clean_record() {
    let dir = scratch();
    let w = RecordWriter::create(&dir, "run", clean(), context(), DirtyPolicy::Allow).expect("clean");
    assert_eq!(w.path(), dir.join("run.jsonl"));
}

#[test]
fn missing_or_malformed_identity_is_refused_even_when_dirty_is_allowed() {
    let dir = scratch();
    let mut bad = clean();
    bad.commit = String::new();
    for policy in [DirtyPolicy::Refuse, DirtyPolicy::Allow] {
        let err = RecordWriter::create(&dir, "run", bad.clone(), context(), policy).expect_err("no commit");
        assert!(matches!(err, RecordError::Identity(_)), "got {err}");
    }

    let mut no_runtime = clean();
    no_runtime.components.remove(&ComponentName::Runtime);
    assert!(no_runtime.validate().is_err());

    let mut dirty_without_diff = clean();
    dirty_without_diff.dirty = true;
    assert!(dirty_without_diff.validate().is_err());

    for field in ["lock", "vendor"] {
        let mut bad_hash = clean();
        match field {
            "lock" => bad_hash.lock_sha256 = "abc".to_owned(),
            _ => bad_hash.vendor_sha256 = "G".repeat(64),
        }
        assert!(bad_hash.validate().is_err(), "{field} hash must be validated");
        let err = RecordWriter::create(&dir, "run", bad_hash, context(), DirtyPolicy::Allow).expect_err(field);
        assert!(matches!(err, RecordError::Identity(_)), "got {err}");
    }

    let mut upper = clean();
    upper.commit = COMMIT.to_uppercase();
    assert!(upper.validate().is_err());

    let mut bad_pack = clean();
    bad_pack
        .components
        .insert(ComponentName::Kpack, Component::Kpack { id: "abc".to_owned() });
    assert!(bad_pack.validate().is_err());
}

#[test]
fn record_names_must_be_plain_and_files_are_never_overwritten() {
    let dir = scratch();
    for name in ["", "../x", "a/b", ".hidden"] {
        let err = RecordWriter::create(&dir, name, clean(), context(), DirtyPolicy::Refuse).expect_err(name);
        assert!(matches!(err, RecordError::InvalidName(_)), "{name:?}: {err}");
    }
    RecordWriter::create(&dir, "run", clean(), context(), DirtyPolicy::Refuse).expect("first");
    let err = RecordWriter::create(&dir, "run", clean(), context(), DirtyPolicy::Refuse).expect_err("second");
    assert!(matches!(err, RecordError::Io { .. }), "got {err}");
}

#[test]
fn identity_round_trips_through_json() {
    let id = Identity::new(
        source(Engine::Native, false),
        "M4",
        [
            (ComponentName::Kpack, Component::Kpack { id: DIFF.to_owned() }),
            (
                ComponentName::Drafter,
                Component::Source(Source {
                    abi: Some(1),
                    ..source(Engine::Native, false)
                }),
            ),
        ],
    );
    let text = serde_json::to_string(&id).expect("serialise");
    let back: Identity = serde_json::from_str(&text).expect("parse");
    assert_eq!(back, id);
    back.validate().expect("valid");
}

#[test]
fn this_build_is_a_valid_identity_and_matches_the_version_line() {
    let src = Source::this_build();
    let id = Identity::new(src.clone(), "M2", []);
    id.validate().expect("the embedded build identity is valid");
    if let Ok(text) = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../BUILD_ID.json")) {
        let build_id: Value = serde_json::from_str(&text).expect("BUILD_ID.json is json");
        assert_eq!(id.lock_sha256, build_id["lock"].as_str().expect("lock"));
        assert_eq!(id.vendor_sha256, build_id["vendor"].as_str().expect("vendor"));
    }
    assert!(BUILD_LINE.contains(&format!("commit={}", src.commit)), "{BUILD_LINE}");
    assert!(
        BUILD_LINE.starts_with(&format!("rust version={}", src.version)),
        "{BUILD_LINE}"
    );
}
