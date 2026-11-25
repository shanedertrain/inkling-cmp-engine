//! The Rust engine on a card: slot reset against an uncommitted call (review V6). Needs a GPU, the model and a pin
//! file: `INK_MODEL_DIR=/model INK_GEMM_ALGOS=pins.json cargo test -p ink-runtime --test engine_gpu -- --ignored`.

use std::path::PathBuf;

use ink_rt_api::{EngineConfig, ErrorKind, Runtime, SegIn};
use ink_runtime::Engine;

fn engine() -> Engine {
    let var = |k: &str| PathBuf::from(std::env::var(k).unwrap_or_else(|_| panic!("{k} is not set")));
    Engine::new(&EngineConfig {
        model_dir: var("INK_MODEL_DIR"),
        gemm_algos: var("INK_GEMM_ALGOS"),
        max_slots: 2,
        num_pages: 8,
        ..EngineConfig::default()
    })
    .expect("engine")
}

/// `reset_slot` of a slot in the uncommitted last call is refused (a request error, nothing changes); a slot outside
/// the call resets; after the call is committed (0 rows drops it) the slot resets and the engine runs on.
#[test]
#[ignore = "needs a GPU, INK_MODEL_DIR and INK_GEMM_ALGOS"]
fn reset_of_a_slot_in_the_uncommitted_call_is_refused() {
    let mut e = engine();
    let ids: Vec<i32> = (1000..1100).collect();
    let prefill = |e: &mut Engine| {
        e.forward(&[SegIn {
            slot: 0,
            ids: &ids,
            logits: 1,
        }])
        .expect("prefill")
    };
    assert_eq!(prefill(&mut e), 1);
    let free = e.free_pages().expect("pages");
    let err = e.reset_slot(0).expect_err("slot 0 is in the uncommitted call");
    assert_eq!(err.kind(), ErrorKind::Request, "{err}");
    assert!(err.message().contains("uncommitted"), "{err}");
    assert_eq!(
        e.free_pages().expect("pages"),
        free,
        "the refused reset kept the call's pages"
    );
    e.reset_slot(1).expect("slot 1 is not in the call");
    // the refused reset left the call intact: it commits, and the slot holds its rows
    e.commit(&[100]).expect("commit");
    assert_eq!(e.length(0).expect("slot"), 100);
    e.reset_slot(0).expect("committed: the slot resets");
    assert_eq!(e.length(0).expect("slot"), 0);
    // a call dropped with a commit of 0 rows leaves the slot empty and resettable
    assert_eq!(prefill(&mut e), 1);
    e.commit(&[0]).expect("drop the call");
    assert_eq!(e.length(0).expect("slot"), 0);
    e.reset_slot(0).expect("reset after the drop");
    assert_eq!(e.free_pages().expect("pages"), 8, "every page back in the pool");
}
