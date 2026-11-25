//! Launch smoke test of the kernel pack (docs/RUST-PLAN.md section 8, row R2): load the embedded cubins, resolve
//! every kernel, and launch each once on zeroed buffers.
//!
//! Each launch: grid 1, block 32, the largest dynamic shared memory the kernel accepts (tried from 160 KiB down, R3:
//! kernels with `extern __shared__` storage faulted with 0), cooperative kernels through `cuLaunchCooperativeKernel`;
//! every pointer parameter, and every pointer field of a by-value struct (offsets from kpack/layout.json), points at
//! one zeroed 256 MiB buffer, re-zeroed before each launch; every scalar and every other struct byte is 0. The stream
//! is synchronised after each launch. This proves loading, symbol lookup, parameter sizes accepted by the driver and
//! that each kernel runs to completion on empty work; it checks no numerics (that is E2-E4). A kernel whose grid, not
//! a count, defines its work (it reads item `blockIdx.x` unconditionally) may fault on zeros: a recipe limit, not a
//! pack defect; its line says so only through the error.
//!
//! Usage: `smoke [FIRST [LAST]]` (built with INK_KPACK_DIR, one GPU visible): kernels FIRST..=LAST in KERNELS order
//! (default all). One JSON line per kernel on stdout, then a summary; exit 1 if any kernel failed. After a fault the
//! context is unusable: the run stops there (run one index per process to see every kernel).

use std::collections::HashMap;
use std::time::Instant;

use ink_cuda::{Arena, Context, KernelArgs, LaunchCfg};
use ink_kernels::{CUBINS, KERNELS, KPACK_ID, Kernels};
use serde_json::{Value, json};

const ZERO_BYTES: usize = 256 << 20;
/// Storage of one by-value parameter (the driver reads the parameter's own size from it).
type Slot = [u8; 512];

/// Byte offsets of every pointer inside struct `name`, at `base`.
fn ptr_offsets(structs: &HashMap<String, Value>, name: &str, base: usize, out: &mut Vec<usize>) {
    for f in structs[name]["fields"].as_array().expect("fields") {
        let ty = f["type"].as_str().expect("type");
        let off = base + f["offset"].as_u64().expect("offset") as usize;
        let (elem, n) = match ty.strip_suffix(']').and_then(|t| t.rsplit_once('[')) {
            Some((e, n)) => (e.trim(), n.trim().parse::<usize>().expect("array length")),
            None => (ty, 1),
        };
        let stride = if elem.ends_with('*') {
            8
        } else if let Some(s) = structs.get(elem) {
            s["size"].as_u64().expect("size") as usize
        } else {
            continue; // scalar (array)
        };
        for i in 0..n {
            if elem.ends_with('*') {
                out.push(off + i * stride);
            } else {
                ptr_offsets(structs, elem, off + i * stride, out);
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let layout: Value = serde_json::from_str(include_str!("../kpack/layout.json"))?;
    let structs: HashMap<String, Value> = layout["structs"]
        .as_array()
        .expect("structs")
        .iter()
        .map(|s| (s["name"].as_str().expect("name").to_owned(), s.clone()))
        .collect();
    let ctx = Context::new(0)?;
    let dev = ctx.device()?;
    let stream = ctx.new_stream()?;
    let t0 = Instant::now();
    let kernels = Kernels::load(&ctx)?;
    let load_ms = t0.elapsed().as_secs_f64() * 1e3;
    println!(
        "{}",
        json!({"item": "load", "kpack": KPACK_ID, "cubins": CUBINS.iter().map(|c| c.file).collect::<Vec<_>>(),
               "kernels": kernels.functions().len(), "ms": load_ms, "device": format!("{dev:?}")})
    );
    let mut arena = Arena::new(&ctx, &stream, ZERO_BYTES)?;
    let zero = arena.take::<u8>(ZERO_BYTES - 256, "zero")?;
    let addr = zero.ptr();
    let addr = bytemuck::bytes_of(&addr);
    let arg = |i: usize, default: usize| -> Result<usize, Box<dyn std::error::Error>> {
        Ok(std::env::args()
            .nth(i)
            .map(|a| a.parse())
            .transpose()?
            .unwrap_or(default))
    };
    let first = arg(1, 0)?;
    let last = arg(2, KERNELS.len() - 1)?.min(KERNELS.len() - 1);
    if first > last {
        return Err(format!("FIRST {first} > LAST {last}").into());
    }
    let (mut ok, mut failed) = (0, Vec::new());
    for (info, f) in KERNELS.iter().zip(kernels.functions()).take(last + 1).skip(first) {
        let mut args = KernelArgs::new();
        for p in info.params {
            if p.ends_with('*') {
                args.buf(&zero);
            } else if structs.contains_key(*p) {
                let mut slot: Slot = [0; 512];
                let mut offs = Vec::new();
                ptr_offsets(&structs, p, 0, &mut offs);
                for o in offs {
                    slot[o..o + 8].copy_from_slice(addr);
                }
                args.value(&slot);
            } else {
                args.value(&[0u8; 16]);
            }
        }
        stream.memset_zero(&zero)?;
        // the largest dynamic shared memory the kernel accepts (the limit is per block, minus its static storage)
        let smem = (0..=20u32)
            .map(|i| 160 * 1024 - i * 8 * 1024)
            .find(|&b| f.set_max_dynamic_smem(b).is_ok())
            .unwrap_or(0);
        let cfg = LaunchCfg {
            smem,
            ..LaunchCfg::linear(1, 32)
        };
        let t = Instant::now();
        // SAFETY: one argument per parameter in order, each slot at least the parameter's size (the driver copies
        // the size the kernel declares); every pointer is the zeroed buffer, every count 0: the kernel's work is empty
        // and whatever it reads at index 0..few lies in the 256 MiB buffer. A fault is reported, not hidden.
        let launched = if info.coop {
            // SAFETY: as above.
            unsafe { stream.launch_cooperative(f, cfg, &mut args) }
        } else {
            // SAFETY: as above.
            unsafe { stream.launch(f, cfg, &mut args) }
        };
        let res = launched.and_then(|()| stream.synchronize());
        let regs = f.num_regs().ok(); // after a fault the context refuses this too
        let ms = t.elapsed().as_secs_f64() * 1e3;
        let line = json!({"key": info.key, "symbol": info.symbol, "params": info.params.len(), "coop": info.coop,
                          "regs": regs, "smem": smem, "ok": res.is_ok(), "ms": ms,
                          "error": res.as_ref().err().map(|e| e.to_string())});
        println!("{line}");
        match res {
            Ok(()) => ok += 1,
            Err(_) => {
                failed.push(info.key);
                break;
            }
        }
    }
    println!(
        "{}",
        json!({"item": "summary", "kernels": KERNELS.len(), "first": first, "last": last, "launched_ok": ok,
               "failed": failed})
    );
    if ok == last + 1 - first {
        Ok(())
    } else {
        Err("smoke test failed".into())
    }
}
