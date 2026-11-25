//! R1 spike (docs/RUST-PLAN.md 3.2 and section 8, row R1): proves on one GPU, through `ink-cuda` only, that
//! the Rust side can do what the C++ engine does with CUDA before any other runtime package builds on it.
//!
//! Items (each printed as one JSON line with `"ok"`, also written to `<out>/spike.jsonl`):
//! startup checks (driver >= 13030, cuBLASLt major 13, every symbol), context and device, arena, cubin module with a
//! module global, kernel launch with a struct passed by value, `cuFuncGetName`, cooperative launch with a grid sync,
//! cuBLASLt bf16 GEMM with a pinned algorithm (pin file written and read back), stream capture of driver launches,
//! memcpy, memset, the GEMM and the cooperative launch with region marks, replay bit-identical to eager execution,
//! node disable by region and re-enable, many replays still bit-identical, events, NVML at 10 Hz.
//!
//! Timings printed here are indicative (no clock or power control, spike buffers): not performance numbers.
//!
//! Usage: `r1_spike --cubin spike.cubin --out DIR [--replays N]`. Writes `gemm_x.bin`, `gemm_w.bin`,
//! `gemm_y_rust.bin` (raw bf16), `gemm_meta.txt` and `gemm-algos.json` for `spike/lt_ref` (the C++ `LtGemm`).

use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context as _, Result, bail};
use bytemuck::{Pod, Zeroable};
use ink_cuda::{
    AlgoPins, Arena, Bf16, Context, DevBuf, DevPtr, Dtype, GemmShape, Graph, KernelArgs, LaunchCfg, LtGemm, Module,
    NodeKind, NodeState, PERIOD_10HZ, PlanClass, Sampler, Stream, startup_checks, summarize,
};
use serde_json::{Value, json};

/// Mirror of `SpikeParams` in spike/spike.cu (48 bytes, align 16; the padding is explicit so the struct is `Pod`).
#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct SpikeParams {
    x: DevPtr<f32>,
    y: DevPtr<f32>,
    a: f32,
    n: i32,
    _pad: [u32; 2],
    bias: [i32; 4],
}
const _: () = assert!(size_of::<SpikeParams>() == 48 && align_of::<SpikeParams>() == 16);
// SAFETY: repr(C) fields of Pod types with explicit padding: 8 + 8 + 4 + 4 + 8 + 16 = 48 = size, no implicit padding.
unsafe impl Zeroable for SpikeParams {}
// SAFETY: as above.
unsafe impl Pod for SpikeParams {}

const N_VEC: usize = 1 << 16;
const GEMM_M: u32 = 8;
const GEMM_N: u32 = 2304;
const GEMM_K: u32 = 2304;
const Z_LEN: usize = 4096;
const BLOCK: u32 = 256;

struct Report {
    file: fs::File,
    failed: Vec<String>,
}

impl Report {
    fn item(&mut self, item: &str, ok: bool, detail: Value) -> Result<()> {
        let line = json!({"item": item, "ok": ok, "detail": detail});
        println!("{line}");
        writeln!(self.file, "{line}")?;
        if !ok {
            self.failed.push(item.to_owned());
        }
        Ok(())
    }
}

struct Bufs {
    x: DevBuf<f32>,
    y0: DevBuf<f32>,
    y: DevBuf<f32>,
    gx: DevBuf<Bf16>,
    gw: DevBuf<Bf16>,
    gy: DevBuf<Bf16>,
    gf: DevBuf<f32>,
    z: DevBuf<f32>,
    ca: DevBuf<f32>,
    cb: DevBuf<f32>,
}

/// Everything the body produces, read back to the host.
#[derive(PartialEq)]
struct Outputs {
    y: Vec<f32>,
    gy: Vec<Bf16>,
    gf: Vec<f32>,
    z: Vec<f32>,
    cb: Vec<f32>,
}

impl Outputs {
    fn read(s: &Stream, b: &Bufs) -> Result<Outputs> {
        let mut o = Outputs {
            y: vec![0.0; b.y.len()],
            gy: vec![Bf16(0); b.gy.len()],
            gf: vec![0.0; b.gf.len()],
            z: vec![0.0; b.z.len()],
            cb: vec![0.0; b.cb.len()],
        };
        s.dtoh(&mut o.y, &b.y)?;
        s.dtoh(&mut o.gy, &b.gy)?;
        s.dtoh(&mut o.gf, &b.gf)?;
        s.dtoh(&mut o.z, &b.z)?;
        s.dtoh(&mut o.cb, &b.cb)?;
        Ok(o)
    }

    /// Bitwise comparison per output (f32 compared as bits, so NaN payloads and -0 count).
    fn diff(&self, o: &Outputs) -> Value {
        fn bits(a: &[f32], b: &[f32]) -> usize {
            a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count()
        }
        json!({
            "y": bits(&self.y, &o.y),
            "gemm_bf16": self.gy.iter().zip(&o.gy).filter(|(a, b)| a != b).count(),
            "post_f32": bits(&self.gf, &o.gf),
            "memset": bits(&self.z, &o.z),
            "coop": bits(&self.cb, &o.cb),
        })
    }

    fn bit_equal(&self, o: &Outputs) -> bool {
        let f = |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
        f(&self.y, &o.y) && self.gy == o.gy && f(&self.gf, &o.gf) && f(&self.z, &o.z) && f(&self.cb, &o.cb)
    }
}

struct Kernels {
    axpy: ink_cuda::Function,
    plus1: ink_cuda::Function,
    coop: ink_cuda::Function,
    coop_grid: u32,
}

/// The work that is run eagerly and captured: memcpy, struct-by-value kernel, GEMM, kernel, memset, cooperative.
fn body(
    s: &Stream,
    b: &Bufs,
    k: &Kernels,
    lt: &mut LtGemm,
    shape: GemmShape,
    mark: &dyn Fn(&str) -> ink_cuda::Result<()>,
) -> Result<()> {
    mark("axpy")?;
    s.dtod(&b.y, &b.y0)?;
    let p = SpikeParams {
        x: b.x.ptr(),
        y: b.y.ptr(),
        a: 0.5,
        n: N_VEC as i32,
        _pad: [0; 2],
        bias: [0, 0, 0, 3],
    };
    let mut args = KernelArgs::new();
    args.value(&p).uses(&b.x).uses(&b.y);
    // SAFETY: spike_axpy(SpikeParams) takes one 48-byte struct; it touches x[0..n) and y[0..n), both N_VEC long.
    // CONTRACT: spike/spike.cu spike_axpy.
    unsafe {
        s.launch(
            &k.axpy,
            LaunchCfg::linear((N_VEC as u32).div_ceil(BLOCK), BLOCK),
            &mut args,
        )
    }?;
    mark("gemm")?;
    lt.run(s, shape, &b.gx, GEMM_K, &b.gw, &b.gy, GEMM_N, GEMM_M)?;
    mark("post")?;
    let n = (GEMM_M * GEMM_N) as i32;
    let mut args = KernelArgs::new();
    args.buf(&b.gy).buf(&b.gf).value(&n);
    // SAFETY: spike_bf16_plus1(const uint16_t*, float*, int); reads gy[0..n), writes gf[0..n), both M*N long.
    // CONTRACT: spike/spike.cu spike_bf16_plus1.
    unsafe {
        s.launch(
            &k.plus1,
            LaunchCfg::linear((n as u32).div_ceil(BLOCK), BLOCK),
            &mut args,
        )
    }?;
    s.memset_zero(&b.z)?;
    mark("coop")?;
    let n = (k.coop_grid * BLOCK) as i32;
    let mut args = KernelArgs::new();
    args.buf(&b.ca).buf(&b.cb).value(&n);
    // SAFETY: spike_coop(float*, float*, int); touches a[0..n) and b[0..n), both grid*BLOCK long; the grid is at most
    // one block per SM, so it is co-resident.
    // CONTRACT: spike/spike.cu spike_coop.
    unsafe { s.launch_cooperative(&k.coop, LaunchCfg::linear(k.coop_grid, BLOCK), &mut args) }?;
    Ok(())
}

/// Clears every output; `z` is filled with ones so that a disabled memset node is visible.
fn reset(s: &Stream, b: &Bufs) -> Result<()> {
    for buf in [&b.y, &b.gf, &b.ca, &b.cb] {
        s.memset_zero(buf)?;
    }
    s.memset_zero(&b.gy)?;
    s.htod(&b.z, &vec![1.0f32; Z_LEN])?;
    s.synchronize()?;
    Ok(())
}

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
}

fn write_bf16(path: &Path, v: &[Bf16]) -> Result<()> {
    let bytes: Vec<u8> = v.iter().flat_map(|b| b.0.to_le_bytes()).collect();
    fs::write(path, bytes).with_context(|| format!("writing {}", path.display()))
}

struct Args {
    cubin: PathBuf,
    out: PathBuf,
    replays: usize,
}

fn parse_args() -> Result<Args> {
    let mut a = std::env::args().skip(1);
    let (mut cubin, mut out, mut replays) = (None, None, 2000);
    while let Some(flag) = a.next() {
        let mut val = || a.next().with_context(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--cubin" => cubin = Some(PathBuf::from(val()?)),
            "--out" => out = Some(PathBuf::from(val()?)),
            "--replays" => replays = val()?.parse()?,
            _ => bail!("usage: r1_spike --cubin FILE --out DIR [--replays N]"),
        }
    }
    Ok(Args {
        cubin: cubin.context("--cubin")?,
        out: out.context("--out")?,
        replays,
    })
}

fn main() -> Result<()> {
    let args = parse_args()?;
    fs::create_dir_all(&args.out)?;
    let mut r = Report {
        file: fs::File::create(args.out.join("spike.jsonl"))?,
        failed: Vec::new(),
    };
    r.item(
        "identity",
        true,
        json!({"build": ink_record::BUILD_LINE, "replays": args.replays}),
    )?;

    // R6 startup checks first, as an error instead of a later panic
    match startup_checks() {
        Ok(rep) => r.item("startup_checks", true, json!({"report": rep.to_string()}))?,
        Err(e) => {
            r.item("startup_checks", false, json!({"error": e.to_string()}))?;
            bail!("startup checks failed: {e}");
        }
    }
    let ctx = Context::new(0)?;
    let dev = ctx.device()?;
    r.item(
        "context",
        dev.compute_capability == (8, 0) && dev.cooperative_launch,
        json!({"name": dev.name, "uuid": dev.uuid, "cc": dev.compute_capability, "sms": dev.sm_count,
               "cooperative": dev.cooperative_launch, "mem_free_total": dev.mem_free_total}),
    )?;
    let s = ctx.new_stream()?;
    let mut arena = Arena::new(&ctx, &s, 64 << 20)?;
    let mm = (GEMM_M * GEMM_N) as usize;
    let coop_grid = u32::try_from(dev.sm_count)?;
    let b = Bufs {
        x: arena.take(N_VEC, "x")?,
        y0: arena.take(N_VEC, "y0")?,
        y: arena.take(N_VEC, "y")?,
        gx: arena.take((GEMM_M * GEMM_K) as usize, "gemm_x")?,
        gw: arena.take((GEMM_N * GEMM_K) as usize, "gemm_w")?,
        gy: arena.take(mm, "gemm_y")?,
        gf: arena.take(mm, "post")?,
        z: arena.take(Z_LEN, "memset")?,
        ca: arena.take((coop_grid * BLOCK) as usize, "coop_a")?,
        cb: arena.take((coop_grid * BLOCK) as usize, "coop_b")?,
    };
    let regions: Vec<Value> = arena
        .plan()
        .regions()
        .iter()
        .map(|g| json!([g.name, g.offset, g.bytes]))
        .collect();
    r.item(
        "arena",
        true,
        json!({"used": arena.plan().used(), "cap": arena.plan().cap(), "regions": regions}),
    )?;

    let module = Module::load(
        &ctx,
        &fs::read(&args.cubin).with_context(|| args.cubin.display().to_string())?,
    )?;
    let kpack = module.read_global("ink_kpack_id", &s)?;
    let kpack =
        String::from_utf8_lossy(&kpack[..kpack.iter().position(|&c| c == 0).unwrap_or(kpack.len())]).into_owned();
    let k = Kernels {
        axpy: module.function("spike_axpy")?,
        plus1: module.function("spike_bf16_plus1")?,
        coop: module.function("spike_coop")?,
        coop_grid,
    };
    let names = [k.axpy.name()?, k.plus1.name()?, k.coop.name()?];
    r.item(
        "module",
        kpack.starts_with("r1-spike-") && names == ["spike_axpy", "spike_bf16_plus1", "spike_coop"],
        json!({"ink_kpack_id": kpack, "cuFuncGetName": names, "regs": [k.axpy.num_regs()?, k.coop.num_regs()?]}),
    )?;

    // inputs
    let mut seed = 0x1234_5678_9abc_def0;
    let hx: Vec<f32> = (0..N_VEC).map(|_| lcg(&mut seed)).collect();
    let hy0: Vec<f32> = (0..N_VEC).map(|_| lcg(&mut seed)).collect();
    let hgx: Vec<Bf16> = (0..GEMM_M * GEMM_K).map(|_| Bf16::from_f32(lcg(&mut seed))).collect();
    let hgw: Vec<Bf16> = (0..GEMM_N * GEMM_K)
        .map(|_| Bf16::from_f32(lcg(&mut seed) * 0.05))
        .collect();
    s.htod(&b.x, &hx)?;
    s.htod(&b.y0, &hy0)?;
    s.htod(&b.gx, &hgx)?;
    s.htod(&b.gw, &hgw)?;

    // GEMM: select by the C++ rule, pin, write, read back, run from the pin
    let shape = GemmShape {
        n: GEMM_N,
        k: GEMM_K,
        ldw: GEMM_K,
        input: Dtype::Bf16,
        output: Dtype::Bf16,
        class: PlanClass::Decode,
    };
    let version = ctx.startup().cublaslt_version;
    let mut picker = LtGemm::new(&s, AlgoPins::new(version))?;
    let sel = picker.select(shape, 64)?;
    let mut pins = AlgoPins::new(version);
    pins.insert(shape, sel.pin)?;
    let pin_path = args.out.join("gemm-algos.json");
    fs::write(&pin_path, pins.to_json())?;
    let pins = AlgoPins::from_json(&fs::read_to_string(&pin_path)?)?;
    let mut lt = LtGemm::new(&s, pins)?;
    let rebuilt = lt.pinned_opaque(shape)?;
    r.item(
        "gemm_pin",
        true,
        json!({"pin": format!("{:?}", sel.pin), "candidate": sel.candidate, "of": sel.candidates,
               "opaque_equal_to_heuristic": rebuilt == sel.opaque,
               "heuristic_opaque": format!("{:x?}", sel.opaque), "rebuilt_opaque": format!("{:x?}", rebuilt)}),
    )?;

    // eager reference
    reset(&s, &b)?;
    body(&s, &b, &k, &mut lt, shape, &|_| Ok(()))?;
    s.synchronize()?;
    let eager = Outputs::read(&s, &b)?;
    let axpy_ok = (0..N_VEC).all(|i| {
        let h = 0.5f32.mul_add(hx[i], hy0[i]) + 3.0;
        (eager.y[i] - h).abs() <= 1e-6 * h.abs().max(1.0)
    });
    let plus1_ok = eager
        .gy
        .iter()
        .zip(&eager.gf)
        .all(|(g, f)| (g.to_f32() + 1.0).to_bits() == f.to_bits());
    let coop_n = eager.cb.len();
    let coop_ok = (0..coop_n).all(|i| eager.cb[i] == ((i + BLOCK as usize) % coop_n) as f32 * 0.5 + 1.0);
    let memset_ok = eager.z.iter().all(|&v| v == 0.0);
    let mut gemm_err = 0f64;
    for m in 0..GEMM_M as usize {
        for n in 0..GEMM_N as usize {
            let k0 = GEMM_K as usize;
            let dot: f64 = (0..k0)
                .map(|kk| f64::from(hgx[m * k0 + kk].to_f32()) * f64::from(hgw[n * k0 + kk].to_f32()))
                .sum();
            let got = f64::from(eager.gy[m * GEMM_N as usize + n].to_f32());
            gemm_err = gemm_err.max((got - dot).abs() / dot.abs().max(0.05));
        }
    }
    r.item(
        "eager",
        axpy_ok && plus1_ok && coop_ok && memset_ok && gemm_err < 1.0 / 64.0,
        json!({"axpy": axpy_ok, "bf16_plus1_exact": plus1_ok, "cooperative_grid_sync": coop_ok, "memset": memset_ok,
               "gemm_max_rel_err_vs_f64": gemm_err, "coop_grid": coop_grid}),
    )?;
    write_bf16(&args.out.join("gemm_x.bin"), &hgx)?;
    write_bf16(&args.out.join("gemm_w.bin"), &hgw)?;
    write_bf16(&args.out.join("gemm_y_rust.bin"), &eager.gy)?;
    fs::write(args.out.join("gemm_meta.txt"), format!("{GEMM_M} {GEMM_N} {GEMM_K}\n"))?;

    // capture
    reset(&s, &b)?;
    let cap = s.begin_capture()?;
    body(&s, &b, &k, &mut lt, shape, &|name| cap.mark(name))?;
    let mut graph = cap.end()?;
    let nodes: Vec<Value> = graph
        .nodes()
        .iter()
        .map(|n| json!({"kind": format!("{:?}", n.kind), "kernel": n.kernel, "dims": n.dims, "region": n.region}))
        .collect();
    let kinds = |kind: NodeKind| graph.nodes().iter().filter(|n| n.kind == kind).count();
    let kernels: Vec<&str> = graph.nodes().iter().filter_map(|n| n.kernel.as_deref()).collect();
    let gemm_nodes = graph
        .nodes()
        .iter()
        .filter(|n| n.region == "gemm" && n.kind == NodeKind::Kernel)
        .count();
    let capture_ok = kinds(NodeKind::Memcpy) == 1
        && kinds(NodeKind::Memset) == 1
        && ["spike_axpy", "spike_bf16_plus1", "spike_coop"]
            .iter()
            .all(|k| kernels.contains(k))
        && gemm_nodes >= 1
        && graph
            .nodes()
            .iter()
            .all(|n| n.kind != NodeKind::Kernel || n.kernel.is_some());
    r.item(
        "capture",
        capture_ok,
        json!({"nodes": nodes, "gemm_kernel_nodes": gemm_nodes}),
    )?;

    // replay == eager
    let replay = |g: &Graph| -> Result<Outputs> {
        reset(&s, &b)?;
        g.launch()?;
        s.synchronize()?;
        Outputs::read(&s, &b)
    };
    let out = replay(&graph)?;
    r.item(
        "replay_bit_identical",
        out.bit_equal(&eager),
        json!({"differing": eager.diff(&out)}),
    )?;

    // node disable: region "post" = bf16_plus1 kernel + memset node
    let disabled = graph.set_region("post", NodeState::Disabled)?;
    let out = replay(&graph)?;
    let post_skipped = out.gf.iter().all(|&v| v == 0.0) && out.z.iter().all(|&v| v == 1.0);
    let rest_equal = out.y == eager.y && out.gy == eager.gy && out.cb == eager.cb;
    r.item(
        "node_disable",
        disabled == 2 && post_skipped && rest_equal,
        json!({"disabled_nodes": disabled, "post_skipped": post_skipped, "others_bit_identical": rest_equal}),
    )?;
    graph.set_region("post", NodeState::Enabled)?;
    let out = replay(&graph)?;
    r.item(
        "node_reenable",
        out.bit_equal(&eager),
        json!({"differing": eager.diff(&out)}),
    )?;

    // many replays under NVML at 10 Hz, events around them; eager loop for comparison (indicative only)
    let sampler = Sampler::start(&dev.uuid, PERIOD_10HZ)?;
    let gpu = sampler.info().clone();
    std::thread::sleep(Duration::from_millis(300));
    let (e0, e1, e2) = (ctx.new_event()?, ctx.new_event()?, ctx.new_event()?);
    s.record(&e0)?;
    for _ in 0..args.replays {
        graph.launch()?;
    }
    s.record(&e1)?;
    for _ in 0..args.replays {
        body(&s, &b, &k, &mut lt, shape, &|_| Ok(()))?;
    }
    s.record(&e2)?;
    e2.synchronize()?;
    std::thread::sleep(Duration::from_millis(300));
    let samples = sampler.stop()?;
    let (graph_ms, eager_ms) = (e0.elapsed_ms(&e1)?, e1.elapsed_ms(&e2)?);
    let summary = summarize(&samples);
    r.item(
        "events_nvml",
        summary.is_some_and(|x| x.n >= 5),
        json!({"label": "indicative (spike, no clock control)", "graph_us_per_iter": graph_ms * 1e3 / args.replays as f32,
               "eager_us_per_iter": eager_ms * 1e3 / args.replays as f32, "gpu": format!("{gpu:?}"),
               "nvml": format!("{summary:?}"), "samples": samples.len()}),
    )?;
    let out = replay(&graph)?;
    r.item(
        "replay_after_load_bit_identical",
        out.bit_equal(&eager),
        json!({"differing": eager.diff(&out)}),
    )?;
    r.item(
        "drop_order",
        true,
        json!({"note": "graph, GEMM handles, module, arena, stream and context drop at the end of main; the context goes last because every object holds it"}),
    )?;

    let ok = r.failed.is_empty();
    r.item("summary", ok, json!({"failed": r.failed}))?;
    if !ok {
        bail!("spike items failed: {:?}", r.failed);
    }
    Ok(())
}
