// Ported from engine/server/tools/ink_server.cpp @ 4171493.
//! The HTTP server: the OpenAI front end of `ink-api` over a back end, bound to loopback.
//!
//! ```text
//! ink-server --model-dir DIR [--host 127.0.0.1] [--port 8407] [--served-model-name inkling-2-mini]
//!            [--max-model-len 262144] [--backend mock|fake|engine] [--mock-delay-ms 20] [--mock-tool-call NAME=ARGS]
//!            [--enable-prompt-tokens-details] [--system-fingerprint S]
//!            [--max-running N] [--session-cap TOKENS] [--kv-pool-tokens TOKENS]
//!            [--gemm-algos FILE [--slots 8] [--kv-pages 256] [--graph-mem-mib 512]
//!             [--draft-fold fold.bin [--draft-ctx 65536] [--spec-trace FILE]]
//!             [--dflash2-dir DIR]] [--step-log FILE]
//! ```
//!
//! `--backend mock` answers every request with a short scripted reasoning block and text block (HTTP and client
//! smoke tests without a GPU), three tokens per step; with `--mock-tool-call NAME=ARGS` (ARGS a JSON object) a prompt
//! that holds no tool result is answered with a call of that tool instead of the text block, so a client completes
//! one tool round trip. `--backend fake` runs the scheduler of `ink-sched` over its `FakeRuntime`: deterministic hash
//! tokens on a virtual clock, no model. `--backend engine` runs the scheduler over the Rust CUDA runtime
//! (`ink-runtime::Engine` through `RuntimeAdapter`, configuration `M4`), built on the scheduler's loop thread. With
//! `--draft-fold` (DSpark) or `--dflash2-dir` (DFlash2) the adapter runs speculative decoding for requests without
//! logprobs (greedy accept, or device-sampled accept for sampled requests); requests with logprobs decode one token per
//! step.
//! Only loopback addresses are accepted (AGENTS hard rule 5). When the back end stops serving (a device failure of the
//! runtime, or a panic on the scheduler's loop: `/health` answers 503 from then on) the server stops taking
//! connections, gives open responses [`SHUTDOWN_GRACE`] to finish (the back end has failed their requests) and exits
//! with status 1.
#![forbid(unsafe_code)]

use std::convert::Infallible;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use ink_api::http::{bind_loopback, router};
use ink_api::{Frontend, Hooks, ServerConfig};
use ink_backend::{Backend, MockBackend, Script};
use ink_record::{Component, ComponentName, Identity, Source};
use ink_rt_api::{
    CallRows, DraftConfig, EngineConfig, Graphs, HeadPrecision, KvPrecision, Runtime, RuntimeError, VerifyWidth,
};
use ink_runtime::RuntimeAdapter;
use ink_sched::{Config, EngineBackend, FakeRuntime};
use ink_text::{Specials, Tokenizer};
use serde_json::json;
use tokio::signal::unix::{SignalKind, signal};
use tokio::sync::Notify;
use tracing::Level;

/// How long open streams get to finish after a termination signal.
const SHUTDOWN_GRACE: Duration = Duration::from_secs(10);

/// How often the server checks that the back end still serves.
const HEALTH_POLL: Duration = Duration::from_millis(100);

/// Tokens per step of the mock back end.
const MOCK_GROUP: usize = 3;

/// `--verify-width`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum VerifyArg {
    /// Probability-gated width.
    Gated,
    /// Always 8 rows.
    Full,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum BackendKind {
    /// Scripted replies.
    Mock,
    /// The scheduler over `FakeRuntime` (no model).
    Fake,
    /// The scheduler over the Rust CUDA runtime (`ink_runtime::Engine`).
    Engine,
}

impl BackendKind {
    fn name(self) -> &'static str {
        match self {
            Self::Mock => "mock",
            Self::Fake => "fake",
            Self::Engine => "engine",
        }
    }
}

#[derive(Parser)]
#[command(name = "ink-server", version = ink_record::BUILD_LINE, about = "OpenAI-compatible HTTP server, loopback only")]
struct Cli {
    /// Directory with the model's `tokenizer.json` and `special_tokens_map.json`.
    #[arg(long)]
    model_dir: PathBuf,
    /// Address to bind; only loopback addresses are accepted.
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    /// Port to bind; 0 picks a free one.
    #[arg(long, default_value_t = 8407)]
    port: u16,
    /// `model` in responses; a request naming another model gets HTTP 404.
    #[arg(long, default_value = "inkling-2-mini")]
    served_model_name: String,
    /// Prompt plus output token limit.
    #[arg(long, default_value_t = 262_144)]
    max_model_len: u32,
    #[arg(long, value_enum, default_value = "mock")]
    backend: BackendKind,
    /// Delay of every step of the mock back end.
    #[arg(long, default_value_t = 20)]
    mock_delay_ms: u64,
    /// Mock back end: answer a prompt without a tool result with a call of tool NAME with arguments ARGS (JSON object).
    #[arg(long, value_name = "NAME=ARGS", value_parser = parse_tool_call)]
    mock_tool_call: Option<(String, String)>,
    /// Report cached prompt tokens in `usage.prompt_tokens_details` (config B: off).
    #[arg(long)]
    enable_prompt_tokens_details: bool,
    #[arg(long, default_value = "ink-srv")]
    system_fingerprint: String,
    /// Concurrent generating sessions of the fake and engine back ends.
    #[arg(long)]
    max_running: Option<u32>,
    /// Prefill cost the scheduler adds to a step while sessions decode, in ms (`ink_sched::Config::stall_budget_ms`,
    /// the interleave rule's budget chunk).
    #[arg(long)]
    stall_budget_ms: Option<f64>,
    /// Largest prefill chunk in rows (`ink_sched::Config::max_chunk`, capped at the runtime's segment limit).
    #[arg(long)]
    prefill_chunk: Option<u32>,
    /// Per-session cap (prompt plus output) of the fake and engine back ends.
    #[arg(long)]
    session_cap: Option<u64>,
    /// KV pool size of the fake and engine back ends, in tokens (the engine's own pages bound it).
    #[arg(long)]
    kv_pool_tokens: Option<u64>,
    /// Engine back end: the cuBLASLt pin file, loaded when it exists, written after start-up otherwise.
    #[arg(long)]
    gemm_algos: Option<PathBuf>,
    /// Engine back end: runtime slots (sessions with state on the card).
    #[arg(long, default_value_t = 8)]
    slots: u32,
    /// Engine back end: KV pages of 1,024 tokens on the card.
    #[arg(long, default_value_t = 256)]
    kv_pages: u32,
    /// Engine back end: store global-layer K/V cache in FP8 E4M3 (lossy; halves global KV VRAM and bandwidth; SWA rings stay bf16).
    #[arg(long)]
    kv_fp8: bool,
    /// Engine back end: device memory the captured CUDA graphs may hold, MiB; the engine refuses to
    /// start when the KV pages and the drafter leave less than this (plus a fixed headroom) free.
    #[arg(long, default_value_t = ink_rt_api::DEFAULT_GRAPH_MEM >> 20)]
    graph_mem_mib: u64,
    /// Engine back end: folded DSpark drafter weights; speculative decoding (greedy and sampled requests) when set.
    #[arg(long)]
    draft_fold: Option<PathBuf>,
    /// Engine back end: DFlash2 checkpoint directory (`model.safetensors`); speculative decoding
    /// with DFlash2 instead of DSpark when set.
    #[arg(long)]
    draft_dflash2: Option<PathBuf>,
    /// Engine back end: drafter positions per slot (a session speculates while its rows stay below them;
    /// defaults to 262,144 for `--draft-dflash2` and 65,536 for `--draft-fold`).
    #[arg(long)]
    draft_ctx: Option<u32>,
    /// Engine back end: verify width of a speculating session: `gated` (1 + the leading drafts whose cumulative drafter
    /// probability stays >= 0.7, rounded up to 1 / 2 / 4 / 8 rows) or `full` (always 8 rows, no probability pass).
    #[arg(long, value_enum, default_value_t = VerifyArg::Gated)]
    verify_width: VerifyArg,
    /// Engine back end, gated width: the verify call's rows per session `m` it is rounded up to (`S m` rows over `S`
    /// sessions, the widest sessions widened), a comma list holding 8, or `ragged` (no rounding).
    #[arg(long, default_value = ink_rt_api::VERIFY_CALL_ROWS)]
    verify_call_rows: String,
    /// Engine back end: verbatim n-gram lookup drafts appended after the drafter's 7 drafts (`0` = off, `8` = up to 8
    /// copied tokens, 16-row single-pass verify; overridable via `INK_LOOKUP`).
    #[arg(long, default_value_t = 8)]
    lookup: u32,
    /// Engine back end: minimum backward verbatim match length `m` required to trigger lookup drafts (overridable via
    /// `INK_LOOKUP_MIN`).
    #[arg(long, default_value_t = 8)]
    lookup_min: u32,
    /// Engine back end: write one JSON line per speculative round (slots, sequences, positions, drafts, accepts).
    #[arg(long)]
    spec_trace: Option<PathBuf>,
    /// Write one JSON line per step and phase (scheduler, runtime) with host timings to this file.
    #[arg(long)]
    step_log: Option<PathBuf>,
}

/// `NAME=ARGS` of `--mock-tool-call`: a tool name and a JSON object, written back as compact JSON.
fn parse_tool_call(s: &str) -> Result<(String, String), String> {
    let (name, args) = s.split_once('=').ok_or("expected NAME=ARGS")?;
    let args: serde_json::Value = serde_json::from_str(args).map_err(|e| format!("ARGS is not JSON: {e}"))?;
    if name.is_empty() || !args.is_object() {
        return Err("NAME must be non-empty and ARGS a JSON object".to_owned());
    }
    Ok((name.to_owned(), args.to_string()))
}

/// The scripted reply of the mock: a reasoning block and a text block naming the prompt's length, or (with
/// `tool_call`, while the prompt holds no tool result) a reasoning block and a call of that tool.
fn mock_backend(tok: &Arc<Tokenizer>, delay: Duration, tool_call: Option<&(String, String)>) -> Result<MockBackend> {
    let id = |name: &str| -> Result<i32> {
        let id = tok
            .token_id(name)
            .with_context(|| format!("the tokenizer has no {name}"))?;
        Ok(i32::try_from(id)?)
    };
    let (model, thinking, text) = (
        id("<|message_model|>")?,
        id("<|content_thinking|>")?,
        id("<|content_text|>")?,
    );
    let (end, eos) = (id("<|end_message|>")?, id("<|content_model_end_sampling|>")?);
    let (tool_result, invoke) = (id("<|message_tool|>")?, id("<|content_invoke_tool_json|>")?);
    let encode = |s: &str| -> Result<Vec<i32>> {
        tok.encode(s, Specials::Recognize)?
            .into_iter()
            .map(|t| Ok(i32::try_from(t)?))
            .collect()
    };
    let reasoning = encode("Mock back end: no model attached.")?;
    let call = match tool_call {
        Some((name, args)) => {
            let mut ids = vec![model];
            ids.extend(encode(name)?);
            ids.push(invoke);
            ids.extend(encode(&format!(
                r#"{{"name":{},"args":{args}}}"#,
                serde_json::Value::from(name.as_str())
            ))?);
            ids.push(end);
            Some(ids)
        }
        None => None,
    };
    let mock = MockBackend::new();
    let tok = Arc::clone(tok);
    mock.set_generator(move |req| {
        let answer = tok
            .encode(&format!("Prompt had {} tokens.", req.prompt.len()), Specials::Recognize)
            .map(|ids| {
                ids.into_iter()
                    .filter_map(|t| i32::try_from(t).ok())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let mut all = vec![model, thinking];
        all.extend(&reasoning);
        all.push(end);
        match &call {
            Some(call) if !req.prompt.contains(&tool_result) => all.extend(call),
            _ => {
                all.extend([model, text]);
                all.extend(answer);
                all.push(end);
            }
        }
        all.push(eos);
        Script {
            groups: all.chunks(MOCK_GROUP).map(<[i32]>::to_vec).collect(),
            step_delay: delay,
            first_delay: delay,
            ..Script::default()
        }
    });
    Ok(mock)
}

/// Default prefill chunk for `--backend engine`: 3,072 rows fits in 64 GiB with 504 pages + 8 slots (`3.7 GiB` free),
/// cuts M1 prefill wall time and TTFT p90/p99 (`bench/raw/2026-11-26-prefill-opt/`), and `--prefill-chunk 2048` lowers it.
const DEFAULT_ENGINE_PREFILL_CHUNK: u32 = 3072;

/// The scheduler configuration of the fake and engine back ends.
fn sched_config(cli: &Cli) -> Config {
    let mut cfg = Config::default();
    if let Some(n) = cli.max_running {
        cfg.max_running = n;
    }
    if let Some(ms) = cli.stall_budget_ms {
        cfg.stall_budget_ms = ms;
    }
    if matches!(cli.backend, BackendKind::Engine) {
        let rows = cli.prefill_chunk.unwrap_or(DEFAULT_ENGINE_PREFILL_CHUNK);
        cfg.max_chunk = rows;
        cfg.max_prefill_rows = cfg.max_prefill_rows.max(rows);
        cfg.t_max = cfg.t_max.max(rows + 64);
        cfg.cost_prefill_tok_ms = 0.140;
        cfg.cost_attn_ms_per_tok_kkey = 0.00195;
    } else if let Some(rows) = cli.prefill_chunk {
        cfg.max_chunk = rows;
        cfg.max_prefill_rows = cfg.max_prefill_rows.max(rows);
        cfg.t_max = cfg.t_max.max(rows + 64);
    }
    if let Some(n) = cli.session_cap {
        cfg.session_cap_tokens = n;
    }
    if let Some(n) = cli.kv_pool_tokens {
        cfg.pool_tokens = n;
    }
    cfg
}

fn fake_backend(cli: &Cli) -> Result<EngineBackend> {
    Ok(EngineBackend::spawn(sched_config(cli), || {
        Ok::<_, Infallible>(FakeRuntime::default())
    })?)
}

/// Why the engine back end's runtime could not be built.
#[derive(Debug)]
enum StartError {
    /// Drafter or its buffers.
    Drafter(RuntimeError),
    /// The Rust engine.
    Engine(RuntimeError),
}

impl std::fmt::Display for StartError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Drafter(e) => write!(f, "drafter: {e}"),
            Self::Engine(e) => write!(f, "engine: {e}"),
        }
    }
}

impl std::error::Error for StartError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Drafter(e) | Self::Engine(e) => Some(e),
        }
    }
}

impl From<RuntimeError> for StartError {
    fn from(e: RuntimeError) -> Self {
        Self::Drafter(e)
    }
}

/// The scheduler over the Rust CUDA runtime: the engine is created on the scheduler's loop thread (bound to that
/// thread). Returns the back end and the runtime's identity.
fn engine_backend(cli: &Cli) -> Result<(EngineBackend, Source)> {
    let gemm_algos = cli
        .gemm_algos
        .clone()
        .context("--backend engine needs --gemm-algos (pinned cuBLASLt algorithms)")?;
    let default_cfg = EngineConfig::default();
    let max_chunk = cli.prefill_chunk.unwrap_or(DEFAULT_ENGINE_PREFILL_CHUNK);
    let config = EngineConfig {
        model_dir: cli.model_dir.clone(),
        gemm_algos,
        max_slots: cli.slots,
        num_pages: cli.kv_pages,
        t_max: max_chunk + 64,
        max_chunk,
        graphs: Graphs::On,
        graph_mem: cli.graph_mem_mib << 20,
        kv_fp8: cli.kv_fp8,
        ..default_cfg
    };
    let call_rows = CallRows::parse(&cli.verify_call_rows).map_err(|e| anyhow::anyhow!("{e}"))?;
    let verify = match cli.verify_width {
        VerifyArg::Gated => VerifyWidth::Gated(call_rows),
        VerifyArg::Full => VerifyWidth::Full,
    };
    let lookup = std::env::var("INK_LOOKUP")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(cli.lookup);
    let lookup_min = std::env::var("INK_LOOKUP_MIN")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(cli.lookup_min);
    let draft = if let Some(dflash2_dir) = cli.draft_dflash2.clone() {
        Some(DraftConfig {
            model_dir: cli.model_dir.clone(),
            fold_bin: PathBuf::new(),
            dflash2_dir: Some(dflash2_dir),
            sessions: cli.slots,
            max_ctx: cli.draft_ctx.unwrap_or(262_144),
            head: HeadPrecision::Int8,
            global_kv: KvPrecision::Bf16,
            verify,
            lookup,
            lookup_min,
        })
    } else {
        cli.draft_fold.clone().map(|fold_bin| DraftConfig {
            model_dir: cli.model_dir.clone(),
            fold_bin,
            dflash2_dir: None,
            sessions: cli.slots,
            max_ctx: cli.draft_ctx.unwrap_or(65_536),
            head: HeadPrecision::Int8,
            global_kv: KvPrecision::Bf16,
            verify,
            lookup,
            lookup_min,
        })
    };
    let trace = cli
        .spec_trace
        .as_ref()
        .map(|p| File::create(p).with_context(|| format!("creating {}", p.display())))
        .transpose()?;
    let (id_tx, id_rx) = std::sync::mpsc::sync_channel(1);
    let backend = EngineBackend::spawn(sched_config(cli), move || {
        let rt = ink_runtime::Engine::new(&config).map_err(StartError::Engine)?;
        tracing::info!(runtime = ?rt.identity(), "Rust engine ready");
        let _ = id_tx.send(rt.identity());
        let rt: Box<dyn Runtime> = Box::new(rt);
        let adapter = match &draft {
            Some(d) => RuntimeAdapter::with_drafter(rt, &config, d, trace)?,
            None => RuntimeAdapter::new(rt, &config),
        };
        tracing::info!(drafter = ?draft, "runtime adapter ready");
        Ok::<_, StartError>(adapter)
    })?;
    let identity = id_rx.recv().context("the runtime library reported no identity")?;
    Ok((backend, identity))
}

/// The body of `/version`: the build line and the identity of the process (`ink-record`). `runtime` is the model
/// runtime's identity; `None` without one (mock and fake back ends: the entry is then this build).
fn version_body(kind: BackendKind, runtime: Option<Source>) -> Result<String> {
    let this = Source::this_build();
    let mut others = vec![(ComponentName::Frontend, Component::Source(this.clone()))];
    if !matches!(kind, BackendKind::Mock) {
        others.push((ComponentName::Scheduler, Component::Source(this.clone())));
    }
    // M1: Rust front end and scheduler without a model runtime (the runtime entry is this build); M3: Rust front end
    // and scheduler over the C++ runtime through ink_rt.h
    let (runtime, config) = match runtime {
        Some(rt) => (rt, "M3"),
        None => (this, "M1"),
    };
    let identity = Identity::new(runtime, config, others);
    identity.validate()?;
    Ok(json!({"version": ink_record::BUILD_LINE, "backend": kind.name(), "identity": identity}).to_string())
}

/// Logs at INFO to stderr; with `step_log`, the per-step timing events (target `ink_step`, DEBUG) as JSON lines to
/// that file.
fn init_logging(step_log: Option<&Path>) -> Result<()> {
    use tracing_subscriber::filter::{LevelFilter, Targets};
    use tracing_subscriber::prelude::*;
    let stderr = tracing_subscriber::fmt::layer()
        .with_ansi(false)
        .with_writer(std::io::stderr)
        .with_filter(LevelFilter::INFO);
    let steps = match step_log {
        Some(path) => {
            let file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
            Some(
                tracing_subscriber::fmt::layer()
                    .json()
                    .with_writer(Mutex::new(file))
                    .with_filter(Targets::new().with_target("ink_step", Level::DEBUG)),
            )
        }
        None => None,
    };
    tracing_subscriber::registry().with(stderr).with(steps).init();
    Ok(())
}

/// Why the server stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stop {
    /// SIGINT or SIGTERM.
    Signal,
    /// The back end stopped serving.
    BackendDown,
}

/// Resolves once `backend` stops serving.
async fn backend_down(backend: &dyn Backend) {
    while backend.healthy() {
        tokio::time::sleep(HEALTH_POLL).await;
    }
}

/// Resolves on SIGINT or SIGTERM.
async fn termination() -> Result<()> {
    let mut term = signal(SignalKind::terminate())?;
    let mut int = signal(SignalKind::interrupt())?;
    tokio::select! {
        _ = term.recv() => {}
        _ = int.recv() => {}
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    init_logging(cli.step_log.as_deref())?;
    tracing::info!("ink-server {}", ink_record::BUILD_LINE);
    let listener = bind_loopback(&cli.host, cli.port).await?;
    let addr = listener.local_addr()?;
    let tok = Arc::new(Tokenizer::load(&cli.model_dir).context("loading the tokenizer")?);
    let (backend, vocab_size, runtime): (Arc<dyn Backend>, _, _) = match cli.backend {
        BackendKind::Mock => {
            let mock = mock_backend(
                &tok,
                Duration::from_millis(cli.mock_delay_ms),
                cli.mock_tool_call.as_ref(),
            )?;
            (Arc::new(mock), None, None)
        }
        BackendKind::Fake => {
            let fake = fake_backend(&cli)?;
            tracing::info!(caps = ?fake.caps(), "scheduler running over FakeRuntime");
            let vocab = fake.caps().vocab;
            (Arc::new(fake), Some(vocab), None)
        }
        BackendKind::Engine => {
            let (engine, identity) = engine_backend(&cli)?;
            tracing::info!(caps = ?engine.caps(), "scheduler running over the Rust engine runtime");
            let vocab = engine.caps().vocab;
            (Arc::new(engine), Some(vocab), Some(identity))
        }
    };
    let cfg = ServerConfig {
        served_model_name: cli.served_model_name.clone(),
        model_dir: cli.model_dir.clone(),
        max_model_len: cli.max_model_len,
        prompt_tokens_details: cli.enable_prompt_tokens_details,
        system_fingerprint: cli.system_fingerprint.clone(),
        vocab_size,
        ..ServerConfig::default()
    };
    let fe = Frontend::new(tok, cfg, Arc::clone(&backend), Hooks::default())?;
    let app = router(Arc::new(fe), version_body(cli.backend, runtime)?);
    tracing::info!(
        "{} listening on http://{addr} (backend {})",
        cli.served_model_name,
        cli.backend.name()
    );

    let stop = Arc::new(Notify::new());
    let server = axum::serve(listener, app).with_graceful_shutdown({
        let stop = Arc::clone(&stop);
        async move { stop.notified().await }
    });
    let cause = std::sync::OnceLock::new();
    let drain = async {
        let why = tokio::select! {
            t = termination() => { t?; Stop::Signal }
            () = backend_down(backend.as_ref()) => Stop::BackendDown,
        };
        cause.get_or_init(|| why);
        match why {
            Stop::Signal => tracing::info!("terminating: waiting up to {SHUTDOWN_GRACE:?} for open streams"),
            Stop::BackendDown => tracing::error!(
                "the back end stopped serving: closing, waiting up to {SHUTDOWN_GRACE:?} for open responses, then exit 1"
            ),
        }
        stop.notify_one();
        tokio::time::sleep(SHUTDOWN_GRACE).await;
        tracing::warn!("open streams cut");
        Ok::<_, anyhow::Error>(())
    };
    tokio::select! {
        served = server => served.context("serving")?,
        drained = drain => drained?,
    }
    if cause.get() == Some(&Stop::BackendDown) {
        anyhow::bail!("the back end stopped serving (device failure or scheduler panic: see the log above)");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::parse_tool_call;

    #[test]
    fn mock_tool_call_flag_takes_a_name_and_a_json_object() {
        assert_eq!(
            parse_tool_call(r#"bash={"command": "echo hi"}"#),
            Ok(("bash".to_owned(), r#"{"command":"echo hi"}"#.to_owned()))
        );
        for bad in ["bash", r#"={"a":1}"#, "bash=[1]", "bash={", "bash=5"] {
            assert!(parse_tool_call(bad).is_err(), "{bad}");
        }
    }
}
