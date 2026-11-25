//! Flags every tool takes to start an engine and to write its record, and the code behind them.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result, bail};
use clap::Args;
use ink_record::{Component, ComponentName, Context, DirtyPolicy, Engine, Gpu, Identity, RecordWriter, Source};
use ink_rt_api::{EngineConfig, Glue, Graphs, Runtime};
use serde_json::Value;

/// Rows per prefill segment (`EngineConfig::max_chunk`).
pub const MAX_CHUNK: u32 = 2048;

/// Configuration of the tools on the Rust runtime (`docs/RUST-PLAN.md` section 2).
pub const CONFIG_RUST: &str = "M4";

/// Starts the Rust CUDA engine (`ink_runtime::Engine`).
#[derive(Debug, Clone, Args)]
pub struct EngineArgs {
    /// Model directory.
    #[arg(long, default_value = "/model")]
    pub model: PathBuf,
    /// cuBLASLt pin file: loaded when it exists.
    #[arg(long)]
    pub gemm_algos: PathBuf,
    /// Sessions with state on the card.
    #[arg(long, default_value_t = 8)]
    pub slots: u32,
    /// M at which each dense GEMM shape's algorithm is timed at start-up (0: the heuristic's first choice).
    #[arg(long, default_value_t = 32)]
    pub tune_m: u32,
    /// Glue path: 1 per-op reference, 6 production.
    #[arg(long, default_value_t = 6)]
    pub glue: u8,
    /// Capture decode-only calls as CUDA graphs: 1 or 0.
    #[arg(long, default_value_t = 1)]
    pub graphs: u8,
    /// Device memory the captured graphs may hold, MiB.
    #[arg(long, default_value_t = ink_rt_api::DEFAULT_GRAPH_MEM >> 20)]
    pub graph_mem_mib: u64,
    /// Store global-layer K/V cache in FP8 E4M3 (lossy; halves global KV VRAM and bandwidth; SWA rings stay bf16).
    #[arg(long)]
    pub kv_fp8: bool,
}

impl EngineArgs {
    /// The glue path chosen by `--glue`.
    ///
    /// # Errors
    ///
    /// Fails for a value other than 1 or 6.
    pub fn glue_mode(&self) -> Result<Glue> {
        match self.glue {
            1 => Ok(Glue::Reference),
            6 => Ok(Glue::Production),
            other => bail!("--glue must be 1 or 6, got {other}"),
        }
    }

    /// The graph setting chosen by `--graphs`.
    ///
    /// # Errors
    ///
    /// Fails for a value other than 0 or 1.
    pub fn graph_mode(&self) -> Result<Graphs> {
        match self.graphs {
            0 => Ok(Graphs::Off),
            1 => Ok(Graphs::On),
            other => bail!("--graphs must be 0 or 1, got {other}"),
        }
    }

    /// Loads the library, creates the engine with `pages` KV pages and prefill chunks of up to `max_chunk` rows
    /// (`t_max` = `max_chunk` + 64, as the C++ tools) and applies the glue path.
    ///
    /// # Errors
    ///
    /// Fails when the library or a flag is refused or the engine cannot be created.
    pub fn open(&self, pages: u32, max_chunk: u32) -> Result<Box<dyn Runtime>> {
        let (glue, graphs) = (self.glue_mode()?, self.graph_mode()?);
        let config = EngineConfig {
            model_dir: self.model.clone(),
            gemm_algos: self.gemm_algos.clone(),
            max_slots: self.slots,
            num_pages: pages,
            t_max: max_chunk + 64,
            max_chunk,
            gemm_tune_m: self.tune_m,
            graphs,
            graph_mem: self.graph_mem_mib << 20,
            kv_fp8: self.kv_fp8,
            ..EngineConfig::default()
        };
        let mut rt = open_runtime(config)?;
        rt.set_glue(glue)?;
        Ok(rt)
    }

    /// The flags as JSON, for the record's context.
    #[must_use]
    pub fn json(&self, pages: u32, max_chunk: u32) -> Value {
        serde_json::json!({"runtime": "Rust", "model": self.model, "gemm_algos": self.gemm_algos,
            "slots": self.slots, "pages": pages, "max_chunk": max_chunk,
            "tune_m": self.tune_m, "glue": self.glue, "graphs": self.graphs, "graph_mem_mib": self.graph_mem_mib,
            "kv_fp8": self.kv_fp8})
    }
}

/// Creates the Rust CUDA runtime (`ink_runtime::Engine`) with `config`.
///
/// # Errors
///
/// Fails when the engine refuses the configuration or CUDA initialization fails.
pub fn open_runtime(config: EngineConfig) -> Result<Box<dyn Runtime>> {
    let rt = ink_runtime::Engine::new(&config)?;
    eprintln!("runtime: {:?} (graphs {:?})", rt.identity(), config.graphs);
    Ok(Box::new(rt))
}

/// Where the tool writes its record (AGENTS.md hard rule 7 and `docs/RUST-PLAN.md` section 6).
#[derive(Debug, Clone, Args)]
pub struct RecordArgs {
    /// Directory of the record (created when missing); the tool's `<tool>.jsonl` goes here.
    #[arg(long)]
    pub record_dir: PathBuf,
    /// Write a `.dirty.jsonl` record when a component is built from a dirty tree (debugging only).
    #[arg(long)]
    pub allow_dirty: bool,
    /// GPU index of the card, for the record.
    #[arg(long)]
    pub gpu: Option<u32>,
    /// Card UUID.
    #[arg(long, default_value = "")]
    pub uuid: String,
    /// vBIOS type of the card (`6D` or `67`).
    #[arg(long, default_value = "")]
    pub vbios: String,
    /// Power cap in watts.
    #[arg(long)]
    pub power_cap_w: Option<u32>,
}

impl RecordArgs {
    /// Creates `<record_dir>/<name>.jsonl` with the identity of this tool and of the runtime behind `rt`.
    ///
    /// # Errors
    ///
    /// Fails when `--uuid` or `--vbios` come without `--gpu`, the directory or file cannot be created, the file
    /// exists, or a component is dirty without `--allow-dirty`.
    pub fn writer(&self, name: &str, rt: &dyn Runtime, config: Value, baseline: &str) -> Result<RecordWriter> {
        self.writer_with(name, rt.identity(), [], config, baseline)
    }

    /// As [`RecordArgs::writer`], for the runtime identity `runtime` and the components `others` beside this tool
    /// (a drafter, or a runtime that is not a [`Runtime`] object).
    ///
    /// # Errors
    ///
    /// As [`RecordArgs::writer`].
    pub fn writer_with(
        &self,
        name: &str,
        runtime: Source,
        others: impl IntoIterator<Item = (ComponentName, Component)>,
        config: Value,
        baseline: &str,
    ) -> Result<RecordWriter> {
        let gpu = match self.gpu {
            Some(index) => Some(Gpu {
                index,
                uuid: self.uuid.clone(),
                vbios: self.vbios.clone(),
            }),
            None if self.uuid.is_empty() && self.vbios.is_empty() => None,
            None => bail!("--uuid and --vbios need --gpu"),
        };
        fs::create_dir_all(&self.record_dir).with_context(|| format!("creating {}", self.record_dir.display()))?;
        let mixed = CONFIG_RUST;
        let mut components: Vec<(ComponentName, Component)> =
            vec![(ComponentName::Tools, Component::Source(Source::this_build()))];
        components.extend(others);
        let identity = Identity::new(runtime, mixed, components);
        let context = Context {
            argv: std::env::args().collect(),
            config,
            gpu,
            power_cap_w: self.power_cap_w,
            baseline: baseline.to_owned(),
            label: name.to_owned(),
        };
        let policy = if self.allow_dirty {
            DirtyPolicy::Allow
        } else {
            DirtyPolicy::Refuse
        };
        Ok(RecordWriter::create(&self.record_dir, name, identity, context, policy)?)
    }

    /// Path of a file in the record directory.
    #[must_use]
    pub fn file(&self, name: &str) -> PathBuf {
        self.record_dir.join(name)
    }
}

/// The identity fields that open each output line of the C++ tools:
/// `"engine":"native","engine_version":"0.2.0","commit":"<40 hex>","dirty":false` (no braces).
#[must_use]
pub fn identity_fields(source: &Source) -> String {
    format!(
        r#""engine":"{}","engine_version":"{}","commit":"{}","dirty":{}"#,
        engine_name(source),
        source.version,
        source.commit,
        source.dirty
    )
}

/// `native` or `rust`: the `engine` field of the C++ tools' output lines.
#[must_use]
pub fn engine_name(source: &Source) -> &'static str {
    match source.implementation {
        Engine::Native => "native",
        Engine::Rust => "rust",
    }
}

/// Fails unless `path` is an existing regular file.
///
/// # Errors
///
/// Fails when `path` does not exist or is not a file.
pub fn require_file(path: &Path, what: &str) -> Result<()> {
    if !path.is_file() {
        bail!("{what} {} is not a file", path.display());
    }
    Ok(())
}
