//! NVML sampler: power, SM clock, the software power-cap flag and the energy counter, at a fixed period (10 Hz in
//! the benchmarks), on its own thread (NVML is thread-safe and needs no CUDA context).
//!
//! Over `nvml-wrapper` (loads `libnvidia-ml.so.1` at [`Sampler::start`]; no `unsafe` here).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use nvml_wrapper::Nvml;
use nvml_wrapper::bitmasks::device::ThrottleReasons;
use nvml_wrapper::enum_wrappers::device::{Clock, TemperatureSensor};
use nvml_wrapper::error::NvmlError;

use crate::error::{CudaError, Result};

/// The benchmark sampling period (10 Hz).
pub const PERIOD_10HZ: Duration = Duration::from_millis(100);

/// One reading.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sample {
    /// Milliseconds since [`Sampler::start`].
    pub t_ms: f64,
    /// Board power in milliwatts.
    pub power_mw: u32,
    /// SM clock in MHz.
    pub sm_mhz: u32,
    /// Whether the clocks are held down by the software power cap at this reading.
    pub sw_power_cap: bool,
    /// Energy counter since driver load, in millijoules.
    pub energy_mj: u64,
    /// Memory clock in MHz.
    pub mem_mhz: u32,
    /// GPU temperature in degrees Celsius.
    pub temp_c: u32,
    /// NVML clock-event reason bits (`nvmlClocksEventReason*`: 0x4 software power cap, 0x8 hardware slowdown,
    /// 0x60 thermal, 0x80 power brake).
    pub reasons: u64,
    /// When the reading started (the clock the benchmarks time their windows with).
    pub at: Instant,
}

/// Static facts about the sampled GPU.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuInfo {
    /// Product name.
    pub name: String,
    /// UUID (`GPU-...`).
    pub uuid: String,
    /// vBIOS version string (records name the vBIOS type, AGENTS rule 7).
    pub vbios: String,
    /// Enforced power limit in milliwatts.
    pub power_limit_mw: u32,
    /// Driver version string.
    pub driver: String,
}

/// A running sampler; [`Sampler::stop`] returns the readings.
#[derive(Debug)]
pub struct Sampler {
    info: GpuInfo,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<Result<Vec<Sample>>>>,
}

fn nvml(call: &'static str) -> impl FnOnce(NvmlError) -> CudaError {
    move |e| CudaError::Nvml {
        call,
        message: e.to_string(),
    }
}

fn gpu_info(d: &nvml_wrapper::Device<'_>, lib: &Nvml) -> Result<GpuInfo> {
    Ok(GpuInfo {
        name: d.name().map_err(nvml("name"))?,
        uuid: d.uuid().map_err(nvml("uuid"))?,
        vbios: d.vbios_version().map_err(nvml("vbios_version"))?,
        power_limit_mw: d.enforced_power_limit().map_err(nvml("enforced_power_limit"))?,
        driver: lib.sys_driver_version().map_err(nvml("sys_driver_version"))?,
    })
}

impl Sampler {
    /// Initialises NVML, finds the GPU `uuid` (as [`crate::DeviceInfo::uuid`] prints it) and starts sampling every
    /// `period`.
    ///
    /// # Errors
    ///
    /// [`CudaError::Nvml`] when NVML cannot load, the GPU is not found, or a first reading fails.
    pub fn start(uuid: &str, period: Duration) -> Result<Sampler> {
        let lib = Arc::new(Nvml::init().map_err(nvml("init"))?);
        let info = gpu_info(&lib.device_by_uuid(uuid).map_err(nvml("device_by_uuid"))?, &lib)?;
        Self::spawn(lib, info, period)
    }

    /// As [`Sampler::start`] for the GPU with NVML index `index` (a container that is given one card sees it as 0).
    ///
    /// # Errors
    ///
    /// [`CudaError::Nvml`] when NVML cannot load, the index does not exist, or a first reading fails.
    pub fn start_index(index: u32, period: Duration) -> Result<Sampler> {
        let lib = Arc::new(Nvml::init().map_err(nvml("init"))?);
        let info = gpu_info(&lib.device_by_index(index).map_err(nvml("device_by_index"))?, &lib)?;
        Self::spawn(lib, info, period)
    }

    fn spawn(lib: Arc<Nvml>, info: GpuInfo, period: Duration) -> Result<Sampler> {
        let stop = Arc::new(AtomicBool::new(false));
        let thread = {
            let (stop, uuid) = (stop.clone(), info.uuid.clone());
            std::thread::Builder::new()
                .name("nvml-sampler".into())
                .spawn(move || sample_loop(&lib, &uuid, period, &stop))
                .map_err(|e| CudaError::Nvml {
                    call: "spawn",
                    message: e.to_string(),
                })?
        };
        Ok(Sampler {
            info,
            stop,
            thread: Some(thread),
        })
    }

    /// The GPU being sampled.
    pub fn info(&self) -> &GpuInfo {
        &self.info
    }

    /// Stops sampling and returns every reading.
    ///
    /// # Errors
    ///
    /// The first NVML error of a reading.
    pub fn stop(mut self) -> Result<Vec<Sample>> {
        self.stop.store(true, Ordering::Relaxed);
        match self.thread.take().map(JoinHandle::join) {
            Some(Ok(r)) => r,
            Some(Err(_)) => Err(CudaError::Nvml {
                call: "join",
                message: "sampler thread panicked".into(),
            }),
            None => Ok(Vec::new()),
        }
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

fn sample_loop(lib: &Nvml, uuid: &str, period: Duration, stop: &AtomicBool) -> Result<Vec<Sample>> {
    let d = lib.device_by_uuid(uuid).map_err(nvml("device_by_uuid"))?;
    let t0 = Instant::now();
    let mut next = t0;
    let mut out = Vec::new();
    while !stop.load(Ordering::Relaxed) {
        let at = Instant::now();
        let t_ms = (at - t0).as_secs_f64() * 1e3;
        let reasons = d.current_throttle_reasons().map_err(nvml("current_throttle_reasons"))?;
        out.push(Sample {
            t_ms,
            power_mw: d.power_usage().map_err(nvml("power_usage"))?,
            sm_mhz: d.clock_info(Clock::SM).map_err(nvml("clock_info"))?,
            sw_power_cap: reasons.contains(ThrottleReasons::SW_POWER_CAP),
            energy_mj: d.total_energy_consumption().map_err(nvml("total_energy_consumption"))?,
            mem_mhz: d.clock_info(Clock::Memory).map_err(nvml("clock_info"))?,
            temp_c: d.temperature(TemperatureSensor::Gpu).map_err(nvml("temperature"))?,
            reasons: reasons.bits(),
            at,
        });
        next += period;
        let now = Instant::now();
        if next > now {
            std::thread::sleep(next - now);
        } else {
            next = now; // fell behind (slow NVML call): do not burst to catch up
        }
    }
    Ok(out)
}

/// Aggregates of a sample window.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Summary {
    /// Number of readings.
    pub n: usize,
    /// Window length (first to last reading) in seconds.
    pub seconds: f64,
    /// Mean board power in watts.
    pub mean_power_w: f64,
    /// Highest board power in watts.
    pub max_power_w: f64,
    /// Mean SM clock in MHz.
    pub mean_sm_mhz: f64,
    /// Lowest SM clock in MHz.
    pub min_sm_mhz: u32,
    /// Fraction of readings with the software power-cap flag set.
    pub cap_fraction: f64,
    /// Energy counter difference (last minus first) in joules.
    pub energy_j: f64,
}

/// Summarises `samples`; `None` when empty.
pub fn summarize(samples: &[Sample]) -> Option<Summary> {
    let (first, last) = (samples.first()?, samples.last()?);
    let n = samples.len();
    let nf = n as f64;
    Some(Summary {
        n,
        seconds: (last.t_ms - first.t_ms) / 1e3,
        mean_power_w: samples.iter().map(|s| f64::from(s.power_mw)).sum::<f64>() / nf / 1e3,
        max_power_w: f64::from(samples.iter().map(|s| s.power_mw).max().unwrap_or(0)) / 1e3,
        mean_sm_mhz: samples.iter().map(|s| f64::from(s.sm_mhz)).sum::<f64>() / nf,
        min_sm_mhz: samples.iter().map(|s| s.sm_mhz).min().unwrap_or(0),
        cap_fraction: samples.iter().filter(|s| s.sw_power_cap).count() as f64 / nf,
        energy_j: last.energy_mj.saturating_sub(first.energy_mj) as f64 / 1e3,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(t_ms: f64, power_mw: u32, sm_mhz: u32, cap: bool, energy_mj: u64) -> Sample {
        Sample {
            t_ms,
            power_mw,
            sm_mhz,
            sw_power_cap: cap,
            energy_mj,
            mem_mhz: 0,
            temp_c: 0,
            reasons: 0,
            at: Instant::now(),
        }
    }

    #[test]
    fn summary_of_a_window() {
        let w = [
            s(0.0, 140_000, 1410, true, 1_000),
            s(100.0, 120_000, 1200, false, 14_000),
            s(200.0, 130_000, 1300, true, 27_000),
        ];
        let r = summarize(&w).unwrap();
        assert_eq!(r.n, 3);
        assert!((r.seconds - 0.2).abs() < 1e-12);
        assert!((r.mean_power_w - 130.0).abs() < 1e-9);
        assert!((r.max_power_w - 140.0).abs() < 1e-9);
        assert!((r.mean_sm_mhz - 1_303.333_333).abs() < 1e-3);
        assert_eq!(r.min_sm_mhz, 1200);
        assert!((r.cap_fraction - 2.0 / 3.0).abs() < 1e-12);
        assert!((r.energy_j - 26.0).abs() < 1e-12);
        assert_eq!(summarize(&[]), None);
    }

    /// Without a GPU (desktop container) NVML is absent: an error, not a panic.
    #[test]
    fn start_without_nvml_is_an_error() {
        if let Err(e) = Sampler::start("GPU-00000000-0000-0000-0000-000000000000", PERIOD_10HZ) {
            assert!(matches!(e, CudaError::Nvml { .. }), "{e}");
        }
    }
}
