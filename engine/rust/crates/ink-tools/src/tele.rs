//! NVML telemetry windows of the benchmarks (`p6::Tele` of `engine/kernels/decode_attn/bench/common.hpp`): a 10 Hz
//! sampler per timed phase and the statistics of the samples inside a window, with the C++ tool's field names.

use std::time::Instant;

use ink_cuda::{PERIOD_10HZ, Sample, Sampler};

use crate::line::Line;

/// The card a benchmark ran on.
#[derive(Debug, Clone, Default)]
pub struct Card {
    /// UUID (`?` when NVML is absent).
    pub uuid: String,
    /// vBIOS version string.
    pub vbios: String,
    /// Enforced power limit in watts.
    pub power_limit_w: f64,
}

impl Card {
    /// Reads the card with NVML index 0 (a container given one card sees it as 0); `?` fields when NVML fails, as
    /// the C++ tools print them.
    #[must_use]
    pub fn read() -> Self {
        match Sampler::start_index(0, PERIOD_10HZ) {
            Ok(s) => Self {
                uuid: s.info().uuid.clone(),
                vbios: s.info().vbios.clone(),
                power_limit_w: f64::from(s.info().power_limit_mw) / 1000.0,
            },
            Err(e) => {
                eprintln!("nvml: {e}");
                Self {
                    uuid: "?".to_owned(),
                    vbios: "?".to_owned(),
                    power_limit_w: 0.0,
                }
            }
        }
    }
}

/// A running sampler for one timed phase.
#[derive(Debug)]
pub struct Tele {
    sampler: Option<Sampler>,
    samples: Vec<Sample>,
}

impl Tele {
    /// Starts sampling the card with index 0 (nothing when NVML is absent).
    #[must_use]
    pub fn start() -> Self {
        Self {
            sampler: Sampler::start_index(0, PERIOD_10HZ).ok(),
            samples: Vec::new(),
        }
    }

    /// Stops sampling; the readings stay for [`Tele::window`].
    pub fn stop(&mut self) {
        if let Some(s) = self.sampler.take() {
            match s.stop() {
                Ok(v) => self.samples = v,
                Err(e) => eprintln!("nvml: {e}"),
            }
        }
    }

    fn inside(&self, t0: Instant, t1: Instant) -> Vec<&Sample> {
        self.samples.iter().filter(|x| x.at >= t0 && x.at <= t1).collect()
    }

    /// Mean power over the energy counter between the first and last reading in `[t0, t1]` (NaN with fewer than 2).
    #[must_use]
    pub fn mean_energy_w(&self, t0: Instant, t1: Instant) -> f64 {
        let s = self.inside(t0, t1);
        match (s.first(), s.last()) {
            (Some(a), Some(b)) if s.len() > 1 => {
                let dt = b.at.duration_since(a.at).as_secs_f64();
                b.energy_mj.saturating_sub(a.energy_mj) as f64 / 1000.0 / dt
            }
            _ => f64::NAN,
        }
    }

    /// Appends the window statistics of the readings in `[t0, t1]` to `line` (`Tele::window` of the C++ harness:
    /// `tele_n` always, the rest with at least 3 readings).
    #[must_use]
    pub fn window(&self, line: Line, t0: Instant, t1: Instant) -> Line {
        let s = self.inside(t0, t1);
        let line = line.uint("tele_n", s.len() as u64);
        let (Some(first), Some(last)) = (s.first(), s.last()) else {
            return line;
        };
        if s.len() < 3 {
            return line;
        }
        let n = s.len() as f64;
        let dt = last.at.duration_since(first.at).as_secs_f64();
        let power: Vec<f64> = s.iter().map(|x| f64::from(x.power_mw) / 1000.0).collect();
        let mut sm: Vec<f64> = s.iter().map(|x| f64::from(x.sm_mhz)).collect();
        let sm_mean = sm.iter().sum::<f64>() / n;
        sm.sort_by(f64::total_cmp);
        let e_j = last.energy_mj.wrapping_sub(first.energy_mj) as f64 / 1000.0;
        let count = |mask: u64| s.iter().filter(|x| x.reasons & mask != 0).count() as u64;
        line.num("tele_hz", (n - 1.0) / dt)
            .num("power_w_mean", power.iter().sum::<f64>() / n)
            .num("power_w_max", power.iter().copied().fold(0.0, f64::max))
            .num("energy_j", e_j)
            .num("energy_dt_s", dt)
            .num("energy_w_mean", e_j / dt)
            .num("sm_mhz_mean", sm_mean)
            .num("sm_mhz_median", sm[sm.len() / 2])
            .uint("sm_mhz_min", s.iter().map(|x| x.sm_mhz).min().unwrap_or(0))
            .uint("sm_mhz_max", s.iter().map(|x| x.sm_mhz).max().unwrap_or(0))
            .uint("mem_mhz", first.mem_mhz)
            .num("sw_power_cap_frac", count(0x4) as f64 / n)
            .uint("hw_slowdown_n", count(0x8))
            .uint("thermal_n", count(0x60))
            .uint("power_brake_n", count(0x80))
            .uint("temp_c_start", first.temp_c)
            .uint("temp_c_max", s.iter().map(|x| x.temp_c).max().unwrap_or(0))
    }
}
