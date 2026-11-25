// Ported from engine/server/src/api/openai.{h,cpp} @ 4171493 (counters, `/metrics` text).
//! Front-end counters and the Prometheus text of `/metrics` (series names follow vLLM's so that `agent_replay` and
//! the fleet gateway read the same ones).

use std::fmt::Write as _;
use std::sync::Mutex;
use std::sync::atomic::{AtomicI64, Ordering};
use std::time::Duration;

use ink_backend::BackendMetrics;
use ink_text::pyfmt::format_g;

/// Upper bounds of the time-to-first-token histogram, in microseconds.
const TTFT_BUCKETS_US: [i64; 8] = [
    100_000, 250_000, 500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000, 20_000_000,
];

/// Counters of the requests a [`Frontend`](crate::Frontend) handled.
#[derive(Debug, Default)]
pub struct FrontendStats {
    pub(crate) requests_total: AtomicI64,
    pub(crate) requests_running: AtomicI64,
    pub(crate) requests_cancelled: AtomicI64,
    pub(crate) requests_failed: AtomicI64,
    pub(crate) prompt_tokens_total: AtomicI64,
    pub(crate) generation_tokens_total: AtomicI64,
    pub(crate) cached_prompt_tokens_total: AtomicI64,
    ttft_us_sum: AtomicI64,
    ttft_count: AtomicI64,
    e2e_us_sum: AtomicI64,
    e2e_count: AtomicI64,
    /// Counts per bucket of [`TTFT_BUCKETS_US`] and one overflow bucket.
    ttft_buckets: Mutex<[i64; TTFT_BUCKETS_US.len() + 1]>,
}

fn micros(d: Duration) -> i64 {
    i64::try_from(d.as_micros()).unwrap_or(i64::MAX)
}

fn count(n: usize) -> i64 {
    i64::try_from(n).unwrap_or(i64::MAX)
}

impl FrontendStats {
    pub(crate) fn add_tokens(counter: &AtomicI64, n: usize) {
        counter.fetch_add(count(n), Ordering::Relaxed);
    }

    pub(crate) fn record_ttft(&self, ttft: Duration) {
        let us = micros(ttft);
        self.ttft_us_sum.fetch_add(us, Ordering::Relaxed);
        self.ttft_count.fetch_add(1, Ordering::Relaxed);
        let bucket = TTFT_BUCKETS_US
            .iter()
            .position(|&b| us <= b)
            .unwrap_or(TTFT_BUCKETS_US.len());
        self.ttft_buckets.lock().expect("lock poisoned")[bucket] += 1;
    }

    pub(crate) fn record_e2e(&self, e2e: Duration) {
        self.e2e_us_sum.fetch_add(micros(e2e), Ordering::Relaxed);
        self.e2e_count.fetch_add(1, Ordering::Relaxed);
    }

    /// `/metrics` in the Prometheus text format.
    pub(crate) fn render(&self, model: &str, m: &BackendMetrics) -> String {
        let load = |c: &AtomicI64| c.load(Ordering::Relaxed);
        let labels = format!("{{model_name=\"{model}\"}}");
        let mut out = String::new();
        let mut series = |name: &str, help: &str, kind: &str, value: f64| {
            let _ = write!(
                out,
                "# HELP {name} {help}\n# TYPE {name} {kind}\n{name}{labels} {}\n",
                format_g(value)
            );
        };
        series(
            "vllm:num_requests_running",
            "Number of requests in model execution batches.",
            "gauge",
            f64::from(m.running),
        );
        series(
            "vllm:num_requests_waiting",
            "Number of requests waiting to be processed.",
            "gauge",
            f64::from(m.waiting),
        );
        series(
            "ink:num_sessions_parked",
            "Sessions whose KV stays resident for prefix reuse.",
            "gauge",
            f64::from(m.parked),
        );
        series(
            "vllm:kv_cache_usage_perc",
            "KV-cache usage. 1 means 100 percent usage.",
            "gauge",
            if m.kv_pages_total == 0 {
                0.0
            } else {
                m.kv_pages_used as f64 / m.kv_pages_total as f64
            },
        );
        series(
            "vllm:prefix_cache_queries_total",
            "Prefix cache queries, in terms of number of queried tokens.",
            "counter",
            m.prefix_cache_queries as f64,
        );
        series(
            "vllm:prefix_cache_hits_total",
            "Prefix cache hits, in terms of number of cached tokens.",
            "counter",
            m.prefix_cache_hits as f64,
        );
        series(
            "vllm:prompt_tokens_total",
            "Number of prefill tokens processed.",
            "counter",
            load(&self.prompt_tokens_total) as f64,
        );
        series(
            "vllm:generation_tokens_total",
            "Number of generation tokens processed.",
            "counter",
            load(&self.generation_tokens_total) as f64,
        );
        series(
            "vllm:spec_decode_num_draft_tokens_total",
            "Number of draft tokens.",
            "counter",
            m.spec_draft_tokens as f64,
        );
        series(
            "vllm:spec_decode_num_accepted_tokens_total",
            "Number of accepted tokens.",
            "counter",
            m.spec_accepted_tokens as f64,
        );
        series(
            "ink:requests_total",
            "HTTP generation requests received.",
            "counter",
            load(&self.requests_total) as f64,
        );
        series(
            "ink:requests_cancelled_total",
            "Requests cancelled by client disconnect.",
            "counter",
            load(&self.requests_cancelled) as f64,
        );
        series(
            "ink:requests_failed_total",
            "Requests rejected or failed.",
            "counter",
            load(&self.requests_failed) as f64,
        );
        out.push_str(
            "# HELP vllm:time_to_first_token_seconds Histogram of time to first token in seconds.\n\
             # TYPE vllm:time_to_first_token_seconds histogram\n",
        );
        let buckets = *self.ttft_buckets.lock().expect("lock poisoned");
        let mut cumulative = 0;
        for (bound, n) in TTFT_BUCKETS_US.iter().zip(buckets) {
            cumulative += n;
            let _ = writeln!(
                out,
                "vllm:time_to_first_token_seconds_bucket{{le=\"{}\",model_name=\"{model}\"}} {cumulative}",
                format_g(*bound as f64 / 1e6)
            );
        }
        cumulative += buckets[TTFT_BUCKETS_US.len()];
        let _ = writeln!(
            out,
            "vllm:time_to_first_token_seconds_bucket{{le=\"+Inf\",model_name=\"{model}\"}} {cumulative}"
        );
        let _ = writeln!(
            out,
            "vllm:time_to_first_token_seconds_sum{labels} {}",
            format_g(load(&self.ttft_us_sum) as f64 / 1e6)
        );
        let _ = writeln!(
            out,
            "vllm:time_to_first_token_seconds_count{labels} {}",
            load(&self.ttft_count)
        );
        out.push_str("# HELP vllm:e2e_request_latency_seconds_sum End-to-end request latency.\n");
        let _ = writeln!(
            out,
            "vllm:e2e_request_latency_seconds_sum{labels} {}",
            format_g(load(&self.e2e_us_sum) as f64 / 1e6)
        );
        let _ = writeln!(
            out,
            "vllm:e2e_request_latency_seconds_count{labels} {}",
            load(&self.e2e_count)
        );
        out
    }
}

/// Counts a request as running while it lives; dropped without [`Inflight::complete`] it counts as cancelled
/// (the transport dropped the request's future).
pub(crate) struct Inflight<'s> {
    stats: &'s FrontendStats,
    cancelled: bool,
}

impl<'s> Inflight<'s> {
    pub(crate) fn start(stats: &'s FrontendStats) -> Self {
        stats.requests_running.fetch_add(1, Ordering::Relaxed);
        Self { stats, cancelled: true }
    }

    /// The request ended by itself, with its result or an error.
    pub(crate) fn complete(mut self) {
        self.cancelled = false;
    }

    /// The writer reported that the client is gone.
    pub(crate) fn client_gone(self) {}
}

impl Drop for Inflight<'_> {
    fn drop(&mut self) {
        self.stats.requests_running.fetch_sub(1, Ordering::Relaxed);
        if self.cancelled {
            self.stats.requests_cancelled.fetch_add(1, Ordering::Relaxed);
        }
    }
}
