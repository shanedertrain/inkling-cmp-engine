// Ported from engine/server/src/backend/native_adapter.{h,cpp} @ 4171493 (EngineBackend).
//! [`Backend`] over a [`Scheduler`] and a [`ModelRuntime`]: one loop thread per card owns both.
//!
//! Loop iteration: apply queued commands, `plan`, `step`, `commit`, deliver events to the requests' callbacks. The
//! pattern is vLLM v1's `EngineCore` loop (schedule, execute, update) and TGI's batching task behind a channel.

use std::collections::BTreeMap;
use std::error::Error;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use ink_backend::{Backend, BackendMetrics, GenRequest, StepFn, StepOutput};

use crate::config::{Config, PrefixReuse};
use crate::plan::{Finish, SeqEvent};
use crate::runtime::{Drafting, ModelRuntime, RuntimeCaps, StepError};
use crate::scheduler::{ConfigError, PromptLimits, Scheduler};

/// How long the idle loop waits for a command before planning again.
const IDLE_WAIT: Duration = Duration::from_millis(20);

enum Cmd {
    // boxed: a request is much larger than the other commands
    Submit(Box<GenRequest>, StepFn),
    Finish(u64),
    Cancel(u64),
}

/// Whole microseconds of `d`, for the per-step timing events (target `ink_step`, level debug).
fn micros(d: Duration) -> u64 {
    u64::try_from(d.as_micros()).unwrap_or(u64::MAX)
}

/// The runtime reported a device failure: the loop fails every request and ends.
struct Dead(String);

/// The loop thread's state.
struct Core {
    sched: Scheduler,
    rt: Box<dyn ModelRuntime>,
    cbs: BTreeMap<u64, StepFn>,
}

/// Why [`EngineBackend::spawn`] failed.
#[derive(Debug, thiserror::Error)]
pub enum SpawnError {
    /// The runtime could not be built.
    #[error("runtime construction failed")]
    Runtime(#[source] Box<dyn Error + Send + Sync>),
    /// The scheduler configuration, narrowed to the runtime's limits, is invalid.
    #[error("invalid scheduler configuration")]
    Config(#[from] ConfigError),
    /// The runtime constructor panicked.
    #[error("the runtime constructor panicked")]
    Panicked,
}

impl Core {
    /// The scheduler and runtime of one loop, narrowing `cfg` to what the runtime can do; with the runtime's caps and
    /// the scheduler's admission limits.
    fn new<R, E, F>(cfg: &mut Config, make_rt: F) -> Result<(Self, RuntimeCaps, PromptLimits), SpawnError>
    where
        R: ModelRuntime + 'static,
        E: Error + Send + Sync + 'static,
        F: FnOnce() -> Result<R, E>,
    {
        let rt = make_rt().map_err(|e| SpawnError::Runtime(Box::new(e)))?;
        let c = rt.caps();
        cfg.slots = cfg.slots.min(c.max_slots);
        cfg.t_max = cfg.t_max.min(c.t_max);
        cfg.page_tokens = c.page_tokens;
        if cfg.max_decode_rows > 0 {
            cfg.max_decode_rows = c.max_decode_rows;
        }
        cfg.max_prefill_rows = cfg.max_prefill_rows.min(c.max_prefill_rows);
        cfg.drafting = c.drafter;
        match c.drafter {
            Drafting::None => cfg.spec_k = 0,
            Drafting::Fixed { k, .. } if cfg.spec_k > 0 => cfg.spec_k = k,
            _ => {}
        }
        if !c.external_pages {
            cfg.trim_parked_pages = false;
        }
        if !(c.external_pages && c.conv_checkpoints) {
            cfg.prefix_reuse = if c.slot_rollback {
                PrefixReuse::SlotRollback
            } else {
                PrefixReuse::ExactResume
            };
        }
        if let Some(pages) = c.kv_pages {
            let configured = match cfg.pool_bytes {
                0 => cfg.pool_tokens,
                bytes => bytes / cfg.kv_bytes_per_token.max(1),
            };
            cfg.pool_bytes = 0;
            cfg.pool_tokens = configured.min(u64::from(pages) * u64::from(c.page_tokens));
            cfg.session_cap_tokens = cfg.session_cap_tokens.min(cfg.pool_tokens);
        }
        cfg.max_running = cfg.max_running.min(cfg.slots);
        let core = Self {
            sched: Scheduler::new(cfg.clone())?,
            rt: Box::new(rt),
            cbs: BTreeMap::new(),
        };
        let limits = core.sched.prompt_limits();
        Ok((core, c, limits))
    }

    fn deliver(&mut self, events: Vec<SeqEvent>) {
        for e in events {
            let done = e.finish.is_some();
            let out = StepOutput {
                tokens: e.tokens,
                logprobs: e.logprobs,
                num_cached_tokens: e.num_cached_tokens,
                session_cap: e.finish == Some(Finish::SessionCap),
                error: match e.finish {
                    Some(Finish::Error(m)) => Some(m),
                    _ => None,
                },
                done,
            };
            if done {
                if let Some(mut cb) = self.cbs.remove(&e.seq) {
                    cb(out);
                }
            } else if let Some(cb) = self.cbs.get_mut(&e.seq) {
                cb(out);
            }
        }
    }

    /// Fails every request the loop knows with `message` (their callbacks get a final error step).
    fn fail_all(&mut self, message: &str) {
        for (_, mut cb) in std::mem::take(&mut self.cbs) {
            cb(StepOutput {
                error: Some(message.to_owned()),
                done: true,
                ..StepOutput::default()
            });
        }
    }

    /// One loop iteration. Returns whether there was anything to do, or [`Dead`] after a device failure (every
    /// request known to the loop has been failed then).
    fn run_once(&mut self, cmds: Vec<Cmd>) -> Result<bool, Dead> {
        let mut worked = !cmds.is_empty();
        let now = self.rt.now_ms();
        for cmd in cmds {
            match cmd {
                Cmd::Submit(req, mut cb) => {
                    let id = req.id;
                    match self.sched.add(*req, now) {
                        Ok(()) => {
                            self.cbs.insert(id, cb);
                        }
                        Err(e) => cb(StepOutput {
                            error: Some(e.to_string()),
                            done: true,
                            ..StepOutput::default()
                        }),
                    }
                }
                Cmd::Finish(id) => self.sched.finish(id, now),
                Cmd::Cancel(id) => self.sched.cancel(id, now),
            }
        }
        let events = self.sched.take_events();
        self.deliver(events);
        let t_plan = Instant::now();
        let plan = self.sched.plan(self.rt.now_ms());
        if !plan.is_empty() {
            worked = true;
            let t_step = Instant::now();
            let results = match self.rt.step(&plan) {
                Ok(results) => results,
                Err(StepError::Device(m)) => {
                    let message = format!("the engine stopped after a device failure: {m}");
                    tracing::error!("{message}");
                    self.fail_all(&message);
                    return Err(Dead(message));
                }
                Err(StepError::Request(m)) => {
                    let mut errs = Vec::new();
                    for g in &plan.segs {
                        self.sched.abort(g.seq, self.rt.now_ms());
                        errs.push(SeqEvent::finished(g.seq, Finish::Error(m.clone())));
                    }
                    self.sched.take_events(); // replaced by the error events
                    self.deliver(errs);
                    Vec::new()
                }
            };
            let t_commit = Instant::now();
            let events = self.sched.commit(&plan, &results, self.rt.now_ms());
            let t_deliver = Instant::now();
            self.deliver(events);
            let t_end = Instant::now();
            tracing::debug!(
                target: "ink_step",
                phase = "sched",
                segs = plan.segs.len(),
                decode_rows = plan.decode_rows,
                prefill_rows = plan.prefill_rows,
                plan_us = micros(t_step - t_plan),
                step_us = micros(t_commit - t_step),
                commit_us = micros(t_deliver - t_commit),
                deliver_us = micros(t_end - t_deliver),
            );
        }
        Ok(worked)
    }

    fn run(mut self, rx: Receiver<Cmd>, stop: &AtomicBool, metrics: &Mutex<BackendMetrics>) {
        let mut cmds = Vec::new();
        while !stop.load(Ordering::Acquire) {
            while let Ok(c) = rx.try_recv() {
                cmds.push(c);
            }
            let worked = match self.run_once(std::mem::take(&mut cmds)) {
                Ok(worked) => worked,
                Err(Dead(message)) => {
                    // submits already queued get the same answer; later ones find the channel closed (`send`)
                    while let Ok(c) = rx.try_recv() {
                        if let Cmd::Submit(_, mut cb) = c {
                            cb(StepOutput {
                                error: Some(message.clone()),
                                done: true,
                                ..StepOutput::default()
                            });
                        }
                    }
                    return;
                }
            };
            *metrics.lock().expect("metrics lock poisoned") = self.sched.metrics();
            if worked {
                continue;
            }
            match rx.recv_timeout(IDLE_WAIT) {
                Ok(c) => cmds.push(c),
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => return,
            }
        }
    }
}

/// [`Backend`] over a [`Scheduler`] and a [`ModelRuntime`], driven by one loop thread.
///
/// Callbacks run on the loop thread, in order, until a [`StepOutput`] with `done`. Dropping the backend stops the
/// thread after the step in flight; callbacks of requests still in flight are dropped without a `done` call. A
/// [`StepError::Device`] from the runtime is fatal: every request the loop knows (running, waiting, queued) gets a
/// final error step, the loop ends, [`Backend::healthy`] turns false and every later submit is answered with an
/// error; the process is expected to exit (`ink-server` does), as the C++ runtime's `ink::fatal` ends it. A panic on
/// the loop thread (scheduler, runtime or callback) ends the loop the same way, without the error steps. There is no
/// per-callback recovery, because the scheduler's state after a panic in the middle of a step is unknown.
pub struct EngineBackend {
    tx: Option<Sender<Cmd>>,
    stop: Arc<AtomicBool>,
    metrics: Arc<Mutex<BackendMetrics>>,
    thread: Option<JoinHandle<()>>,
    caps: RuntimeCaps,
    limits: PromptLimits,
}

impl std::fmt::Debug for EngineBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EngineBackend").finish_non_exhaustive()
    }
}

impl EngineBackend {
    /// Starts the loop thread, builds the runtime on it with `make_rt` and waits until it is ready. A CUDA runtime
    /// belongs to the thread that made it, so it is created here and never moves. `cfg` is narrowed to the
    /// runtime's limits first (slots, rows, page size; the runtime's drafting, `spec_k` 0 without a drafter; exact-resume
    /// prefix reuse only without external pages and conv checkpoints, parked pages kept whole without external pages;
    /// the KV pool and the session cap at most the runtime's own pages).
    ///
    /// # Errors
    ///
    /// [`SpawnError::Runtime`] when `make_rt` fails, [`SpawnError::Config`] when the narrowed configuration is
    /// invalid; the thread has ended in both cases.
    pub fn spawn<R, E, F>(mut cfg: Config, make_rt: F) -> Result<Self, SpawnError>
    where
        R: ModelRuntime + 'static,
        E: Error + Send + Sync + 'static,
        F: FnOnce() -> Result<R, E> + Send + 'static,
    {
        let (tx, rx) = mpsc::channel();
        let (ready_tx, ready_rx) = mpsc::sync_channel(1);
        let stop = Arc::new(AtomicBool::new(false));
        let metrics = Arc::new(Mutex::new(BackendMetrics::default()));
        let thread = thread::Builder::new()
            .name("ink-sched".into())
            .spawn({
                let (stop, metrics) = (Arc::clone(&stop), Arc::clone(&metrics));
                move || {
                    let core = match Core::new(&mut cfg, make_rt) {
                        Ok((core, caps, limits)) => {
                            let _ = ready_tx.send(Ok((caps, limits)));
                            core
                        }
                        Err(e) => {
                            let _ = ready_tx.send(Err(e));
                            return;
                        }
                    };
                    core.run(rx, &stop, &metrics);
                }
            })
            .expect("spawning the scheduler thread");
        let (caps, limits) = match ready_rx.recv() {
            Ok(Ok(ready)) => ready,
            Ok(Err(e)) => {
                let _ = thread.join();
                return Err(e);
            }
            // the thread ended before it answered: `make_rt` panicked
            Err(_) => {
                let _ = thread.join();
                return Err(SpawnError::Panicked);
            }
        };
        Ok(Self {
            tx: Some(tx),
            stop,
            metrics,
            thread: Some(thread),
            caps,
            limits,
        })
    }

    /// The runtime's limits and capabilities, as it reported them when the loop started.
    #[must_use]
    pub fn caps(&self) -> RuntimeCaps {
        self.caps
    }

    /// Queues `cmd`. While `self` is alive the loop ends only by a device failure or a panic; then a submit is
    /// answered at once, on the caller's thread, with an error (finish and cancel have nothing left to stop).
    fn send(&self, cmd: Cmd) {
        let Some(tx) = &self.tx else { return };
        if let Err(mpsc::SendError(Cmd::Submit(_, mut cb))) = tx.send(cmd) {
            cb(StepOutput {
                error: Some("scheduler loop stopped".to_owned()),
                done: true,
                ..StepOutput::default()
            });
        }
    }
}

impl Backend for EngineBackend {
    fn submit(&self, req: GenRequest, on_step: StepFn) {
        self.send(Cmd::Submit(Box::new(req), on_step));
    }

    fn finish(&self, id: u64) {
        self.send(Cmd::Finish(id));
    }

    fn cancel(&self, id: u64) {
        self.send(Cmd::Cancel(id));
    }

    /// The scheduler's session cap and KV pool ([`PromptLimits`]), the checks `Scheduler::add` makes first.
    fn check_prompt(&self, tokens: usize) -> Result<(), String> {
        self.limits.check(tokens).map_err(|e| e.to_string())
    }

    /// Metrics as of the end of the last loop iteration.
    fn metrics(&self) -> BackendMetrics {
        *self.metrics.lock().expect("metrics lock poisoned")
    }

    /// False once the loop thread has ended (a device failure, or a panic in the scheduler, the runtime or a callback).
    fn healthy(&self) -> bool {
        self.thread.as_ref().is_some_and(|t| !t.is_finished())
    }
}

impl Drop for EngineBackend {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        drop(self.tx.take()); // wakes an idle loop
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}
