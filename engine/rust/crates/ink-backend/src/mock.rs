// Ported from engine/server/src/backend/mock_backend.{h,cpp} @ 4171493.
//! Scripted back end for tests, the HTTP smoke test and the parity replay.
//!
//! Each submitted request takes the next [`Script`] (first in, first out) or the one the generator returns for it,
//! and emits one [`StepOutput`] per group of tokens on its own thread, honouring `finish` and `cancel` between steps.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::{Backend, BackendMetrics, GenRequest, StepFn, StepOutput, TokenLogprobs};

/// How a script ends once its groups are out.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Ending {
    /// A bare `done` marker: the front end reports `abort` unless it finished the request itself.
    #[default]
    Done,
    /// `done` with an error (`"mock back end error"`).
    Error,
    /// `done` with `session_cap` set: the front end reports `length`.
    SessionCap,
}

/// What the mock does for one request.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Script {
    /// The tokens of each model step.
    pub groups: Vec<Vec<i32>>,
    /// Logprobs per step, parallel to `groups` (missing steps carry none).
    pub logprobs: Vec<Vec<TokenLogprobs>>,
    /// Reported on the first step.
    pub num_cached_tokens: Option<u32>,
    /// Wall-clock delay before every step but the first (tests of cancel).
    pub step_delay: Duration,
    /// Wall-clock "prefill" delay before the first step.
    pub first_delay: Duration,
    /// How the request ends after the last group (unless it was stopped first).
    pub ending: Ending,
}

/// Makes the script of a request when no scripted one is queued.
pub type Generator = Box<dyn Fn(&GenRequest) -> Script + Send + Sync>;

#[derive(Default)]
struct Inner {
    live: Mutex<HashMap<u64, Arc<AtomicBool>>>,
    submitted: Mutex<Vec<GenRequest>>,
    running: AtomicU32,
    cancels: AtomicU32,
    gen_tokens: AtomicU64,
    prompt_tokens: AtomicU64,
}

impl Inner {
    fn stop(&self, id: u64) {
        if let Some(flag) = self.live.lock().expect("lock poisoned").get(&id) {
            flag.store(true, Ordering::SeqCst);
        }
    }
}

/// A [`Backend`] that replays scripted token streams.
#[derive(Default)]
pub struct MockBackend {
    inner: Arc<Inner>,
    fifo: Mutex<VecDeque<Script>>,
    generator: Mutex<Option<Generator>>,
    threads: Mutex<Vec<JoinHandle<()>>>,
}

impl std::fmt::Debug for MockBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MockBackend")
            .field("running", &self.inner.running.load(Ordering::SeqCst))
            .finish_non_exhaustive()
    }
}

impl MockBackend {
    /// A mock with no scripts: a request without one gets an empty script (it ends at once).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Queues the script of the next request.
    pub fn push(&self, script: Script) {
        self.fifo.lock().expect("lock poisoned").push_back(script);
    }

    /// Sets the generator used when the queue is empty.
    pub fn set_generator(&self, generator: impl Fn(&GenRequest) -> Script + Send + Sync + 'static) {
        *self.generator.lock().expect("lock poisoned") = Some(Box::new(generator));
    }

    /// The requests submitted so far, in order.
    #[must_use]
    pub fn submitted(&self) -> Vec<GenRequest> {
        self.inner.submitted.lock().expect("lock poisoned").clone()
    }

    /// Number of [`Backend::cancel`] calls.
    #[must_use]
    pub fn cancels(&self) -> u32 {
        self.inner.cancels.load(Ordering::SeqCst)
    }

    fn script_for(&self, req: &GenRequest) -> Script {
        if let Some(script) = self.fifo.lock().expect("lock poisoned").pop_front() {
            return script;
        }
        self.generator
            .lock()
            .expect("lock poisoned")
            .as_ref()
            .map(|generate| generate(req))
            .unwrap_or_default()
    }
}

fn play(inner: &Inner, script: &Script, stop: &AtomicBool, mut on_step: StepFn) {
    if !script.first_delay.is_zero() {
        thread::sleep(script.first_delay);
    }
    for (i, group) in script.groups.iter().enumerate() {
        if stop.load(Ordering::SeqCst) {
            break;
        }
        if i > 0 && !script.step_delay.is_zero() {
            thread::sleep(script.step_delay);
        }
        if stop.load(Ordering::SeqCst) {
            break;
        }
        inner.gen_tokens.fetch_add(group.len() as u64, Ordering::SeqCst);
        on_step(StepOutput {
            tokens: group.clone(),
            logprobs: script.logprobs.get(i).cloned().unwrap_or_default(),
            num_cached_tokens: if i == 0 { script.num_cached_tokens } else { None },
            ..StepOutput::default()
        });
    }
    let stopped = stop.load(Ordering::SeqCst);
    let mut done = StepOutput {
        done: true,
        ..StepOutput::default()
    };
    if !stopped {
        match script.ending {
            Ending::Done => {}
            Ending::Error => done.error = Some("mock back end error".to_owned()),
            Ending::SessionCap => done.session_cap = true,
        }
    }
    on_step(done);
}

impl Backend for MockBackend {
    fn submit(&self, req: GenRequest, on_step: StepFn) {
        let script = self.script_for(&req);
        let id = req.id;
        let stop = Arc::new(AtomicBool::new(false));
        self.inner
            .prompt_tokens
            .fetch_add(req.prompt.len() as u64, Ordering::SeqCst);
        self.inner.submitted.lock().expect("lock poisoned").push(req);
        self.inner
            .live
            .lock()
            .expect("lock poisoned")
            .insert(id, Arc::clone(&stop));
        self.inner.running.fetch_add(1, Ordering::SeqCst);
        let inner = Arc::clone(&self.inner);
        let handle = thread::spawn(move || {
            play(&inner, &script, &stop, on_step);
            inner.running.fetch_sub(1, Ordering::SeqCst);
            inner.live.lock().expect("lock poisoned").remove(&id);
        });
        let mut threads = self.threads.lock().expect("lock poisoned");
        // dropping the handle of a finished thread releases it; without this a long-running mock server keeps one
        // unjoined thread per request ever served
        threads.retain(|h| !h.is_finished());
        threads.push(handle);
    }

    fn finish(&self, id: u64) {
        self.inner.stop(id);
    }

    fn cancel(&self, id: u64) {
        self.inner.cancels.fetch_add(1, Ordering::SeqCst);
        self.inner.stop(id);
    }

    fn metrics(&self) -> BackendMetrics {
        BackendMetrics {
            running: self.inner.running.load(Ordering::SeqCst),
            prompt_tokens: self.inner.prompt_tokens.load(Ordering::SeqCst),
            generation_tokens: self.inner.gen_tokens.load(Ordering::SeqCst),
            ..BackendMetrics::default()
        }
    }

    fn healthy(&self) -> bool {
        true
    }
}

impl Drop for MockBackend {
    fn drop(&mut self) {
        let threads = std::mem::take(&mut *self.threads.lock().expect("lock poisoned"));
        for handle in threads {
            // A panicking script thread was already reported by the panic hook.
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;

    use super::*;

    fn request(id: u64) -> GenRequest {
        GenRequest {
            id,
            prompt: vec![1, 2, 3],
            ..GenRequest::default()
        }
    }

    fn collect(mock: &MockBackend, id: u64) -> Vec<StepOutput> {
        let (tx, rx) = mpsc::channel();
        mock.submit(
            request(id),
            Box::new(move |s| {
                let _ = tx.send(s);
            }),
        );
        rx.iter().collect()
    }

    #[test]
    fn emits_one_step_per_group_then_done() {
        let mock = MockBackend::new();
        mock.push(Script {
            groups: vec![vec![5, 6], vec![7]],
            num_cached_tokens: Some(2),
            ..Script::default()
        });
        let steps = collect(&mock, 1);
        let tokens: Vec<_> = steps.iter().map(|s| s.tokens.clone()).collect();
        assert_eq!(tokens, [vec![5, 6], vec![7], vec![]]);
        assert_eq!(steps[0].num_cached_tokens, Some(2));
        assert_eq!(steps[1].num_cached_tokens, None);
        assert!(steps[2].done && steps[2].error.is_none() && !steps[2].session_cap);
        assert_eq!(mock.submitted()[0].prompt, [1, 2, 3]);
    }

    #[test]
    fn endings_set_error_and_session_cap() {
        let mock = MockBackend::new();
        mock.push(Script {
            ending: Ending::Error,
            ..Script::default()
        });
        mock.push(Script {
            ending: Ending::SessionCap,
            ..Script::default()
        });
        assert_eq!(collect(&mock, 1)[0].error.as_deref(), Some("mock back end error"));
        assert!(collect(&mock, 2)[0].session_cap);
    }

    #[test]
    fn generator_serves_requests_without_a_queued_script() {
        let mock = MockBackend::new();
        mock.set_generator(|req| Script {
            groups: vec![vec![i32::try_from(req.id).expect("small id")]],
            ..Script::default()
        });
        assert_eq!(collect(&mock, 9)[0].tokens, [9]);
    }

    #[test]
    fn cancel_stops_the_script_and_is_counted() {
        let mock = MockBackend::new();
        mock.push(Script {
            groups: vec![vec![1]; 1000],
            step_delay: Duration::from_millis(5),
            ..Script::default()
        });
        let (tx, rx) = mpsc::channel();
        mock.submit(
            request(3),
            Box::new(move |s| {
                let _ = tx.send(s);
            }),
        );
        rx.recv().expect("first step");
        mock.cancel(3);
        let rest: Vec<_> = rx.iter().collect();
        assert!(rest.len() < 999, "the script stopped early ({} steps left)", rest.len());
        assert!(rest.last().is_some_and(|s| s.done && s.error.is_none()));
        assert_eq!(mock.cancels(), 1);
    }
}
