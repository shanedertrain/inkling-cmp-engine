//! The back-end stream of one request.

use std::sync::Arc;

use ink_backend::{Backend, GenRequest, StepOutput};
use tokio::sync::mpsc::{UnboundedReceiver, unbounded_channel};

/// One request submitted to a back end. Dropping it before [`Run::stop`] cancels the request, so a client that
/// disconnects (the handler's future is dropped) never leaves the back end generating.
pub(crate) struct Run {
    backend: Arc<dyn Backend>,
    id: u64,
    steps: UnboundedReceiver<StepOutput>,
    stopped: bool,
}

impl Run {
    /// Submits `request`; its steps arrive through [`Run::next`].
    pub(crate) fn start(backend: Arc<dyn Backend>, request: GenRequest) -> Self {
        let id = request.id;
        let (tx, steps) = unbounded_channel();
        backend.submit(
            request,
            Box::new(move |step| {
                // the receiver is gone once the request finished or was cancelled
                let _ = tx.send(step);
            }),
        );
        Self {
            backend,
            id,
            steps,
            stopped: false,
        }
    }

    /// The next step; `None` when the back end dropped the request without a final step.
    pub(crate) async fn next(&mut self) -> Option<StepOutput> {
        self.steps.recv().await
    }

    /// Tells the back end the request is over: `cancel` for a vanished client, else `finish`.
    pub(crate) fn stop(&mut self, cancel: bool) {
        if self.stopped {
            return;
        }
        self.stopped = true;
        if cancel {
            self.backend.cancel(self.id);
        } else {
            self.backend.finish(self.id);
        }
    }
}

impl Drop for Run {
    fn drop(&mut self) {
        self.stop(true);
    }
}
