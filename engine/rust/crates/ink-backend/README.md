# ink-backend (package F4)

The `Backend` trait between the front end and whatever produces tokens (`submit`, `finish`, `cancel`, `metrics`,
`healthy`) with its types (`GenRequest`, `SamplingParams`, `StepOutput`, `TokenLogprobs`, `BackendMetrics`), and
`MockBackend`: scripted token groups, one thread per request, honouring `finish` and `cancel` between steps (tests,
`ink-replay`, `ink-server --backend mock`). Port of `engine/server/src/backend/{backend.h,mock_backend.*}` @ `4171493`
(deleted at cut-over U2). Implementations: `MockBackend` here, `ink-sched::EngineBackend` (scheduler and runtime on one
loop thread).

Contract: `on_step` runs on the back end's thread, in order, until a `StepOutput` with `done`; a back end that ends a
request by itself sends `done` (with `session_cap` for the per-session cap, `error` for a failure). The front end owns
stop handling and may receive the tokens of one step beyond the stop; it drops them.
