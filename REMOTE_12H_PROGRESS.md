# Remote 12-hour fresh-model assignment: progress

Plan: [REMOTE_12H_PLAN.md](REMOTE_12H_PLAN.md). Owned paths: this file,
`REMOTE_12H_HANDOFF.md`, `experiments/upstream-remote-12h/`,
`scripts/remote12h_supervisor.py`, `tests/test_remote12h_supervisor.py`.

## Clock (UTC)

| Mark | Time |
| --- | --- |
| Assignment start (setup counts) | 2026-11-18T03:01:05Z |
| No new native job after (9.5 h) | 2026-11-18T12:31:05Z |
| Hard training stop (10 h, supervisor deadline) | 2026-11-18T13:01:05Z |
| Final deadline (12 h) | 2026-11-18T15:01:05Z |

## Host

Apple M3 Max, 14 cores (10P+4E), 96 GiB RAM (103079215104 B), macOS 27.2.1
(26B131), ~92 GiB free disk at launch, swap 0. Rust 1.93.0 (system rustup; no
isolated toolchain under `artifacts/upstream-repro/toolchain`). Checkout:
`~/workspace/netflix_upstream_repro` (previous idle upstream-repro checkout,
switched from `research/upstream-reproduction` @ db394c1, which is already on
origin). Native tree `artifacts/upstream-repro/fresh` at pin
`49c9b3676e6bab90fdf77e9a2da984edbfe4fe5c`, only the zero-sentinel qualifying
payload differs; `validate_fresh()` passed. Data receipt ratings SHA256
`5a554ca04854b25a0422771b40a02415e488212513a475f3eda10c01238fa2f7`.

## Preflight

- `test_upstream*.py`: 29 tests OK.
- `--plan-job bknbrx-100` and `--plan-job cfnade-96`: each a single native job
  (deps ingest/newsplit already receipted by setup). No pre-existing
  bknbrx/cfnade outputs in `fresh/preds_new`.
- Supervisor (`scripts/remote12h_supervisor.py`) toy tests: 4 OK, including a
  deadline stop of a process tree whose grandchild ignores SIGTERM (SIGKILL
  after grace, no leftover group members).

## Supervision

tmux session `remote12h`, `caffeinate -dimsu`. Supervisor runs the wrapper in
its own process group; samples group RSS, `kern.memorystatus_level`, swap and
disk every 60 s to `experiments/upstream-remote-12h/<job>/resources.jsonl`;
stops only its group at the deadline, on sustained memory pressure (level <20%
and swap growth >4 GiB for 5 samples) or disk <10 GiB; heartbeat
commit/push of its status files every 30 min.

## Jobs

| Job | State | Started | Finished | Notes |
| --- | --- | --- | --- | --- |
| bknbrx-100 | running | 2026-11-18T03:03:00Z | | `UPSTREAM_REPRO_THREADS=8 UPSTREAM_REPRO_BLAS_THREADS=1`; native config n_epochs 5, seed 42, Hogwild, n_threads 8 (unpatched) |
| cfnade-96 | not started | | | conditional on measured cost fitting before 12:31Z |

## Log

- 03:01Z start; 03:03Z bknbrx-100 launched (trainx => probex phase).
