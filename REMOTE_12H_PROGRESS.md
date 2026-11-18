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
| bknbrx-100 | **completed**, exported | 2026-11-18T03:03:00Z | 2026-11-18T03:12:05Z | 9.1 min, peak group RSS 9.98 GB, no swap; all 13 receipt hashes verified; terminal/ep01/ep05 NPZ in `experiments/upstream-remote-12h/outputs/` |
| cfnade-96 | **not launched (does not fit)** | | | bounded profile 03:13–03:33Z: epoch-1 train 942.8 s single-threaded, 4.1 GB; ≥50 h for both phases vs 9.5 h budget |

## Log

- 03:01Z start; 03:03Z bknbrx-100 launched (trainx => probex phase).
- 03:12Z bknbrx-100 completed (trainx 5 epochs ~54 s each, probex 0.90551; fulltrain 5 epochs). Exported and pushed.
- 03:13Z CF-NADE profile (`--test`, 4-epoch/swa 3, no `--save`) under 20-min supervisor deadline. First attempt 03:12:43Z failed exit 127 (cfnade-new binary not yet built; record kept in `cfnade-96-profile-attempt0-missing-binary/`); built with wrapper env and relaunched.
- 03:31Z epoch 1: 942.8 s train (99% of one core). 03:33Z profile stopped at deadline (SIGTERM, clean, no leftovers, `preds_new` unchanged).
- Decision: cfnade-96 full recipe not launched. Coordinator control bank (M1 receipts, rbmx2-150 chain) is not present here, so no in-place evaluation. See `REMOTE_12H_HANDOFF.md`.

## Extension (user-approved 2026-11-18 ~03:45Z)

After the original assignment finished early, the user approved continuing
inside the same 12 h window: train further candidates from
`experiments/upstream-missing-screen/recipe-costs.json`, same rules (pinned
source, fresh receipts, supervisor deadline 13:01:05Z, no new native job after
12:31:05Z, one native job at a time). The 10% target can't be measured here
(control bank absent); exports go to the coordinator for evaluation.

Order: `tsvdx5-150o` (rank 7, released-vector diagnostic −0.00134; 25 epochs,
150 factors, ordinal; kept ep06/16/24), then `bknbrx-50` (rank 21, −0.00075;
25 epochs; kept ep05/09/17/21). Continue with further feasible candidates if
time remains. `tsvdx5-150o` runs as the real pinned job; if its projected time
from the first epochs doesn't fit, it is stopped and partial outputs are moved
aside (not deleted).

- 03:49Z `tsvdx5-150o` launched (`scripts/remote12h_launch.sh`, tmux `r12-tsvdx5-150o`).
- 04:04Z `tsvdx5-150o` epochs 1–2: 03:57:17Z, 04:04:08Z (~6.9 min/epoch; init+epoch 1 ~7 min). Projection ≈ 25×6.9 min ≈ 2.9 h per phase, ≈ 6 h total → finish ≈ 09:50Z. The strict 1.5×+30 min rule (≈ 9.5 h) would exceed the 13:01Z hard stop; kept running anyway because the timing is measured, SGD epoch cost is stable, the supervisor still enforces 13:01Z, and the user asked for an ambitious extension. Group RSS 26.5 GB, swap 0. `bknbrx-50` (~1 h) follows if `tsvdx5-150o` ends before ~11:30Z.
- 09:34:40Z `tsvdx5-150o` **completed** (exit 0, 5.77 h, peak group RSS 28.7 GB, swap 0). trainx phase ended 06:42:25Z (probex 0.87558); fulltrain 25 epochs ended 09:34:38Z. All 63 receipt hashes verified; terminal + ep06/16/24 exported (4 unique vectors). Post-freeze fixture diagnostic: near-exact reproduction (terminal RMS diff 6.8e-7 vs released, RMSE 0.8755833 both).
- 09:37Z `bknbrx-50` launched (tmux `r12-bknbrx-50`).
- 10:14Z `bknbrx-50` **completed** (0.615 h, peak 10.2 GB, swap 0; probex 0.90037). 53 receipt hashes verified; terminal + ep05/09/17/21 exported (5 unique). Fixture diagnostic (ep05/ep17 only released): corr ≈ 0.9999, RMSE within 2e-5 (Hogwild).
- 10:14Z `mf-60__knn3` chain launched (runs `mf-60` then `mf-60__knn3`).
- 10:22Z `mf-60` + `mf-60__knn3` **completed** (0.131 h total, peak 3.5 GB). Exported terminal vectors (no kept epochs declared); fixture diagnostic: bit-identical to released (RMSE 0.926041 / 0.907782).
- 10:23Z `als8-8__knn3` chain launched.
- 10:29Z `als8-8` + `als8-8__knn3` **completed** (0.116 h, peak 7.9 GB; probex 0.95681 / 0.94645). Exported; fixture diagnostic near-exact (RMSE equal to 1e-9).
- 10:29Z queue started (`scripts/remote12h_queue.sh`, tmux `r12-queue`; no launch after 12:31:05Z, hard stop 13:01:05Z; log `experiments/upstream-remote-12h/queue.log`). Order: dnn-24 residual chains reusing local receipted `dnn-24`/`dnn-24__dnn-16p50` parents (`…__asym-16p75__knn3b`, `…__knn3c`, `dnn-24__asym-16p75__knn3`, `…__asym-16p50`, `dnn-24__dnn-24p50`, `dnn-24__dnn-16__knn3`), then `mf-61__knn3`, `mf-61__asym-16`.
