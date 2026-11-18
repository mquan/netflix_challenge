# Remote temporal-model expansion: use the remaining laptop window

Branch: `research/remote-temporal-push`. This replaces the completed bounded
assignment, not its results. Preserve all bknbrx exports and CF-NADE profiles.
The receiving laptop is integrating bknbrx and has separately reached
**0.8644717283** ProbeX development RMSE with fresh-only nonlinear blending.
Those nonlinear artifacts are on `research/fresh-nonlinear-blend`; the remote
does not need to download that bank to perform this assignment.

## Why this is the next larger bet

The remote has 96 GiB RAM; the receiving laptop has 32 GiB. Dense temporal
models allocate large user/day arrays and provide an opportunity the smaller
machine cannot comfortably pursue. We want complete fresh parents, their
saved epochs/components, and compatible residual chains. Avoid another
untargeted blend or capacity sweep.

The fixed priority queue below is authorized in full, subject to measured
time/RAM and the original availability window. Do not stop after the first
success merely because the plan has no further assignment. Record a reason
for each skipped job and continue to the next feasible item. Scores do not
determine whether to launch the next queued model.

| Priority | Exact pinned job | Purpose | Fulltrain memory lower bound |
| --- | --- | --- | ---: |
| 1 | `tsvdx5-150o` | A dense temporal model with an ordinal head; 25 epochs per phase | 26.91 GB |
| 2 | `tsvdx5-300` | A different temporal regime; 300 factors, 23 epochs per phase | 49.74 GB |
| 3, capacity stretch | `tsvdx4-4800lm` | High-capacity low-memory temporal regime; 27 epochs per phase | 35.35 GB |
| 3a, after stretch parent | `tsvdx4-4800lm__knns` | Residual support-neighborhood chain, reusing its receipt-verified parent | Measure actual peak |
| 3b, after stretch parent | `tsvdx4-4800lm__knnf` | Factor-neighborhood chain, reusing the same parent | Measure actual peak |
| First fallback, profile first | `cfnade-12` | A new autoregressive family at 40 hidden units/rank 12, 40 epochs per phase; substantially smaller than the rejected 320-hidden/rank-96 preset | Measure actual peak |
| Second fallback | `bknbrx-50` | A complete cheaper neighborhood recipe when another temporal parent cannot fit the remaining time; 25 epochs per phase | Measure actual peak |

The stretch model has a lower RAM bound than the 300-factor dense model
because its low-memory implementation uses a different parameterization.
The 4800 name is not evidence it will train quickly. These are decimal GB
**lower bounds**, not measured peak RAM or promises of fitting. Leave at least
20% physical RAM for other allocations/OS and enforce the pressure guard.

Planning revision before new training results: the source also exposes the
small canonical CF-NADE preset. The 96-preset's 50-hour projection does not
establish the smaller preset's runtime. Profile `cfnade-12` at its actual
40-hidden/rank-12 settings before rejecting the whole family. The released
epoch-40 diagnostic improved the earlier ridge by 0.000239, so it is a lower
priority than the main temporal work but a useful new-family fallback. This
is one fixed native recipe, not a capacity sweep or permission to retry 96.

Released-vector diagnostics against the older 0.86632007 ridge suggested
bundle improvements of about 0.00134 for 150o and 0.00164 for 4800lm. The
300-family's 0.00145 diagnostic came from its NLPP chain, not the raw parent
assigned here. None of these establish a fresh gain, a gain over the newer
nonlinear blend, or additivity between families.

## A label-use boundary discovered during planning

**Do not run `__nlpp_paropt`, `__nlpp` descendants, or `__epochs` legacy jobs
in this assignment.** The pinned `tsvdx5-new.rs::run_paropt` loads all ProbeX
labels and passes them to `fit_with_nm`; its subsequent outputs would carry
additional full-development selection into the blender's scoring folds.
The legacy epoch jobs embed published fitted coefficients. Those methods
would require a separately designed nested or independently selected recipe.
Here we train the raw 300-factor parent and refit snapshot weights locally.

Pinned source/model hyperparameters remain unchanged at
`49c9b3676e6bab90fdf77e9a2da984edbfe4fe5c`. Qualifying outcomes stay zero;
query identities may remain implicit inputs. Downloaded prediction vectors
and pretrained weights never enter fresh training.

## Clock, profiling and queue decisions

Use the remaining **original** 12-hour availability, starting
2026-11-18T03:01:05Z. Do not assume the user granted another 12 hours:

- Do not begin a new full recipe after **2026-11-18T12:31:05Z**.
- Stop all owned training no later than **2026-11-18T13:01:05Z**.
- Finish export, commits and handoff by **2026-11-18T15:01:05Z**.
- If these times have already passed, publish a blocked-by-window handoff
  instead of launching. The user can explicitly extend the deadline later.

Run only one memory-heavy parent at a time. Use wrapper RAYON=8/BLAS=1 and
preserve internal pinned settings. No GPU acceleration is assumed. More RAM
does not make a single-threaded implementation scale across cores.

Start priority 1 promptly after preflight; it is the primary investment in
this window. Supervise its first initialization and two trainx epochs.
Estimate both phases plus inference/export from measured timings, using at
least a 1.5x time margin and a 30-minute export allowance. Include memory
growth and the larger fulltrain user/day population in the estimate. If the
primary clearly cannot complete by the training cutoff, stop it early,
preserve partials as a failed feasibility finding, and try the first feasible
fallback. Do not rerun the large CF-NADE preset.
Do not let an obviously infeasible primary consume the whole window.

For subsequent parents, perform a bounded feasibility profile **before**
committing the rest of the window: first inspect the dispatcher's available
test interface and use a separate profiling directory/helper with a 20-minute
cap where possible. If a faithful profile needs a helper, preserve pinned
sources and record helper/source hashes and all changed settings. Profile
outputs are not completed canonical models or blend inputs. If two complete
epochs cannot be observed within that cap, report the timing uncertainty;
use a shorter-cost fallback rather than assume the recipe fits.

Do not launch a doomed full canonical job just to learn its speed when a
separate profile can answer the question. If a canonical run must be stopped,
keep unreceipted partials; do not delete outputs or bypass the wrapper guard.
No optimizer checkpoint/resume is assumed. Snapshot prediction files are not
resumable model state.

If 300 fails its cost gate, assess the stretch model; if stretch also cannot
fit, profile and try `cfnade-12`, then `bknbrx-50`. If 300 succeeds, assess stretch next. After a stretch
parent completes, prefer its cheap compatible chains over another large
parent. After all feasible temporal work, run feasible fallbacks in that order
if time remains. For CF-NADE, a faithful two-epoch profiling command uses the
already-built `cfnade-new --test --hidden 40 --rank 12 --epochs 2 --data x`
with no `--save`, under a 20-minute cap and from the native fresh checkout.
Record the modified profiling budget separately; only the complete canonical
`--run-job cfnade-12` earns a full native receipt. Include fulltrain and all
40 epochs per phase in its time projection; do not extrapolate from capacity
ratios alone.
Skip any already-receipted job only after verifying all of its hashes.

If every remaining item fails a documented feasibility gate, stop rather
than invent another model or silently change a scientific recipe. A profile
is useful evidence, but a completed canonical model is the primary outcome.

## Setup and native commands

Use the existing idle upstream-training checkout and its venv/native data.
Fetch this branch and switch to it after preserving unrelated changes. Do
not create another fresh data tree or repeat setup that already passed.
This branch descends from your completed `research/fresh-models-12h` work.

```sh
git fetch origin research/remote-temporal-push
git switch --track origin/research/remote-temporal-push
PYTHONPATH=scripts .venv/bin/python -m unittest discover -s tests -p 'test_upstream*.py' -v
PYTHONPATH=scripts .venv/bin/python -m unittest discover -s tests -p 'test_remote12h*.py' -v
.venv/bin/python scripts/upstream_repro.py --plan-job tsvdx5-150o
.venv/bin/python scripts/upstream_repro.py --plan-job tsvdx5-300
.venv/bin/python scripts/upstream_repro.py --plan-job tsvdx4-4800lm__knns
.venv/bin/python scripts/upstream_repro.py --plan-job tsvdx4-4800lm__knnf
.venv/bin/python scripts/upstream_repro.py --plan-job cfnade-12
.venv/bin/python scripts/upstream_repro.py --plan-job bknbrx-50
```

For an existing local branch, use a normal switch and fast-forward-only pull.
Resolve no divergence by force-pushing or rewriting research history.
Confirm `validate_fresh()`, native receipts, available disk and no conflicting
trainers before launching. The existing bknbrx-100 receipt is preserved.

Use `scripts/remote12h_supervisor.py` with assignment-specific directories
under `experiments/remote-temporal/`. Its command interface was checked on the
receiving machine. Example for the primary, after checking the actual clock:

```sh
UPSTREAM_REPRO_THREADS=8 UPSTREAM_REPRO_BLAS_THREADS=1 .venv/bin/python -u scripts/remote12h_supervisor.py \
  --name tsvdx5-150o --out-dir experiments/remote-temporal/tsvdx5-150o \
  --deadline 2026-11-18T13:01:05Z --sample-every 30 --grace 30 \
  --min-memory-level 20 --max-swap-growth-mb 4096 --min-disk-gb 15 \
  -- .venv/bin/python -u scripts/upstream_repro.py --run-job tsvdx5-150o
```

Run detached with sleep prevention. Add a 30-minute heartbeat publisher scoped
to this branch's files, or commit progress manually; do not use an old queue
publisher that pushes another branch. Log per-epoch/phase times and sampled
whole-process-group RSS. Check disk during both phases: epochs, components,
train predictions and qualifying predictions can use substantially more disk
than the final compressed ProbeX vectors.

The supervisor's pressure guard is not a hard RSS cap. Additionally stop only
this assignment's process group if RSS persistently exceeds 80% of physical
RAM or resource sampling shows it cannot coexist with the host. Preserve its
logs and partials. Do not alter pinned low-memory flags to force a fit.

## Export and receiving-side evaluation

Publish `REMOTE_TEMPORAL_PROGRESS.md`, `REMOTE_TEMPORAL_HANDOFF.md`, and
`experiments/remote-temporal/`. A small queue driver/exporter and focused tests
may be added as needed. Every completed recipe must have the wrapper's native
receipt, a full hash verification of its outputs, native logs and resource
profile. Do not export an incomplete full recipe as complete.

Transfer raw finite float32 ProbeX vectors, shape `(1128164,)`, in canonical
order, with NPZ/source-NPY/content/receipt hashes and all four ProbeX metadata
alignment hashes, following the previous transfer contract. Export:

- Each parent's terminal output and every receipt-bound saved ProbeX epoch
  and component. This avoids choosing epochs using outer-fold scores. Expect
  about 25/23/27 epochs respectively only where the native dispatcher actually
  emitted them; inspect receipts rather than invent filenames. Preserve
  pipeline-kept epochs in any case (150o: 6/16/24; 4800lm: 3/13/21/24).
- Any completed chain's terminal output.
- For `cfnade-12`, terminal and receipt-bound saved epochs; pipeline-kept
  epoch 40 duplicates the terminal if numerically identical. For `bknbrx-50`,
  pipeline-kept epochs include 5/9/17/21. Deduplicate copies by content.
- Keep component predictions unclipped and label their semantics explicitly.
  Rating predictions are clipped later at ridge input, not during export.

The previous `remote12h_export.py` requires explicit `--kept` and writes into
the old assignment directory; extend it or make a new assignment-specific
exporter with tests. Do not use a zero-epoch shortcut that omits components or
silently overwrites earlier transfers. Never publish raw data, model weights
or fulltrain/qual predictions in Git. Preserve those native files locally.

The coordinator will verify alignment and provenance, reproduce the frozen
ridge/nonlinear controls, and run predeclared single-bundle and combined-bundle
comparisons. All new learned weights/stopping decisions must exclude the
scoring users through every layer. No remote ProbeX score chooses queue order,
epoch subsets or model hyperparameters. Retain negative and failed outcomes.

At cutoff, publish completed/interrupted/skipped jobs, measured cost, exact
restart needs, export locations/hashes and confirmation of process cleanup.
Fresh ProbeX scores remain exploratory development evidence, not the 10%
qualifying target or an official Netflix result.
