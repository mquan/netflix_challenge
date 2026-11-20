# Own NN pool: full refits and production-input audit

Observed November 19, 2026 Pacific / November 20 UTC. The ten-percent goal
remains unmet: best recovered qualifying groups are 0.8605620602 and
0.8615662023. The adaptive development best remains 0.8591267885.

The complete known-row frozen-model comparison found no material aggregate
loss from the previously disclosed production input realization differences:

| Frozen model | Original development inputs A | Paired production-known inputs B |
| --- | ---: | ---: |
| Neural item154 | 0.8593837349 | 0.8593816465 |
| Tree item154 | 0.8600069538 | 0.8600074580 |

All five original folds replayed exactly for each model. This is input
sensitivity through frozen models, not a newly trained B-realization CV
study, and does not explain the qualifying gap. Reports and the separate
compensated scorer are in `experiments/upstream-production-parity-audit/`.
The original combined process completed its neural portion, then failed
when Torch and LightGBM loaded conflicting OpenMP runtimes. Preserve that
failed entry point as historical evidence; run tree inference separately.

All five new full neural refits completed their initial worker-level query
replays at 07:04:44 UTC. All five independent serialized-pipeline admissions
completed at07:15:53 UTC. The refits used the already
declared architectures, seeds and median epoch budgets. They fit the
existing paired production-known B bank and matching qualifying B bank,
with the original own native additions. The original A-development score
is not a matched B-trained CV claim. The base full NN154 will be reused.
No qualifying outcomes enter these fits or their stopping budgets.

Each recipe passed a 513-row actual-source assembly and exact float32
scaling smoke check. This small preflight does not replace full lineage
validation or complete saved-pipeline replay. Completed workers write
candidate receipts under `experiments/upstream-nn-pool6-full-refit/`.

Adversarial review identified operational and verification gaps in the
initial frozen executor. Active scripts remain unchanged. A separate
watchdog checks original worker/queue/declaration hashes and terminates
an identity-checked orphan worker if its supervisor disappears. Its
recorded-PID approach has a brief launch-before-status race; fix the
supervisor itself before reusing this execution design. A separate
admission queue rechecks frozen execution hashes after completion and
requires the new independent verifier to reopen serialized ridge weights,
scaler and all actual neural models, then replay every known/qualifying
prediction. It also validates original expected implementation hashes,
the production B manifest files and known row identities. A further mandatory
gate, `verify_upstream_nn_pool6_original_bank.py`, validates every original
paired-bank code/source/data expectation and requires all five frozen full-fit
plans to contain those exact expected digests. This closes the executor's
overwriting of a current bank-code hash even though the actual bank code is
currently unchanged. Initial worker
receipts alone are insufficient for admission. No model promotion or
qualifying audit is authorized by an initial completion receipt alone.

Both admission gates passed. At07:22:22 UTC the assembly froze the equal
six-NN pool, the fixed25%tree154/75%pool candidate and one known-only
seven-family calibration. Independent verification across all127 simplex supports
and exact candidate arithmetic passed before the separate recovered audit.
The completed audit found no meaningful qualifying gain; see
`NNPOOL6_QUALIFYING_RESULT.md`. The primary candidate is the known7 calibration; the two fixed candidates
are predeclared controls, without selecting coefficients from their audits.
Immediately before outcome access, the audit rechecks the original plan,
calibration, frozen vector and preaudit proof hashes. Negative tests reject
changes to either plan/calibration before any prediction or outcome access.
The solver also recovered a known seven-weight optimum, and the exact-vector
checker rejected a one-ULP corruption. These preflight tests do not replace
the full real-data gates.

Apparent calibration fit is not nested CV, and recovered audits
are adaptive rather than official or untouched Test results.

The remote attention32 producer has returned and passed the frozen v3
local transfer/provenance checker, adding its two complete own prediction
pairs. It has not yet earned a new ensemble accuracy claim. Original
remote exponential RBM200 and its claimed nonneural extension remain
separate from this local full-refit queue.
