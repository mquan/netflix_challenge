# Production for the verified tree91/contextNN86 combination

The fixed91 plain-tree/plain-NN development mean completed at0.8601526901,
worse than the preceding0.8600300602. Its tree alone improved. One explicitly
fixed half mean of that tree91 vector and the unchanged contextNN86 vector
scores **0.8599745622436025** on the same1,128,164 known-ProbeX rows and462,858
users. Both families use the original five nested held-user folds. The actual
ten originating fold receipts, saved artifacts, prior full replay evidence and
prediction identities were rechecked before independent compensated scoring.
This is adaptive development evidence; it is not a recovered qualifying score.

The production recipe must preserve both originating family schemas:

| Family | Prediction columns | Ridge columns | Features plus ridge | Per-seed budget |
| --- | ---: | ---: | ---: | --- |
| Original tree91 | 91 | 167 | 228 | 873 / 746 / 855 rounds |
| Context NN86, rank16 | 86 | 162 | 223 | 19 / 20 / 20 epochs |

Seeds remain2026,7,13. Budgets are medians of the five actual originating
receipts for each seed. Fit a separate ridge on all known ProbeX rows for each
family. Preserve original tree settings, context architecture, optimizer,
training-only scaler, floating-point operations, family clipping and the fixed
half mean of the two float32 family predictions. Fit both families afresh;
reuse no different-schema full-fit predictions.

Use the existing exact paired bank builders `upstream_extension_bank.py`
(tree91) and `upstream_cf12_bank.py` (NN86). Keep the original85 recovered
bank route and all native source provenance intact. Require identical known
and qualifying row identities across both banks. Each family receives its own
immutable production plan, ridge file, bank binding, model receipts and saved
model replay. Separate fit/replay processes avoid the observed OpenMP runtime
clash. The parent plan binds both family plans before fitting; the final
receipt binds their replayed vectors and the exact float32 mean.

The existing matching86 queue remains intact until the replacement controller
is implemented and its real guards pass. Supersede it only while verified idle,
with no child or partial fit. Native uncheckpointed jobs retain priority. New
fits require measured memory headroom and a shared fitting lock. Keep the
all91 context experiment queued behind production; compare any later result
against this new development best before promoting it.

Freeze and replay the complete2,817,131-row mean before invoking a recovered
label audit. Independently score both exact source groups with compensated
summation and verify all code, data, receipt and prediction bindings. Only
RMSE strictly below0.85725 on **both** groups satisfies the goal. No qualifying
outcome, residual or earlier qualifying score selects the family combination,
feature layouts, budgets or weights. No public pretrained prediction features
are admitted. The resulting audit remains adaptive community-recovered
evidence, not official Netflix or untouched Test performance.
