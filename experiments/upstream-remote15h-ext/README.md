# Remote15h extension: claimed split3 producers for the spare window

**Claim (2026-11-20 ~06:20Z, M3 Max remote):** the producers below are being trained here.
Please don't queue them on the coordinator machine. Unstarted ones can be released on request.

The assigned remote15h work (s3-dnn24, s3-attn32, s3-rbm200e) is untouched and finishes
first. The RBM was at epoch 4 of 25+25 at 06:07Z, and these jobs use the ~9h left before
the same fixed 17:44:19Z training deadline. They run **serially after** the assigned driver
exits, under the same rules: original pinned recipes verbatim from the split3 registry, full
data/capacity/seeds/epochs, both `train_3→probe_3` and `fulltrain→qual` phases, own canonical
arrays, qualifying targets zero, and no public numerical features.

| Order | Producer | Family | Epochs/phase | Published pairs |
|---:|---|---|---:|---|
| 1 | s3-bknbrx-100w | neighbourhood (kNN regression) | 12 | terminal, ep01 |
| 2 | s3-mfrbmx-60h | MF+RBM hybrid | 12 | terminal, ep06, ep08 |
| 3 | s3-tx60o | timeSVD++ ordinal, NBR100 | 25 | terminal, ep02, ep24 |
| 4 | s3-tx50n | timeSVD++ with item features | 22 | terminal |
| 5 | s3-bk4-g100 | BellKor-style integrated (time) | 22 | terminal |
| 6 | s3-rbm100p | RBM, HT=p | 25 | terminal |
| 7 | s3-bknbrx-200t | neighbourhood | 15 | terminal |
| 8 | s3-mfrbmx-50w | MF+RBM hybrid | 12 | terminal, ep08, ep11 |

**Why these:** they're root split3 producers (no parent jobs) with both phases and qual
outputs. None appears as completed, running, queued or returned on
`research/full-upstream-reproduction` as of 06:10Z. That branch covers attn16/32/64,
dnn16t/24/32nm, nade12s, rbm200, rbm200e, tx100r and CF96. The coordinator's split3
*neural* additions were individually negative, so this window adds non-neural
families (kNN, timeSVD++, BellKor-style, MF-RBM) that are not yet in the split3 bank.
Late-queue jobs are admitted only if their reserved hours remain. Any job still training
at the deadline is terminated and marked ineligible.

**Tooling:** `scripts/upstream_remote15h_ext.py` and `plan.json`. The helper differs from
the frozen helper in only three places (shown by `diff`):
1. its PLAN path;
2. its epoch-line regex, which also accepts the unscored qual-phase `Epoch NN` lines
   printed by `lib.rs report_rmse`, while still requiring the exact 1..E trace twice;
3. its terminal==last-epoch check, which runs only when that snapshot exists, since not
   every recipe sets EPSAVE.

Plan source/data/split3/row-map hashes are identical to the frozen plan. The native root
is `artifacts/upstream-remote15h-ext`, freshly prepared, with the same Accelerate cargo
config. Exports go to the same draft release `remote15h-own-20261120` as
`<job>-export.tar.gz`, with receipts in `execution/<job>/`.
