# Round 12: re-measure the Stage C offset on the current bank's families

Branch `claude-stage-c-offset` (worktree `.claude/worktrees/claude-stage-c-offset`),
started 2026-10-13 from `claude-bayes-session-factors` (548ff92). This is
GAP_AUDIT Bet 4.

## Why

The goal is 0.85725 *test* RMSE. We only measure probe dev CV, and Phase 0
(PHASE0_REHEARSAL.md) mapped one to the other: test ≈ dev CV − 0.0074. So the
target is dev CV ≈ 0.8646, and the gap from R9-LAT-A2h (0.8668706) is ≈0.0023.

That offset came from a 9-column bank: five masked-history columns, K1, two kNN
columns and an RBM. It contained no Bayes column and no transformer, and these
two families now carry the bank. Round 11's error analysis also shows that 63%
of probe rows (67% of squared error) have no same-day training rating. Stage C
retrains on probe, which gives most qualifying rows exactly that recent context.
How much a column gains from retraining therefore depends on its family:
0.009 for factor/kNN columns and 0.006 for masked-history columns in Phase 0.

## Design

Same procedure as Phase 0, with two families added to its core bank:

| Column | fit (P2) and a (original probe), model trained without P2 | b (retrained on all training ratings) |
|---|---|---|
| Core 9 | Phase 0's `artifacts/rehearsal/A-*` | Phase 0's b columns |
| B4 (Bayes timeSVD++ rank 128, item-bin 2, drift 2, seed 13, 160 sweeps) | **new chain** `R12-B4-union` on `artifacts/rehearsal/union-p2-original-probe.bin` | `artifacts/r8/B4-full-r128.s0160.f32` |
| A2h (transformer, 1536 steps) | `A2h-select.select.pt` (trained on train − P2, epoch 12 selected on P2) scored on P2 and on the view dataset | `artifacts/masked/A2h-refit.refit.e12.f32` |

- **Bayes needs a union dataset.** Its prediction is a posterior mean
  accumulated during the chain, so a saved state cannot re-score a second
  dataset. `scripts/rehearsal_union.py create` writes a dataset with P2's
  training, implicit arrays and header mean, whose probe is P2 rows followed by
  original probe rows. One chain then gives both columns from the same
  posterior, and `split` cuts them apart. The binary, seed and settings are
  B4's (`artifacts/r8/bin/netflix-bayes-0c1f29d`). Threads go from 4 to 8, which
  only changes Hogwild ordering.
- **A2h is scored with the code that trained it.**
  `scripts/masked_sequence_predict.py` imports the trainer from the A2
  worktree. Its `masked_sequence.py` and `masked_history.py` hashes match
  A2h-select's recorded `source_sha256` and `base_source_sha256`. Check: the P2
  score must reproduce the recorded epoch-12 holdout RMSE, 0.867402.
- **Blend.** `scripts/rehearsal_blend.py` (Phase 0 recipe) on three specs:
  core + B4, core + A2h, and core + B4 + A2h. Report a − b (retraining gain) and
  c − b (dev CV vs. Stage C analogue) next to Phase 0's 0.00755 / −0.00018.

Probe labels choose nothing in a or b: the A2h epoch comes from P2, the B4
sweep count was fixed in advance (160), and the blender is fit on P2.

## Log

- 2026-10-13 07:40: union dataset built (1,407,566 P2 + 1,408,395 original
  probe rows; 97,664,546 training ratings). `R12-B4-union` launched (8 threads).
  A2h predictions on P2 and the view dataset launched on MPS.
- 2026-10-13: A2h checkpoint reproduces its P2 score exactly (0.8674023 vs.
  recorded 0.867402). On the original probe it scores 0.90005 (a column) vs. the
  retrained refit's 0.89014 (b column), a standalone retraining gain of 0.0099.

## Results

Each test-score mapping is dev CV − (a − b) − (c − b).

| Bank | fit (P2, in-sample) | a | b | c | a − b | c − b | dev CV → test |
|---|---:|---:|---:|---:|---:|---:|---:|
| Phase 0 core 9 | 0.84309 | 0.88350 | 0.87595 | 0.87577 | 0.00755 | −0.00018 | −0.0074 |
| core 9 + A2h | 0.84054 | 0.88094 | 0.87316 | 0.87287 | **0.00779** | −0.00029 | **−0.0075** |

core + A2h fold a − b: .00791 .00835 .00735 .00745 .00787 (Phase 0 core:
.00760 .00827 .00690 .00735 .00762). Adding the transformer leaves the offset
essentially unchanged, +0.0001. Report: `experiments/r12/P12-core-A2h.json`.
