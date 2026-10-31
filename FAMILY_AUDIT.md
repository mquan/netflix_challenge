# New-family audit (STRUCTURAL_PLAN lane 1)

Written 2026-10-31 by Claude, after the week-0 scoping showed that blend gains
now come from **new model families** (≈0.0007–0.001 each in the ridge proxy)
and that within-family quality converts poorly (slopes: Bayes 0.14, GP/local
0.10, A2 0.028, others ≈0). This is reading only; no compute was used.

## What we already have

| Family | Columns | Leave-family-out (proxy) |
|---|---|---:|
| A2 sequence transformer | A2, A2b, A2e, A2f, A2h | 0.00173 |
| Bayes MCMC factorization machine | B2, B3, B4 | 0.00127 |
| Masked-history networks (MH/MT, CF-NADE-motivated) | MH3, MT0–MT9 | 0.00079 |
| GP / local nonlinear residual models | gp-full ×3, gp2, local-global, local-local | 0.00067 |
| Conditional RBMs | 8 columns | 0.00047 |
| Integrated models | I2, I2R, neighborhood64 | 0.00024 |
| kNN on residuals, user-kNN | ~25 columns | 0.00021 |
| AutoRec | 3 columns | 0.00009 |
| SGD factor models (joint, T-series) | ~11 columns | 0.00004 |

Already covered or closed, so not counted as new: CF-NADE-style distribution
heads (became MH/MT), the winners' 2009 techniques (GAP_AUDIT.md §1: every
row paid, dead, or estimated ≤0.0003), GNN/IGMC local paths (LP: nothing beyond
A2h), the item-side twin (A1), ordinal Bayes (E5), choice objectives (SO1, H9)
and metadata.

## Candidates, ranked by expected error-structure novelty

Ranking criteria: (a) how differently the model can err from the families
above, (b) evidence that the mechanism pays here, (c) cost on the M3 Max.

| Rank | Family | Why it may be new | Evidence here | Cost |
|---|---|---|---|---|
| **1** | **Local low-rank ensemble (LLORMA-style)**: many low-rank models, each fit with kernel weights around an anchor (user, item) pair; prediction is a kernel-weighted mix | Replaces one global factorization with *locally different* factorizations. Our only local family is a per-user GP on frozen embeddings, not local *factor learning* | The GP/local family has the second-highest quality slope (0.10) and LOFO 0.00067, so local nonlinearity pays. MOONSHOTS proposed local experts; never built | CPU C++: weighted SGD per anchor on kernel-restricted rows. Roughly 1–2 days of code; 50–100 anchors at about 5–20 min each |
| **2** | **Feature-based GBDT base predictor**: a tree model on per-(user, movie, date) features (neighbour ratings of the target movie, item/user factor vectors, similarity aggregates, support, time), trained on ratings, not residuals | A different learner class (axis-aligned piecewise-constant) as a *base* model. All base models today are smooth factor, kernel or neural models; trees appear only in the blender | No direct test. HR's honest residual trees on A2h features added ≈0, but those were residual correctors on a strong model, not an independent base column | CPU, Python (LightGBM). 1–2 days of features; trains in hours on P2. Must respect the SO1/E3 Python-idle rules |
| **3** | **Discrete latent-class / co-clustering mixtures** (Bayesian co-clustering, user rating profile, aspect models): users and movies belong to soft clusters with cluster-pair rating distributions | Discrete, multimodal structure with per-cluster rating *distributions*. It errs very differently from continuous factors | Distribution models (RBM) pay as a family (LOFO 0.00047) despite weak standalone scores | CPU C++, Gibbs or EM. About 2 days of code |
| 4 | Global Effects / Global Time Effects on residuals of each model (BigChaos §5.5–5.6) | Cheap correctors | GAP_AUDIT estimate 0.00005–0.0002 | Hours, numpy. Filler only |
| 5 | Closed-form linear item autoencoders (EASE/SLIM) on ratings | Learned dense item-item regression | Overlaps kNN (LOFO 0.0002) | Hours. Low priority |

**Also worth one test:** OrdRec-style user-specific rating thresholds inside the
Bayes model (Koren & Sill 2011). The Bayes family has the highest quality slope
(0.14), and per-user thresholds model *how* each user uses the 1–5 scale,
which E5's fixed thresholds did not. Treat it as a quality upgrade (lane 2 of
the revised ranking), not a new family.

## Proposed first three builds

1. **LLORMA-style local low-rank (CPU, C++)**, starting now. It is the
   strongest prior.
2. **Feature-based GBDT base predictor (CPU, Python)**, in parallel when Codex's
   Python jobs allow (STATUS.md in-flight table).
3. **Discrete co-clustering mixture (CPU, C++)** after 1's pilot reads out.

The GPU stays with E3, then lane 2's choice/long-context transformer.

## Pilot protocol (fixed before any score)

Each family pilot trains on the P2 holdout (T − P2) and predicts P2. It is
scored with the shared `scripts/p2_candidate_gate.py`: bank11 + N1 + control
duplicate vs bank11 + N1 + candidate, fold 0 for fitting and selection as in
SO1.

**Pilot gate: ≥0.0010 P2 gain.** This is stricter than earlier gates, because
P2-bank gains have shrunk 5–20× in the full bank (N1: −0.0012 on P2 →
−0.00017 full).

Pass → a dev run (trained on T, predicting the probe) and the full-bank test
(≥0.0005, all five folds). Fail → close the family without hyperparameter
sweeps, except one pre-declared size or kernel-width pair if a pilot is within
0.0003 of the gate.

## L1 pilot: local low-rank ensemble — fixed before any P2 score (2026-10-31)

**Code.** `netflix llorma` (`src/llorma.hpp`), built separately as
`build/netflix-llorma` (the shared `build/netflix` is untouched).
- Local models are weighted SGD factorizations of the *baseline-only*
  residuals of a frozen joint model (its biases and time effects).
- Kernels are Epanechnikov on arccos distance in a geometry made of the top-k
  principal components of the joint model's effective user vectors and item
  factors (`--geometry-pca`, computed in-binary by subspace iteration).
- The bandwidth is set per anchor so its support holds a fixed fraction of
  users and of movies (`--coverage`).
- Weights are rescaled to mean 1 per anchor for the SGD step. Prediction is
  baseline + the kernel-weighted mix of local predictions, falling back to
  the baseline where no anchor covers a row.
- `tests/test_llorma.py` (5 tests) covers: learning beyond the baseline on a
  rank-2 fixture, byte-for-byte reproducibility for fixed threads, baseline
  fallback and support for uncovered rows, PCA geometry and quantile
  bandwidth, and rejection of bad settings and of a mismatched joint model.

**Why these settings.** LLORMA's fixed bandwidth of 0.8 on K1's
256-dimensional factors covered only the anchor row itself; at 1.2 it covered
about 2,000 rows. Hence the low-dimensional geometry and quantile bandwidth.
A 2-anchor timing pilot at PCA-16 / coverage 0.2 covered 4.3M and 8.2M rows
in 14 s and 27 s per anchor. It read only its own log, not the gate.

**Pilot run.** P2 holdout `artifacts/validation/train-matched-20260927.bin`.
Joint model `artifacts/rehearsal/A-K1.bin` (K1 trained on T − P2). Settings:
100 anchors, rank 20, PCA-16 geometry, coverage 0.2, 15 epochs, lr 0.01,
reg 0.02, decay 0.95, seed 2026, 4 threads. Output `artifacts/l1/L1-p2.f32`
plus support.

**Gate.** `scripts/p2_candidate_gate.py`, protocol SO1-P2-fold0-matched-pair-v1.
- Bank: bank11 + N1. Control is the duplicate of `A-K1.p2.f32` (already in
  bank11), so the primary comparison measures L1 beyond the bank.
- **Pass: ≥0.0010.** One pre-declared retry is allowed at coverage 0.4 if
  the result is within 0.0003 of the gate. No other sweep.

### L1 result: closed (2026-10-31)

`experiments/L1-p2-gate-result.json` (shared `scripts/p2_candidate_gate.py`,
plan `experiments/L1-p2-gate-plan.json`; receipts bind the prediction hashes).

| | P2 |
|---|---:|
| L1 standalone | 0.89239 (its baseline-only base: 0.98590; 20,057 rows uncovered) |
| **Primary: bank11 + N1 + K1-dup vs + L1** | **+0.0000023**, 95% ΔMSE interval [−0.0000063, +0.0000141] |
| Gate | ≥0.0010 gain → **fail**; outside the 0.0003 retry window, so no coverage-0.4 run |

The local low-rank ensemble is a reasonable standalone model, but its
information is already in the bank. The trainer completed all 100 anchors
(log). The original shell receipt step failed only because zsh treats
`status` as read-only; `experiments/L1-p2.json` records this.

**Lesson for the plan.** The scoping figure of ≈0.0007–0.001 per family was
measured on families already in the bank, which are there because they
paid. That is survivorship bias. A typical *new* family should be expected to
add much less, as L1 did. The remaining candidates (feature GBDT base model,
co-clustering mixture) need a sharper reason to err differently before
compute is spent on them.
