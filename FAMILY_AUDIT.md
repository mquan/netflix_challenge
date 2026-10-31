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
