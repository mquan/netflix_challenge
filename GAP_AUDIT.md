# Gap audit: Grand Prize techniques vs. this project

Audit date 2026-10-06. Reference: **R8-B4-A2e-NN2, 0.8674131 dev CV** (77
prediction columns, about 119 tree features, LightGBM-s3 ⊕ NN2). Target ≈0.8646
dev CV, so the gap is ≈0.0028. Recent additions pay about 0.0001 each.

Sources read: every round/plan doc on this branch, plus the docs that live only on
other branches (`STRATEGY_RESET.md`, `RBM_CONTEXT.md`, `SESSION_META.md`,
`HISTORY_EXPERIMENTS.md`, `GP_COMPACT.md`, `C5_FOLLOWUPS.md`, `ADAPTIVE_RIDGE.md`,
`ENRICHMENT.md` on `main`/`bet-*`; `PAPER_IMPLEMENTATIONS.md`, `PUSH_EXPERIMENTS.md`
on `paper-implementations`), plus `src/`, `scripts/` and `src/bayes.hpp` on
`bet-b-bayes-timesvdpp`. Papers: Koren, *BellKor Solution to the Netflix Grand
Prize* (2009); Töscher, Jahrer & Bell, *BigChaos Solution…* (2009); Piotte &
Chabbert, *Pragmatic Theory Solution…* (2009). The rest come from what is known
about Koren KDD'08/'09, Salakhutdinov RBM/PMF/BPMF, Paterek 2007, Takács et al.,
Lawrence & Urtasun 2009, Sill et al. FWLS (The Ensemble), and Rendle/Zhang/Koren 2019.

## 0. Protocol differences that change how paper gains transfer

Read these before trusting any number in the table.

1. **Paper RMSEs are quiz scores from models retrained on train+probe.** BigChaos
   §3.1 gives a probe→quiz gain of 0.0030–0.0090 per predictor. Pragmatic Theory's
   predictor list shows the same thing, e.g. bk1-a1000x 0.8876 probe → 0.8794 quiz,
   and bk1-a1000x-knn1-3 0.8855 → 0.8766. Our Phase 0 measured 0.0087 for K1 and
   0.0076 for the RBM. Once that is subtracted:
   - **The "RBM deficit" (ROUND6) is mostly a protocol artifact.** R4DR at 0.8995
     probe ≈ 0.892 quiz, which matches BellKor's best frequency RBM (0.8928 quiz).
     More RBM work is not under-explored. It has reached the paper's level.
   - **The factor gap is real but only about 0.004.** B4 (0.8897 probe ≈ 0.881
     quiz-equivalent) vs. BellKor timeSVD++ with frequencies at 0.8762–0.8777. K1
     (0.8941 → ≈0.8854) vs. 0.8814 for timeSVD++ at f=200 without frequency factors.
   - Pragmatic Theory's best single predictor, bk4-f200z4-nlpp1-knn3-1, scores 0.8713
     quiz (≈0.879 probe). That is ≈0.010 better than any single column here. It is a
     chain: integrated model → polynomial post-processing → kNN.
2. **Epoch and meta-parameter choice.** BellKor and BigChaos tuned on probe and
   stopped training at the best *linear-blend* RMSE (BigChaos §3.2, Alg. 1). BellKor
   deliberately over-trained factor models and RBMs "because it helps the blend"
   (§IV-A, §VI-C). We select on the P2 holdout by standalone RMSE. So their
   over-training gains do not show up here unless the epoch is selected by blend gain
   on P2. Selecting it on dev CV would leak probe information.
3. **User-fold CV vs. the probe→qual transfer.** Winners fit blenders on probe rows
   and applied them to qualifying rows of the *same users* in the *same sessions*.
   Our 5-user-fold CV never lets a user's own probe rows inform that user's
   prediction.
   - This makes the dev metric conservative for user-specific blending: Pragmatic
     Theory's per-user linear regression, BigChaos's cluster bins, and per-user
     residual features.
   - Most of that signal is probably absorbed by retraining on probe in Stage C
     (b_u,t for shared sessions). Phase 0's c−b = −0.00018 is consistent with that.
   - A within-user random split of probe would mirror the probe/qual split, but it
     can't separate the gain from what retraining would already capture. Treat
     per-user blending as unmeasurable here, not as dead.
4. **Quiz blending (BigChaos §7) is out of scope.** It is linear blending using
   leaderboard RMSE feedback, which has no analogue without quiz labels. Several of
   the "top contributors" in BigChaos's Figure 6 / Appendix rank list were chosen by
   quiz-blend gain, so those rankings overstate their probe-blend value.
5. **The 0.0074 offset was measured on a 9-column bank** (five masked-history
   columns, K1, two kNN, one RBM). By family it was ≈0.009 for factor/kNN columns and
   ≈0.006 for masked-history columns. The current bank's marginal columns are
   different: Bayes (factor-like) and A2/A2e/MT (neural, ≈0.006). Two consequences:
   - The real offset for the 77-column blend could be ±0.0005 away from 0.0074. That
     is a sixth of the remaining gap, so re-measuring it is a high-value calibration,
     not a model gain.
   - Bayes posterior-mean columns were never rehearsed. MCMC chains retrained on more
     data can also behave differently from SGD retrains at a fixed epoch.

## 1. Technique table

Status: **Paid** = tried, kept, measured gain. **Dead** = tried, not retained.
**Partial** = a related or weaker version was tried. **Not tried**.
Δ = change in dev CV (negative is better) on the bank it was measured against.
Estimates are against the *current* bank. They are judgment calls, typically
1/3–1/2 of the paper's gain because our bank already holds most of the signal.

### Baselines and temporal effects

| Winner technique (source) | Status | Evidence | Est. gain / cost (96 GB Mac) |
|---|---|---|---|
| Time-aware baseline b_u + α_u·dev_u(t) + b_u,t + b_i,Bin(t), with c_u(t) scaling (BellKor §III-A) | Paid (inside joint) | `src/joint.hpp` (user_alpha, day_bias, item_bin_bias, user_scale, day_scale); C1 0.93492 | — |
| Frequency f_ui → b_i,f (PT discovery; BellKor §III-B, 0.9555→0.9278 baseline) | Paid in SGD joint and RBM; **absent from Bayes FM** | `item_frequency_bias` in joint.hpp; `--frequency-bias-rate` in rbm.hpp; `src/bayes.hpp` groups are user, user-bag, item, item-bag, day, item-bin, session, drift (no frequency group) | Adding an item×freq group to Bayes: 0.0001–0.0003 (see bet 2); cost ≈1 day of code |
| Baseline predictors themselves as blend columns (BellKor PQ1, two baselines in their blend) | Not tried as columns (time-baseline.bin exists; its column isn't in the bank) | bank list in `artifacts/r8/R8-B4-A2e-s3.json` | ≈0.00003; minutes. Filler only |
| Global Effects (Bell & Koren 2007; BigChaos §5.5, 16 effects) and Global Time Effects (BigChaos §5.6, 24 kernel-smoothed effects) applied to residuals of other models | Not tried | — | 0.00005–0.0002. BigChaos calls GTE "strong to minimize blend RMSE" on their bank, but our joint models and tree meta-features already cover the main ones. Cheap numpy on CPU, no GPU |
| Weekday effect (BigChaos §5.7) | Dead (in-model version) | MT4 `--week-days 7`: +0.0000729 vs MT3 (ROUND7) | ≈0 |
| PT time variables: elapsed since the user's/movie's first, mean and median rating date; absolute date | Partial | meta: user mean day, first/last; extra_meta: movie_age; contexts: day − user mean (`gbdt_blend.py`, `round4_boost.py`) | Median-date and user-first-rating offsets: ≈0.00003; minutes |
| Pragmatic Theory "non-linear envelope" (sigmoid output with a shifted inflection point) | Not tried | — | ≈0 for the blend (the tree is already nonlinear); might help single models a little |

### Factor models

| Winner technique (source) | Status | Evidence | Est. gain / cost |
|---|---|---|---|
| SVD++ with N(u) including qualifying pairs (Koren KDD'08) | Paid | README; `check-implicit`; every joint/Bayes/MT model uses train+probe+qual implicit data | — |
| timeSVD++ with p_u(t) = p_u + α_u·dev + p_u,t (BellKor eq. 14) | Paid / partial | J4 `--day-factors 128` (in bank, overfits after one epoch); K1 256f 0.894117 | — |
| Frequency-specific item factors q_i,f (BellKor eq. 16, "barely beneficial") | Partial (weak bank) | PQ2 200f: 0.8981 at epoch 6, then overfit; −0.00044 on B7, which is far weaker than ours (`STRATEGY_RESET.md`) | ≤0.0001 in SGD. Untested inside Bayes |
| Large f (500–2000) and over-trained variants (BellKor PQ2 family: 6 variants, f up to 2000) | Dead for SGD | K2 512f traced the 256f curve (ROUND6); T6/J7 | ≈0 for SGD |
| BPMF / Bayesian timeSVD++ "flipped" (Salakhutdinov & Mnih 2008; Rendle 2019) | **Paid (largest round-8 axis)** | B2 r64 −0.00089; B3 r128 −0.00036; B3 at sweep 160 −0.00012; B4 (+item-bin, drift factors) −0.00012 (ROUND8) | Rank 256, session factors and frequency group are untried (bet 2) |
| Flipped/item-side models: item-based RBM (BigChaos RBMV6), item AutoRec, item-side history twin | Dead or weak | IA1 −0.000082 (R6-IA); A1 item twin −0.0000013 (R8-CM-A1); item-bag only inside Bayes | Flipped conditional RBM: ≈0.0001; ≈1 day of CPU |
| Asymmetric / NSVD models: user represented only by the items they rated (Paterek NSVD1/2; Koren ASVD; PT Asymmetric 1/3/4; BigChaos SBRAMF) | Not tried as a standalone factor model | Conceptually covered by masked-history/A2 (user = encoded history), which pay | 0.00005–0.00015. BigChaos's NSVD1 (0.9317 standalone) still ranked 9th in its final-blend list, but that list is quiz-selected (§0.4) |
| BRISMF (Takács et al.; PT §3.10) | Partial (dominated) | J1–J6 are BRISMF plus time and implicit terms | ≈0 |
| NNMF / non-negative MF (BellKor 2007 #17; PT nnmf*) | Not tried | — | ≈0.00005 |
| MMMF (BigChaos §5.9) | Not tried | — | ≈0 |
| SVD with adaptive user factors (BigChaos §5.14 SVD-AUF; Takács "retrain user features") | Dead | ADAPTIVE_RIDGE pilot: +0.0000013 at 1/16 coverage (`main:ADAPTIVE_RIDGE.md`) | — |
| KRR on factor vectors per user (Paterek 2007; BigChaos SVD-AUF-KRR); NLPMF (Lawrence & Urtasun) | **Paid** as the local GP | GP −0.00105 (MOONSHOTS); temporal GP −0.00075 (ROUND2); on the compact bank −0.0024 (`GP_COMPACT.md`) | Saturated: J2 and implicit-geometry GPs add <0.00003 (`C5_FOLLOWUPS.md`) |
| Pragmatic Theory BK4 integrated model with low-rank time-dependent biases (single predictor 0.8713 quiz with nlpp and kNN3) | Not tried | Our closest is I2R at 0.89458 | Uncertain: 0.0001–0.0005. The SGD factor axis is saturated in our blend, but this is a 0.010-better single model. Cost: one or two weeks of implementation plus a day of CPU per run. High risk |

### Neighborhood models

| Winner technique (source) | Status | Evidence | Est. gain / cost |
|---|---|---|---|
| Learned global item-item neighborhood with w_ij and c_ij (Koren KDD'08) | Paid | N1 (−0.0034 at B4, RESULTS); I2R integrated | — |
| Time-aware neighborhood with learned β_u, e^{−β_u Δt} (BellKor §V, PQ3 0.8870 quiz) | Partial / dead at pilot | TN0/TN1 two-epoch pilots: learned decay added 0.0000003 (`STRATEGY_RESET.md`); PUSH-neighbor-temporal on I2R; fixed-λ time decay in kNN columns (paid in R4-B) | 0.00005–0.0002 if converged with frequency biases. BellKor's run used all of R(u), not a top-64 graph |
| Residual kNN on *every* model (BigChaos §5.2; PT kNN1–5) | Paid for SGD; **not tried on Bayes, A2/A2e, MT** | 13+ kNN columns on joint models carried ≈85% of GBDT gain at B6. kNN on K1 added +0.00001 (R6-K1); kNN on RBM −0.000044 (R6-RBMK) | On posterior-mean Bayes residuals: 0.0001–0.0003 (bet 5). Needs a `--train-predictions` export in `bayes`; kNN runs are ≈25 s each |
| kNN on k-fold cross-validated residuals, "correct residuals" (BigChaos §3.3, K=34) | Not tried | In-sample residuals only (`knn --base-train`). HOLDOUT single-split residual-tree pilot was dead (−0.0000932 vs. its control, worse than R5) | 0.0001–0.0003, but it costs K full retrains of the base model (5× K1 ≈ 5–8 CPU hours) |
| Similarity variants: Pearson, Spearman, set correlation, MSE correlation, factor distance (BigChaos §5.2.1; PT §3.19) | Partial | Graph uses residual cosine; variants only in power, shrink, k and time decay | 0.00005–0.0001 per variant; minutes each. Correlated with the existing 13 kNN columns |
| User-user kNN (BellKor 2007 #20; PT §3.22) | Dead alone | R8-C −0.0000435 (only paid jointly with movie tags, R8-CM-s3 −0.000125) | — |

### RBM and neural models

| Winner technique (source) | Status | Evidence | Est. gain / cost |
|---|---|---|---|
| Conditional RBM, hidden units conditioned on N(u) (Salakhutdinov 2007) | Paid | R2: −0.0031 on the 12-column bank (NEURAL_BLEND) | — |
| RBM with conditional visible units: user, user-day and frequency biases (BellKor §VI-A) | **Paid** | R6-RBM −0.00099; RBM2–5 each −0.0001 to −0.0002; seed and size diversity saturated (ROUND6) | Saturated |
| Frequency-modulated hidden-visible weights W(1+C_fj) (BellKor §VI-A, "barely improves") | Not tried | — | ≈0 |
| Day-specific hidden units (BellKor §VI-B, "not encouraging") | Not tried | — | ≈0.00005 |
| Gaussian-visible RBM (BigChaos RBMV5); PT time/decomposed/split RBMs | Not tried | — | ≈0.0001 for one more family member; ≈1 day of CPU |
| Masked/denoising history networks (AutoRec, CF-NADE; post-prize) | **Paid** | MH3 −0.00061; MT1 −0.0010; A2 −0.0006; A2b −0.00028; A2e −0.00013 (ROUND7–8) | Seed replicates saturated (MT10 +0.0000165) |
| Classification view: P(r>k) logistic models (PT §3.17) | Partial / dead | MH1 cross-entropy was worse than MSE by 0.0003; R8-A3-dist P1/P5/entropy features +0.0000657; distributional head chose 0 epochs (ROUND3) | ≈0 |

### Post-processing and blending

| Winner technique (source) | Status | Evidence | Est. gain / cost |
|---|---|---|---|
| Per-user/per-movie cubic post-processing of a model, nlpp (PT §3.15) | Not tried | — | 0.00005–0.00015. Tree features user_frac1/5 and user std already describe skew. Cheap numpy |
| Clustering post-processing (PT §3.16); per-user linear regression (PT §3.18) | Not tried; **not measurable under user folds** (§0.3) | — | Unknown under this protocol |
| GBDT blend with support, frequency and date features (BellKor §VII) | Paid | B6 −0.0013 over ridge; residual-tree refit −0.00077 (ROUND3) | — |
| Feature-weighted linear stacking with meta-features (Sill et al., The Ensemble) | Paid | Contextual ridge with 4 contexts is FWLS (`moonshot_nn_blend.contexts`) | — |
| Neural blending, 2-layer nets with log-support inputs (BigChaos §6.5) | Paid | NN2 ⊕ LightGBM −0.00022 (ROUND8) | — |
| **Latent features in the blender: timeSVD++ 20-D user/movie factors and bias terms, RBM hidden units, k-best-neighbor raw ratings, similarities and supports** (BellKor §VII-A: 0.8664→0.8661→0.8660; BigChaos §6.9–6.11: "great additional inputs", in its top-2 quiz-blend contributor PB-115) | **Not tried** | The tree has only scalar support/date/meta features. No user or movie identity or latent slicing | **0.0002–0.0006**; a few CPU hours (bet 1) |
| Ensemble NN blend, ENNBlend: 1,200 small nets on random 4-predictor subsets plus a frequency-binned linear blend (BigChaos §6.6; its #1 quiz contributor) | Not tried | NN2 + NN3 averaging gave nothing (0.8676572 vs. 0.8676503) | 0.0001–0.0003 (bet 3); CPU hours |
| Bagged GBDT (BigChaos §6.7) | Partial | 3 LightGBM seeds with bagging fraction 0.8; seed averaging −0.0001 (ROUND6) | ≈0.00005 |
| KRR blend on a probe subset (BigChaos §6.8); polynomial regression blend; support/date/frequency-binned linear blends (§6.3) | Not tried | — | 0.00005–0.0002 as a third blender in the fixed average |
| NN blend over tree-blend outputs (BigChaos PB-115/117) | Not tried | Requires nested out-of-fold tree outputs. NEURAL_BLEND warns that a naive version leaks | 0.0001–0.0002 if done with nested folds |
| Variable multiplications, pairwise products of predictors (PT §4.4) | Covered | The tree models interactions | ≈0 |
| Optimize each predictor's epoch for blend gain, and over-train (BigChaos §3.2; BellKor §IV-A, §VI-C) | Not tried | P2 selection uses standalone holdout RMSE | 0.0001–0.0003 if selected on P2 blend gain (needs P2-trained columns for the bank: the rehearsal infrastructure exists). Adding a second, over-trained epoch of the same model as an extra column is cheap where per-epoch or per-snapshot columns exist (Bayes `.s00NN.f32`; MT/A2 `--save-epochs`) |
| Metadata / content features (Pilászy & Tikk 2009) | Dead | R8-meta −0.0000422 alone (ROUND8) | — |
| Probe-refit: retrain everything on train+probe at fixed epochs, apply the probe-fit blender to qual (all three teams) | Rehearsed, not executed | Phase 0: retraining gains 0.00755; transfer loss 0.00018 (PHASE0_REHEARSAL) | Required for any test claim. Re-measuring the offset on the current bank could move the target ±0.0005 (§0.5) |
| Clipping to [1,5]; unclipped inputs to the blender (BigChaos §3.2) | Paid / check | Scores are clipped; columns are raw float32 | — |

## 2. What the evidence says about where 0.0028 can come from

- **Per-family saturation is well documented.** It covers SGD factor models, item
  kNN on SGD residuals, RBM seeds and sizes, MT seeds, metadata, user-kNN and
  distribution features. None of these will deliver 0.0003.
- **The axes that paid ≥0.0003 in round 8** were a *new learner* (Bayes MCMC) and a
  *new encoder* (A2 transformer), each first arrival paying 0.0006–0.0009. Later
  variants of each paid 0.0001–0.0004. Beyond that, the winners' unexploited levers
  are mostly on the **blend side**: latent features, ENNBlend-style diversity, and
  blend-aware epoch choice. These are cheap here because the bank is frozen.
- **Five more 0.0001 additions will not close 0.0028.** A credible path is two
  structural bets at 0.0003–0.0006, plus the blend-side bets, plus a re-measured
  offset.

## 3. Ranked shortlist (each has a plausible case for ≥0.0003)

### Bet 1: latent user/movie/neighbor features in both blenders (BellKor §VII-A, BigChaos §6.9–6.11)

- **Why it should pay.** Both blenders currently slice probe rows only by support,
  date and activity. BellKor gained −0.0003/−0.0004 from 20-D timeSVD++ factors,
  bias terms and kNN features on a 24-predictor GBDT. BigChaos's second-best
  final-blend component was an NN blend extended with SVD and RBM features. Nothing
  in this project's history tests it.
- **Cost and risk.** It costs one CPU afternoon and no GPU.
- **Design**, all features label-free and in probe order:
  1. **Factor features (16).** Read `artifacts/K1-joint256.bin` in numpy. The layout
     follows `save_joint`: u64 magic, u64 fingerprint, u32 rank, u32 epoch, then 12
     (u64 size + f32[]) arrays in the order user_bias, user_alpha, user_scale,
     item_bias, item_bin_bias, item_frequency_bias, day_bias, day_scale,
     user_factors, item_factors, implicit_factors, drift.
     - Form the effective user vector p_u + |N(u)|^−½ Σ y_j from the implicit CSR.
     - Project the user and item vectors onto the top 8 principal directions of the
       item factors, fitted on items only. That gives 8 + 8 columns.
     - Alternative: train `build/netflix joint --factors 20 --iterations 40
       --learning-rate .008 --regularization .02 --decay .97 --patience 8
       --threads 6` and take raw factors (BellKor used f=20).
   2. **Bias features (≈6).** b_u, α_u·dev_u(t), b_u,t, c_u·c_u,t,
      b_i + b_i,Bin(t), and b_i,f for each probe row (BellKor added "all individual
      bias terms").
   3. **RBM hidden features (20).** p(h|V) of a 20–50-unit conditional RBM. Train a
      small `build/netflix rbm --hidden 20 --conditional-biases …` (≈1 h), or take
      the top-20 PCA of R4DR's hidden activations. This needs a small
      `eval-rbm --hidden-out` addition to `src/rbm.hpp`.
   4. **kNN features (15).** For the top 3 neighbors of i in R(u) on
      `artifacts/r4/neighbors256.bin`: c_ij·r_uj, c_ij, log support, r_uj and e^{−|Δt|/500}
      (BigChaos Table 8). This needs an `--extract K` flag on the `knn` subcommand in
      `src/neighborhood.hpp`.
- **Run.**
  - First LightGBM, with the recorded R8-B4-A2e-s3 command plus `--extra-support
    <the ≈57 new f32 files>`, output `artifacts/r8/R8-LAT-s3`. It takes about 1.5 h
    on 5 processes. Watch memory: x grows from ≈119 to ≈176 float32 columns × 1.4M
    rows, about 1 GB.
  - Then `scripts/round8_nn_blend.py --like artifacts/r8/R8-LAT-s3.json --output
    artifacts/r8/R8-LAT-NN2 --lr 3e-4 --dropout .2 --seeds 2026 7 13 --epochs 60
    --patience 5`.
- **Ablations.** Run three arms: factors+biases only, +RBM, +kNN. If nothing else is
  ready, run factors+biases first.
- **Caveat.** User factors let the tree carve users into look-alike groups. Under
  user-fold CV that has to generalize to unseen users, which is honest and matches
  the paper. The paper's gain was measured on qual users who were also in the probe.

### Bet 2: Bayes "B5", rank 256 with session factors and a frequency group (BellKor eqs. 11/14/16 inside Rendle's MCMC FM)

- **Why it should pay.** Bayes was the best-paying axis of round 8. Every Bayes
  column lacks the frequency term BellKor credits with the largest baseline gain.
  Day-specific user factors (`--session 2`) were never run. Rendle et al.'s best
  ML10M result used rank 512, and B3's rank-128 column added −0.00036 on top of r64.
- **Code.** Add a `kItemFreq` group to `src/bayes.hpp` on `bet-b-bayes-timesvdpp`:
  one-hot item × ⌊log_6.76 F_ui⌋ bin. Reuse `frequency_bin()` and the
  session_counts from joint.hpp, linear mode by default and `--item-freq 2` for
  q_i,f. Copy the zero-init and probe-label-isolation tests that the existing groups
  already have.
- **Pilot.** Rank 16 on the 10% holdout subset, 40 sweeps, as in the `0c1f29d`
  diagnosis. Compare against lean defaults to check that the frequency group helps
  the posterior mean before scaling.
- **Full run.**
  ```
  artifacts/r8/bin/netflix-bayes-<new> bayes --data data/processed/netflix.bin
    --model artifacts/bayes/B5-full-r256.bin --predictions artifacts/bayes/B5-full-r256.f32
    --factors 256 --samples 160 --burn-in 16 --snapshot-every 16 --checkpoint-every 8
    --threads 5 --seed 29 --item-bin 2 --drift 2 --session 2 --item-freq 1
  ```
  Expect ≈2.5× B4's 180 s/sweep, so about 20 h. Skip the p2 run: B3 and B4 were
  still improving at 160 sweeps, so predeclare sweep 160.
- **Memory.** Session factors at rank 256 are ≈(#user-days ≈ 20M) × 256 × 4 B ≈
  20 GB. Pilot with `--profile` and watch swap (mps-memory-check rule). Fall back to
  `--session 1` with rank 256 if needed.
- **Gate.** ≥0.0003 on the R8-B4-A2e-NN2 protocol. Prior: 0.0002–0.0005.

### Bet 3: a third, structurally different nested blender in the predeclared average (BigChaos ENNBlend §6.6, binned linear §6.3, KRR §6.8)

- **Why it should pay.** NN2 ⊕ LightGBM paid −0.00022 because the two blenders'
  errors differ, while another MLP variant (NN3) gave nothing. ENNBlend is a
  different inductive bias: many tiny nets on random 4-column subsets plus
  log-support inputs, then a support- or frequency-binned ridge. It was BigChaos's
  single best quiz-blend component.
- **Design.** A new `scripts/round9_enn_blend.py` cloned from `round8_nn_blend.py`.
  - `--like` gives the same inputs, folds and inner split.
  - Per outer fold: N=300–600 nets, k=4 random columns + log|R(u)| + log|R(i)|,
    one hidden layer of 8 units. Train each on the inner-fit users and early-stop on
    the inner-valid users.
  - Stack their outputs with a ridge binned by the 4 frequency bins, fit on outer
    training users only. Refit and score the held-out fold once.
  - Average with fixed weights (1/3 each with LightGBM and NN2), predeclared before
    scoring.
- **Cost.** CPU only, 2–4 h with 4 processes.
- **Prior.** 0.0001–0.0003. It has the weakest standalone case on this shortlist,
  but it is cheap and reuses everything.

### Bet 4: re-measure the Stage-C offset on the current bank (protocol bet, §0.5)

- **Why it matters.** It doesn't lower dev CV. It decides whether the true remaining
  gap is ≈0.0023 or ≈0.0033, and it rehearses the Bayes and A2 retrain path that
  Stage C needs anyway.
- **Design.**
  - Extend `artifacts/rehearsal/spec-core.json` with B3/B4 (the p2 checkpoints
    already exist: `artifacts/bayes/B2-p2-r64.bin`, `B4-p2-r128.bin`) and A2e.
  - Score the p2 posterior on the `rehearsal_view` dataset. Bayes needs a
    predict-only path, `--init-model … --samples 0`, or a short continuation chain
    on the view data. The masked models use `scripts/masked_history_predict.py`.
  - Then run `scripts/rehearsal_blend.py --spec … --output artifacts/rehearsal/P0-r8`.
- **Cost.** Mostly the Bayes view chains, ≈1 day of CPU.

### Bet 5: residual kNN on posterior-mean Bayes residuals (and on A2e), the "kNN on every model" rule (BigChaos §5.2, PT kNN3)

- **Why it might pay.** kNN on SGD residuals saturated because those residuals are
  all alike. Posterior-mean residuals are less overfit, which is BigChaos's §3.3
  argument for out-of-fold residuals. PT's kNN3 "improves virtually any model".
- **Design.**
  - Add `--train-predictions FILE` to `bayes`: posterior-mean training-row
    predictions accumulated after burn-in, which needs one extra 400 MB f32 array.
  - Then run `build/netflix knn --data data/processed/netflix.bin --graph
    artifacts/r4/neighbors256.bin --base-train B4.train.f32 --base-probe
    artifacts/r8/B4-full-r128.s0160.f32 --power 1 --shrink .5 --time-decay .003
    --predictions artifacts/r8/knn256td-B4.f32 --support …`. This needs B4 re-run
    with the flag, or a resumed chain from `B4-full-r128.bin` with `--init-model`.
- **Cost.** The kNN itself takes 25 s. Re-sampling 32+ sweeps to accumulate train
  means takes ≈2 h.
- **Prior.** 0.0001–0.0003. Kill it if the kNN column moves B4 standalone by less
  than 0.002. kNN on K1 moved 0.8941→0.8923 and still added nothing to the blend.

**Not on the shortlist despite paper prominence:**
- More RBM variants: already at the paper's level once corrected for protocol.
- Frequency factors in SGD: BellKor itself says "barely beneficial", and our PQ2
  pilot agrees.
- Global/time effects: likely 0.0001.
- PT's BK4+nlpp+kNN3 chain: the highest-ceiling single model, but weeks of work with
  an unproven blend value. Revisit if bets 1–3 come in under 0.0005 combined.

Order of launch: bets 1 and 3 together on CPU, since the bank is frozen, they
reuse `round4_boost`/`round8_nn_blend` and nothing new is trained. Bet 2's pilot
can run in parallel. Bet 4 should run before the final Stage C, and its result
reframes the target.
