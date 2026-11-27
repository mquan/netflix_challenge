# Public-bank baseline comparison

The published `cross3` score, the local public-bank blend, and the fresh-training effort answer different questions. The local audits use the same recovered group names as the pinned upstream label file.

| Result | Recovered source0 RMSE | Recovered source1 RMSE | Final fit |
| --- | ---: | ---: | --- |
| Upstream `cross3`, pinned manifest | 0.855883 | — | Global ridge fitted using simulated rounded feedback from all qualifying labels |
| Local 619-column stack | 0.85615421 | 0.85719035 | Contextual ridge fitted on development ratings |
| Fresh Own17x + CF96 chains, context and CatBoost, November 24 | 0.85956670 | 0.86059917 | Blend and correction weights fitted on known ratings |

The upstream README's example reports 0.855944. The manifest at the same pin gives 0.855883 for the current `cross3` recipe. The latter is 0.00027121 below the local source0 result. Upstream forms its ridge statistics on all 2,817,131 qualifying rows using an offline simulation of rounded RMSE feedback, then scores the `is_test=0` rows. We verified that the local frozen audit's label file is byte-identical to the upstream file. These figures compare the same recovered scoring group under different fitting protocols. They give no estimate of the gap on an untouched holdout. [Pinned upstream manifest](https://github.com/gtakacs/netflix_prize/blob/49c9b3676e6bab90fdf77e9a2da984edbfe4fe5c/ensembles.toml#L46-L88) · [Upstream fit and evaluation](https://github.com/gtakacs/netflix_prize/blob/49c9b3676e6bab90fdf77e9a2da984edbfe4fe5c/src/bin/ridge.rs#L1928) · [Local audit](experiments/QUAL-target-final-result-20261116.json) · [Fresh audit](experiments/OWN17X-cf96-context-tree-20261124.json)

The column sets and the regression differ as well. Both use three source splits. Upstream selects its split-specific model groups and fits one weight per rating column with lambda 1000 and an unpenalized intercept. Its feedback blend clips inputs to [1, 5] and outputs to [1.05, 4.95]. The local stack uses an audited selection of 536 released base columns, 77 released Probe/ProbeX-fitted intermediate columns and six local columns. It takes only two old-split intermediate blends, both FWLS, while `cross3` also selects old-split GBM and MLP outputs. The local final ridge uses five context terms and clips output to [1, 5]. The 0.00027121 gap includes all these differences. [Upstream ridge](https://github.com/gtakacs/netflix_prize/blob/49c9b3676e6bab90fdf77e9a2da984edbfe4fe5c/src/bin/ridge.rs) · [Local fit and import policy](experiments/QUAL-public619-meta-ridge-freeze-20261116.json)

## What the local columns add

I refit the 613 released columns with the original five contexts, the same 1,128,164 development rows, and the original fixed alpha of 0.001. I removed the six local columns from every context block in the saved development statistics and solved the resulting ridge system. I replayed the 619-column control and checked it against the frozen predictions within 0.0000005 per rating. The public-only predictions were frozen before reading recovered labels for scoring.

| Fixed recipe | Source0 RMSE | Source1 RMSE |
| --- | ---: | ---: |
| 613 released columns | 0.8565870646 | 0.8576170265 |
| 613 released + six local columns | 0.8561542082 | 0.8571903481 |
| Improvement from the six columns | 0.0004328564 | 0.0004266784 |

The six-column package moved source1 below 0.85725 in this comparison. It contains five local predictors and NN385, which is locally fitted on released predictions. The comparison estimates the package's contribution at the original penalty. It leaves individual contributions and a separately tuned public-only penalty open. [Ablation receipt](experiments/QUAL-public613-fixed-recipe-ablation-20261124.json)

## What the fresh-training effort is missing

The existing matched single-split diagnostic held the 1,128,164 ProbeX rows, five user folds, final ridge, and clipping fixed. It measured the following final-layer CV scores.

| Bank | Prediction columns | ProbeX RMSE |
| --- | ---: | ---: |
| Early fresh bank | 14 | 0.86771194 |
| Released base bank | 137 | 0.86048531 |
| Released base + FWLS outputs | 147 | 0.85661076 |
| Full released stack | 188 | 0.85632324 |

The early fresh bank trailed the released base bank by 0.00723. The released intermediate layers improved on the released bases by a further 0.00416. Fitting and scoring the full released final ridge on the same ProbeX rows lowers RMSE by just 0.000156 relative to its final-layer CV. The imported intermediate columns retain upstream's earlier training and selection, so this audit measures the final layer only. FWLS outputs can themselves consume GBM and MLP outputs. [Matched gap audit](https://github.com/mquan/netflix_challenge/blob/711d041/experiments/upstream-expansion/gap-audit.json)

The early audit does not quantify the remaining gap for today's larger own bank. The November 24 runs give a more relevant signal. The 65-predictor expansion gained about 0.00049 on both recovered groups. The subsequent CatBoost correction gained about 0.00005–0.00006. On the native five-fold control, CF96 alone gained 0.000089, while CF96 with its residual-neighborhood outputs gained 0.000564, with gains in all five folds. [November 24 audit and score history](https://github.com/mquan/netflix_challenge/blob/2f19abd/OWN_PROGRESS.md) · [Matched CF96 report](https://github.com/mquan/netflix_challenge/blob/5dfb573/experiments/remote-context-blend/RESULTS_FULLBANK_CF.md)

The recovered audit reproduced the Own17x control and measured the CF96 gain. Fitting separate weights for the native contextual ridge and the CatBoost residual improved all five ProbeX folds over the CF96 refit by 0.000096 pooled. This is non-nested development evidence because the stack reuses previously fitted predictions. The selected qualifying vector was frozen before opening recovered labels and scored independently. [Experiment receipt](experiments/OWN17X-cf96-context-tree-20261124.json) · [Replay script](scripts/own17x_cf96_followup.py)

| Fresh blend | Source0 RMSE | Source1 RMSE |
| --- | ---: | ---: |
| Own17x control | 0.85984307 | 0.86084907 |
| Own17x + CF96 and four residual-neighborhood columns | 0.85965742 | 0.86067301 |
| Own17x + CF96, context and CatBoost corrections | **0.85956670** | **0.86059917** |

The final row improves on the previous CF48 and CatBoost best by 0.00008601 on source0 and 0.00007107 on source1. Source1 is 9.6484% below the 0.9525 reference and 0.00334917 above the 0.85725 target.

For fresh training, the most informative comparison is a package-level addition to the current fixed native-bank control. Test the base and its residual outputs together, evaluate the upstream-style blend layers on those same user folds, and carry qualifying scores forward only after freezing the predictions. The published top-level `cross3` score is useful for checking the historical reconstruction. The 0.85725 target on both recovered groups remains the fresh-training benchmark.
