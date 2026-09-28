# Second round: temporal kernels and learned local experts

All work continues in `/private/tmp/netflix-moonshot` on `moonshot-local-kernels`.
The first validated improvement is published at commit `1e4f386`. Original
training files and processes remain untouched. New experiment names are unique.

## Fixed reference and evaluation

The reference is the previous six frozen components **plus all three first-round
GP columns**, scoring **0.8850493143** in the original five-context C++ blender.
A new model must improve this reference, not merely reproduce the first gain.
Pilot users match round one: deterministic 10% by user, 141,157 probe rows.
Pilot analysis uses a cheaper three-context blend (intercept, history size,
movie popularity), so its absolute RMSE is not comparable to full-run results.
Promotion uses the original five-context C++ blend on the whole probe.

## Bet A: persistent taste plus temporary taste

The first GP ignores time when borrowing residuals. Temporal biases in its
source model need not capture temporary *movie-specific* preferences.
The new covariance is

```
K_total((i,t),(j,s)) = K_movie(i,j) * [a + (1-a)*exp(-abs(t-s)/30)]
```

This is a sum of a persistent GP and an independent transient GP, with
`a=0.5`. A purely transient model (`a=0`) is an ablation. All other GP settings
stay fixed. Both kernels remain positive semidefinite by sum/product closure.
The nearest 32 *training* movies are selected with this same covariance.

| Pilot addition to prior GP ensemble | Pilot CV RMSE | Delta |
|---|---:|---:|
| No new column | 0.8872026391 | — |
| Purely transient, noise 0.1 | 0.8863214527 | -0.0008811863 |
| Persistent + transient, noise 0.1 | **0.8862758046** | **-0.0009268345** |
| Implicit geometry, noise 0.1 | 0.8869913264 | -0.0002113127 |

Each improves all five pilot folds. Promote persistent+transient first.
These are development comparisons; the pilot was used to select the promoted
architecture. Users outside the pilot provide a useful consistency check but
are not an untouched final test because of prior checkpoint selection.

### Full-probe temporal result

The full-probe persistent+transient GP reaches **0.8922935424** alone, versus
0.8978688248 for the frozen source model. It adds a useful signal beyond the
first GP family:

| Original five-context full-probe ensemble | CV RMSE |
|---|---:|
| Round-one reference | 0.8850493143 |
| Add one temporal GP column (noise 0.1) | **0.8842984317** |
| Add all three temporal GP columns | 0.8842971518 |

The one-column gain is **0.0007508826 RMSE**, improving all five folds.
Outside pilot users it is 0.0007579788. The approximate paired user-cluster
95% interval for change in MSE is [-0.00144226, -0.00121488]. Inference took
104.2 seconds on one CPU thread, niceness 15. These intervals retain the
selection and shared-estimator caveats documented in round one.

**Retain one temporal column.** The extra two buy only 0.0000013 RMSE, so the
one-column treatment is the reference for the local-expert comparison.

## Bet B: geometry of movie selection

Use the source SVD++ model's **implicit-feedback movie vectors** for the GP
kernel instead of its explicit rating vectors. This tests whether the geometry
of the movies users choose to rate contains a distinct local signal. It is
not a full missing-not-at-random model. It gives a smaller pilot gain; defer
full computation until after the temporal test. CLI `--implicit-mix` also
supports a convex combination of the two kernels, but that combination was
only tested for numeric/label isolation, not scored on Netflix in this round.

## Bet C: learn separate taste axes inside overlapping movie groups

The kernel bet freezes the geometry. The larger bet is to **learn new local
latent representations**. Cluster the frozen item embeddings into eight
spherical k-means groups, route each item to its two best groups with soft
weights, and train an eight-factor residual model separately in each group.
Each user gets an independent preference vector for every group.

The source joint model remains a frozen mean function. Every SGD target is a
training-rating residual against that mean. Probe labels only report scores;
the run always executes the specified six epochs and emits every epoch's
predictions. Final-epoch comparisons are specified before training. The source
model still carries the original probe checkpoint-selection bias.

A global 16-factor residual model is the control: both models evaluate 16
active latent dimensions per rating. The local model has more total parameters,
so this tests the package of locality and sparse extra capacity, not locality
at identical parameter count. Both use seed 20260928, learning rate .03 with
.9 decay, factor regularization .05, bias regularization .2, and identical
shuffled user/rating traversal. The groups=1 route is a true global model.

The design is inspired by local/global matrix models such as
[GLOMA](https://ojs.aaai.org/index.php/AAAI/article/view/10752) and
[mixtures of matrix approximations](https://proceedings.neurips.cc/paper/2017/hash/3dd48ab31d016ffcbf3314df2b3cb9ce-Abstract.html).
It is a residual, item-routed prototype rather than a reproduction of either
paper. Published benchmark gains are not forecasts for the original probe split.

The implementation writes per-epoch prediction columns, not a reloadable local
model checkpoint; rerun the deterministic training command to regenerate them.
The numeric and leakage tests cover both global and overlapping routes.

### Full-probe local-expert result

Fixed epoch six, with the one-column temporal ensemble as reference:

| Addition | Standalone model RMSE | Ensemble CV RMSE |
|---|---:|---:|
| None | — | 0.8842984317 |
| Global residual factors (1 group × 16) | 0.8984542466 | 0.8842337043 |
| Local residual factors (8 groups × 8, 2 active) | 0.8973776647 | 0.8842285534 |
| Both residual models | — | **0.8841835475** |

Local factors improve the standalone score more than the global control, but
have almost the same contribution to the ensemble: a difference of only
0.0000052 RMSE. The combination adds 0.0001149 versus the temporal ensemble;
all five folds improve. This is **not evidence of a major advantage from local
experts**. Archive the tested configurations and keep their columns available;
do not expand this architecture on the strength of these results alone.

Across the round, the measured ensemble improves from **0.8850493143 to
0.8841835475**. Most of that gain is temporal GP conditioning. None of these
bets achieved the proposed 0.005 step-change threshold. The next architecture
bet remains an ordinal model conditioned on both user and item histories,
which changes the representation and training objective more substantially.
No GPU training was launched while the original agent occupied it.

Final prediction artifact: `artifacts/L1-both-blend.f32`. This is cross-fitted;
its `.fit` companion is in-sample and must not be reported as CV. Local models
are evaluated at the preselected sixth epoch, not at their best observed epoch.

## Reproduction

```sh
make test-moonshot-gp moonshot-local
python3 tests/test_moonshot_local.py
export VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MOONSHOT_PYTHON=/path/to/python-with-numpy
nice -n 15 bash scripts/moonshot_gp_bets.sh
nice -n 15 bash scripts/moonshot_local_bets.sh
```

Experiment runners reject reused names. Use fresh names/output prefixes when
repeating. Models and data are read from the same frozen inputs as round one;
see `experiments/moonshot-inputs.sha256`. Source hashes, exact commands and
line timestamps are captured by `scripts/experiment.py` for each run.
