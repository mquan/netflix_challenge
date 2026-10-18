# Netflix paper implementation review

Worktree: `paper-implementations` at `/private/tmp/netflix-paper-impl`, based on
`551864a`. The main `score-iteration-2` worktree has uncommitted experiments and
was left intact. The initial source review has now been followed by a local BAR
implementation and full-Probe experiments, described below.

## Verdict

The 2023 [BAR paper](https://arxiv.org/pdf/2304.14209) offers a compact,
interpretable **representation** of users. It does not report a winning
held-out Netflix score. Its claim that the 16/32-bit model can get below the
winner's 0.856 test RMSE refers to error on the observed ratings used to fit
the model. The paper says prediction was secondary, says its probe error is
several percent higher, and compares that probe error with individual RBMs,
not the winning ensemble (Section III, Figures 3–4).

The [2009 BellKor writeup](https://www2.seas.gwu.edu/~simhaweb/champalg/cf/papers/KorenBellKor2009.pdf)
describes only part of the final winning system (Introduction). It reports
component RMSEs on the Quiz set and combines time-aware biases, factorization,
neighborhood models, RBMs, and hundreds of predictors in a blend. There is no
single simple formula there that gives the final score. Its use of labeled
Probe rows for model fitting is valid for its Quiz/Test evaluation; those rows
must instead be removed when Probe itself is the evaluation set (Section II).

## BAR source audit

The authors' [code](https://github.com/seandeyo/BAR) was inspected at commit
`855c7404c02f55072acbb0e9c8d45be55fa608cf`.

| Stage | What the source does | Consequence for a Prize comparison |
| --- | --- | --- |
| [`matrix.c`](https://github.com/seandeyo/BAR/blob/855c7404c02f55072acbb0e9c8d45be55fa608cf/matrix.c) | Converts every known rating to a dense user × movie character matrix; drops dates. | The published input includes labeled Probe ratings. There is no Probe exclusion. |
| [`sort.c`](https://github.com/seandeyo/BAR/blob/855c7404c02f55072acbb0e9c8d45be55fa608cf/sort.c) | Computes movie and user averages and ranks movies using all observed ratings. | Probe labels influence preprocessing and selection. |
| [`bits.c`](https://github.com/seandeyo/BAR/blob/855c7404c02f55072acbb0e9c8d45be55fa608cf/bits.c) | Fits user bits to the selected movies; `testpred` computes error over nonzero observations in its input. | Its `.test` file is training error, despite the filename. |
| [`weights.c`](https://github.com/seandeyo/BAR/blob/855c7404c02f55072acbb0e9c8d45be55fa608cf/weights.c) | Fits each movie's weights using all its observed ratings, then scores those same ratings. | Even Probe ratings outside the selected movie subset are used in the final weight fit. |

The Probe interpretation above follows from the published code and the
paper's description of the Probe as part of the rating data. The code has no
hold-out path or separate Probe/Qualifying prediction command. It would need
an adapter and retraining with Probe labels excluded to produce a comparable
local development score.

The published implementation is expensive despite the compact *final* model.
`matrix.c` allocates 480,189 × 17,770 `int` ratings: about 31.8 GiB, then
writes about 7.95 GiB of rating characters. In `bits.c`, six dense replicated
`double` tensors (`w`, `x`, `wA`, `xA`, `wR`, `xR`) alone require about 50.8 GiB
for 148 movies and 16 bits, or 101.7 GiB for 32 bits, before allocator and
other arrays. The paper's Section II-B notes a sparse variant but says its
reported results use the standard dense algorithm. This is a practical reason
to port the method to the existing CSR dataset before attempting a full run.
The upstream `cc -O3 -fopenmp -c bits.c` build fails on stock Apple Clang
(`unsupported option '-fopenmp'`). Follow-up inspection found Homebrew GCC and
libomp already installed, so this is a compiler configuration issue, not a
missing runtime. The local implementation uses standard C++ threads.

## BellKor and this worktree

Koren's 2009 paper describes time-changing movie and user biases, rating-day
frequency effects, timeSVD++, time-aware neighborhoods, conditional RBMs, and
GBDT blending. The paper reports a 0.9278 Quiz RMSE for its frequency-aware
baseline (Section III-B), and describes GBDT blends from 454 and 75 predictors
with 0.8603 and 0.8606 Quiz RMSE (Section VII-A). These are component/blend
Quiz scores, not the undisclosed Test labels in that paper.

This repository already has related implementations:

| Paper component | Local counterpart | Fidelity |
| --- | --- | --- |
| Time-aware baseline and timeSVD++ | `src/netflix.cpp`, `src/joint.hpp` | Joint temporal, implicit and factor training exists; coefficients and optimization are local experiments. |
| Learned item neighborhood | `src/neighborhood.hpp` | Approximate graph and frozen temporal baseline; not a full reproduction of the integrated BellKor model. |
| Conditional RBM | `src/rbm.hpp` | Related architecture trained without Probe labels. |
| Predictor blend, including GBDT | `src/ensemble.hpp`, `scripts/gbdt_blend.py` | Five user folds for blender evaluation; base epochs still selected on all Probe labels. |

The latest recorded local development result at this worktree's base commit is
**0.8804020935 Probe RMSE**, from a 15-column GBDT blend (`RESULTS.md`, B6).
This is neither a Quiz nor a private Test score. The main worktree contains
newer uncommitted experiments, which are intentionally absent from this
isolated branch.

## Local implementation and experiments

`src/bar.hpp` adds `bar` and `eval-bar` commands using the existing CSR data.
Training movie means, user residual means, movie selection, user-bit fitting,
and final movie regression all exclude Probe labels. Checkpoints are selected
by subset training RMSE; Probe labels are used only after the model is fitted.

The dense RRR method retains replicas for missing pairs. Four flat float32
arrays replace the upstream six double arrays, and reflected values are
accumulated directly. The 59-movie/8-bit run allocates 3.38 GiB for replicas.
All-movie regression aggregates rating statistics by user bit pattern and
adds ridge 0.001 for numerical stability. This, float32 storage, and a
deterministic seed are explicit differences from the upstream implementation.

There is also a paper/source discrepancy: Equation (2) admits every prediction
within half a star of the observed residual, so the nearest projection is the
identity inside that interval. Upstream `bilinproj` instead moves such points
to the interval's center. The port supports both via `--projection paper`
(default) and `--projection upstream`. Its root solver uses the analytic
derivative of the bilinear constraint with safeguarded Newton iterations.
Bits are extracted by rounding concurred values, matching upstream
`storebits`. Neither mode is claimed to reproduce the published scores exactly.

Full-data experiments, using all 480,189 users and selecting 59 movies
from training ratings (8,579,888 observed ratings in that selected matrix):

| Run | Configuration | Training RMSE after movie refit | Full Probe RMSE |
| --- | --- | ---: | ---: |
| BAR1 | 8 attributes, 59 movies, 200 iterations, paper interval projection | 0.9036484883 | 0.9834745089 |
| BAR2 | 8 attributes, 59 movies, 1,000 iterations, upstream projection; best training checkpoint at 350 | 0.8807776576 | 0.9741851128 |

The BAR mean-only baseline scores **0.9879307111** on the same Probe. BAR2
improves that by 0.0137455983, but is well behind the existing local ensemble's
0.8804020935. Its optimization loop took about 890 seconds on four threads.
The paper projection and upstream projection were run for different budgets;
these results do not isolate projection choice as the sole cause of the
difference. At the shared 200-iteration checkpoint, subset training RMSE was
0.9679087360 for the paper rule and 0.9366192452 for the upstream rule.

Both final RMSEs clip predictions to `[1,5]`; subset training RMSE used for
checkpoint selection is unclipped, following the upstream evaluation. Raw
float32 predictions remain available for independent scoring and blending.
The full Probe has 1,408,395 ratings. Commands, source hashes and complete logs
are in `experiments/BAR*.json`, `.log`, and `.events.jsonl`.
The independent Python scorer `scripts/score_probe.py` confirms the full-Probe
scores; results and prediction hashes are in
`experiments/BAR-independent-scores.json`. Both saved models reproduce their
prediction columns byte for byte when reloaded through `eval-bar`.

`tests/test_bar.py` verifies projection constraints, movie regression normal
equations, model reload equality, independent scoring on a fixture, and that
altering Probe labels leaves trained parameters and predictions identical.
The existing smoke, joint-model and timeline tests also pass.

## Further comparison

The 8-attribute/59-movie configuration is one small configuration studied in
the paper. These runs do not evaluate its stronger 16/32-attribute models on
148 movies, and cannot establish a performance ceiling for BAR. A future
comparison can increase that capacity and test whether a BAR prediction
column improves the existing user-fold blend. An untouched Quiz/Test audit
would be needed for a winning-score claim.

The [2026 revival repository](https://github.com/gtakacs/netflix_prize)
provides another useful check: its published reference ensemble uses 302
predictors and reports 0.856716 Quiz / 0.857684 Test RMSE, illustrating again
why the two splits must be identified separately. These are the authors'
reported reference results, not measurements reproduced in this worktree.
