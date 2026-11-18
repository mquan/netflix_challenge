# Missing-predictor screen and first selected fresh replay

The diagnostic screen is complete: 126 released base-bank columns absent from the fresh saved-output bank were tested individually, plus 26 groups sharing a native training root, against the frozen 0.8663200739003126 fresh context control. All additions refit the final regression in the existing five user folds, with lambda 1000, an unpenalized intercept, native per-column input clipping, and output clipping to [1.05,4.95]. Context interactions remain restricted to the original fourteen fresh predictions.

These are **exploratory released-vector diagnostics**, not independently achieved scores. The same development rows select the candidates, so ranking is not selection-adjusted evaluation. Released epoch blends may incorporate historical development-fitted coefficients; those columns are flagged and must be replaced by fresh saved epochs and fold-local fits in an independent bank. No released vectors were added to the fresh prediction artifacts.

| Diagnostic arm | Conditional development RMSE | Gain versus fresh control | Native cost finding |
| --- | ---: | ---: | --- |
| `tsvdx5-1000__nlpp__knn3` | 0.864288110 | 0.002031964 | Four new jobs; at least 156.28 GB for base factors/neighborhood matrices/datasets |
| `tsvdx4-4800lm` grouped outputs | 0.864680554 | 0.001639520 | Six new jobs in the full dependency union; base minimum 35.35 GB |
| `tsvdx5-400__nlpp__knn3` | 0.864720029 | 0.001600044 | Four new jobs; base minimum 64.96 GB |
| `tsvdx5-300__nlpp__knn3` | 0.864868095 | 0.001451979 | Four new jobs; base minimum 49.74 GB |
| `tsvdx5-150o` grouped outputs | 0.864976740 | 0.001343334 | One parent; minimum 26.91 GB before other model buffers and overhead |
| `cfnade-96` grouped outputs | 0.865289104 | 0.001030969 | One 95-epoch parent; native time and peak RAM unmeasured |
| `bknbrx-100` grouped outputs | 0.865540850 | 0.000779223 | One five-epoch parent; native time and peak RAM unmeasured; pinned eight-thread Hogwild is nondeterministic |
| `rbmx2-150__knnf` | 0.866270849 | 0.000049225 | Cached parent and factors; selected for a cheap fresh replay |

Memory numbers use decimal GB. The machine has 32 GiB / 34.36 GB physical RAM. These are source-derived **lower bounds**, not measured native peak allocations. In particular, 150o's 26.91 GB lower bound does not establish that it fits safely after all remaining allocations. `SparseUD` counts were independently reproduced from metadata: 17,280,041 user/day pairs in trainx+probex, and 18,136,482 in fulltrain+qual. Non-low-memory Tx allocates TWO dense user/day factor arrays, which is why factor count alone badly understates its cost. The lower bounds include user/item factors, item frequency factors, applicable dense neighborhood matrices, and a subset of mandatory dataset arrays; they omit other buffers, indices and overhead.

The complete resolved explicit and implicit dependency order is saved for each candidate. The NLPP arm includes its parameter-search job, not just the base and neighborhood steps. Local parent reuse requires a native receipt and verification of every recorded local output; an imported ProbeX-only artifact does not qualify as a reusable native parent. `recipe-costs.json` records all locally measured prior job times and explicitly leaves unmeasured new-job times/RAM unknown. It does not invent a numerical gain-per-hour ranking where no runtime was measured.

Five released names do not resolve to pinned native producers: `tsvdx5-1200lm__epochs`, `tsvdx5-400__knn3`, `rbmx2-500__knn-w`, `rbmx2-500__knn-m`, and `rbmx2-10r__knn3`. Their diagnostic scores remain available, but their recipe bundles are marked unresolved. Similar names are not silently substituted.

The first fresh replay is complete. `rbmx2-150__knnf` reused the receipt-verified local `rbmx2-150` parent, training residual predictions and both 750-dimensional item-factor matrices. No expensive parent was retrained. The exact pinned job took 62.71 seconds; the timed wrapper took 63.48 seconds and reported 6.47 GB maximum resident memory. This is the command's resource accounting, not a separately sampled per-process peak profile.

Adding its fresh rating prediction as one global feature yields **0.866286461848646**, versus 0.8663200739003126 before the replay: a gain of **0.0000336120516666**, with improvements in all five folds. The bank now has fifteen fresh model/chain outputs, 42 unique saved snapshots and five components: **62 prediction columns**, plus the same five context main effects and 70 interactions, for **138 regression design columns including the intercept**. The original fourteen control reproduced at 0.8677119362283469 and remains unchanged.

The replay does not numerically match the released vector: prediction RMS difference is **0.0561464591401858**. Its clipped base RMSE is 0.902769796 versus the fixture's 0.902641589. The historical base fixture is unavailable, so the parent/factor provenance contribution cannot be isolated here. No claim of numerical equivalence or floating-point-only explanation is made. The fresh measured blend gain is smaller than the released-vector diagnostic predicted and remains a valid fresh exploratory development comparison.

Evidence lives in `experiments/upstream-missing-screen/`: frozen screen inputs/arms in `manifest.json`, every score and paired fold change in `evaluation.json`, complete bundle dependencies and known/unknown costs in `recipe-costs.json`, metadata counts in `memory-metadata.json`, the pre-execution choice in `selected-replay-plan.json`, and the exported fresh vector, native log/receipt and fresh-only blend evaluation in `replay/`. Out-of-fold blend predictions remain in ignored `artifacts/upstream-missing-screen/`.

Reproduce the diagnostic screen, resource/dependency audit, and completed replay evaluation:

```sh
PYTHONPATH=scripts .venv/bin/python scripts/upstream_missing_screen.py
PYTHONPATH=scripts .venv/bin/python scripts/upstream_recipe_costs.py
PYTHONPATH=scripts .venv/bin/python scripts/upstream_selected_replay.py
PYTHONPATH=scripts .venv/bin/python -m unittest discover -s tests -p 'test_upstream*.py' -v
```

Use `upstream_recipe_costs.py --recount-memory` only to recompute the metadata union counts; otherwise their input hashes are checked. The evaluation command never starts training. Native training remains available through `upstream_repro.py --run-job rbmx2-150__knnf`, which verifies and skips its completed receipt.

Next: profile a bounded feasible new-family candidate, starting with `bknbrx-100` or CF-NADE, before committing to a larger queue. Preserve the exact pinned thread/configuration behavior and record time and RAM; unknown runtime is the remaining selection evidence. The highest-ranked dense temporal recipes exceed physical RAM even before all overheads and should not be launched unchanged. No large queue or other-machine worker is running.
