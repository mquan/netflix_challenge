# Six eligible own families: new verified qualifying best

One known-only convex blend of six fully replayed own model families scores
**0.8605620601761576 on source0** and **0.8615662023089099 on source1**. Both
improve the preceding five-family scores0.8605716961956835 /0.861583222505447.
The worse group gains **9.5469%** over Cinematch; both groups still miss the
strict **0.85725** target. These are adaptive community-recovered audits, not
an official or untouched Test result.

The newly eligible full tree154 was added because its predeclared fixed-half
known nested-CV result improved. All five previous eligible families remain
in the declared six-family fit. No recovered outcome selected families,
coefficients, hyperparameters, snapshots, or model budgets.

| Family | Known-only fitted weight |
|---|---:|
| tree50 | approximately0 |
| NN50 | 0.19703208916758494 |
| NN90 | 0.13950334506593706 |
| NN109 | 0.1348533099359374 |
| NN154 | 0.2867135416579108 |
| tree154 | 0.24189771417262973 |

Weights are nonnegative and sum to one, fitted by least squares on complete
known OOF vectors. The apparent known-fit RMSE **0.8590482079240619** is not
nested cross-validation. The separately verified development best remains
**0.8592012938198756** from the original tree154/NN154 fixed half mean.

All six originating development/full saved-family models retain actual replay
proofs and exact source/data/model bindings. The full2,817,131-row qualifying
vector was frozen before label access. A separate verifier recomputed
compensated sufficient statistics, enumerated all63 nonempty simplex supports,
reconstructed both stored known/full float32 vectors exactly, and scored every
canonical row in both recovered groups (1,408,789 /1,408,342).

The first six-family fit stopped verification before qualifying-label access:
SLSQP's original objective tolerance1e-14 accepted coefficients differing by
0.00005975 from the independently solved optimum. Its script, plan, frozen
vector, failed checker and diagnostic are preserved. A separate namespace
retains the same families, objective, uniform initialization, optimizer,
iteration cap, and original predictions, tightening only numerical tolerance
to1e-17. It passes the unchanged independent coefficient accuracy criterion.
No recovered outcome informed this numerical correction.

Evidence: [independent final verification](experiments/upstream-own6-tight-calibration/independent-verification.json),
[declared precision correction](experiments/upstream-own6-tight-calibration/plan.json),
[known-only weights and source lineage](experiments/upstream-own6-tight-calibration/calibration.json),
[pre-audit frozen vector identity](experiments/upstream-own6-tight-calibration/frozen.json),
[original optimizer diagnostic](experiments/upstream-own6-calibration/optimizer-diagnostic.json),
[full tree154 fixed-half result](ITEM154_TREE_RESULT.md).
