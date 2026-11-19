# Own166 GPU study: existing development best retained

All five neural folds and all15 actual saved MPS-model replays pass. The
standalone neural arm scores0.8594144161. Its fixed half mean with the older
tree91 scores0.8593995547; with the expanded own107/50 tree it scores
0.8592892332. Each is above the current verified0.8592720613, so no new global
record or full-known qualifying refit is launched from this study.

The all59 captured-snapshot ridge diagnostic improved0.8625801964 to
0.8624919426 with independent five-fold replay. That improvement did not carry
through to the tested hidden128/rank32 nonlinear recipe. No source feature
subset was selected after seeing the scores; the complete fixed batch was fit.

The larger hidden512/rank64 recipe is a separate, already declared study on
the same393 inputs. It retains original user folds, optimizer, dropout and
stopping procedure, and must pass full saved-model replay and scoring. Any
improvement against its own control is recorded separately from a new global
record. Qualifying outcomes do not determine its weights or budgets.

Evidence: experiments/upstream-warm166-mps-context/{manifest,evaluation,
independent-verification,nn-replay}.json and
experiments/upstream-rbm50-tree-warm166-mps/{plan,verification}.json.
Best qualifying remains0.8608774626 /0.8618605526; the10% goal remains unmet.
