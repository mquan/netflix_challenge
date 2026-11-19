# All90 MPS neural study and expanded-tree combination

The predeclared fixed half mean of the expanded own107/50 tree and larger
own107/all90 MPS neural model scores **0.8592720612630157**, improving the
preceding verified mean0.8593661639. All1,128,164 known-ProbeX rows and462,858
users retain the same five nested held-user folds. Four folds improve and one
worsens; lower aggregate RMSE is promoted.

All15 actual saved neural models reload and replay bitwise on the Apple GPU.
The tree retains its15 actual saved-model replays. The combination checker
rehashed all ten originating fold receipts and model artifacts, verified row
identities and assembled fold predictions, and scored the stored float32 mean.
A separate stored-mean and compensated full-row score check agrees exactly.
The fixed weights were committed while the neural study was still fitting.

The larger neural model alone scores0.8593871254. Its mean with the older
tree91 scores0.8593755806; both are worse than the newer tree/NN combination.
The GPU study's static preceding reference was NN50, so its internal improvement
flags are not a claim to beat every subsequently completed experiment.

This is a separately declared float32 MPS recipe: hidden128/rank32,334 inputs,
device-specific dropout RNG, unchanged optimizer and nested stopping procedure.
Its five folds fitted in about7.5 minutes, before replay and independent scoring.
This is actual full-data timing; the earlier3.36x benchmark used synthetic data
and does not establish a matched full-data CPU speedup.

The selected production recipe preserves heterogeneous features: the CPU tree
uses294 inputs (50 supports plus ridge), and the GPU NN uses334 (all90 supports
plus ridge). Both retain107 own raw prediction columns and the same183-column
paired ridge. Per-seed budgets come from medians of actual originating folds.
Both families must fit on all known labels and pass actual full saved-model
replay before freezing the fixed50/50 mean and accessing recovered outcomes.
The controller waits for the independently checked paired90 bank and memory.

The small production integration check passed real fit/save/load/inference
replay for both widths in separate fresh processes, as production uses. A
single-process trial encountered conflicting OpenMP runtimes; it produced no
production result. Full-data production and qualifying checks remain pending.

Current qualifying best remains0.8609266298 /0.8619141823; both miss0.85725.
No qualifying outcome or downloaded numerical prediction feature enters this
development fit or combination. Scores remain adaptive development/recovered
audit evidence, not official or untouched Netflix Test.

Evidence: experiments/upstream-latent90-mps-context/{manifest,evaluation,
independent-verification,nn-replay}.json;
experiments/upstream-rbm50-tree-latent90-mps/{plan,verification,
independent-mean-check}.json; and
experiments/upstream-rbm50-tree-latent90-production/{declaration,
integration-check,launch}.json.
