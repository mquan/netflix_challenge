# Reuse completed own bases for two exact published KNN3 predictors

Reproduce the exact original s3-dnn32nm__knn3 and s3-nade12s__knn3 jobs.
Their original own base models already completed7/7 and8/8 phases with
independent proofs. All eight original terminal train_3/probe_3/fulltrain/qual
vectors are source-bound, fully finite and staged by hardlink into the private
source tree. This reuses completed base training while preserving every base
path/value and avoiding another1.6GB copy.

The source-resolved registry commands match exactly. Run the unchanged
Knn3Config defaults: threshold0.25, minimum10/maximum60 neighbors,
shrinkage20000, regularization0.01, mixing0.8 and baseline penalties25/10.
Use original4 Rayon/4 OpenBLAS threads with the established environment
whitelist, so model knobs cannot leak from the shell. Execute both complete
original source phases; no neighbor/item/user/data subsampling or budget cap.
KNN3 is non-iterative here: full neighbor construction occurs at model creation,
so validate both complete native builds rather than inventing an epoch budget.

Preserve original pinned source/binary/data hashes and complete own parent
lineage. Source training excludes every canonical known-ProbeX row; qualifying
targets remain zero. After each chain, require both actual native phase markers,
original default configuration and both neighbor/weight build counts. Rehash
every declared output, check complete finite query vectors and project every
known row with the original exact identity map. A separate watcher repeats
source/base/data/output and full-known projection checks before completion.
No public numerical prediction feature or recovered outcome enters fitting.

Controller71695 and independent watcher71696 are queued. Wait for the current
CPU tree and both preceding full-size neural studies to finish. Then use two
stable samples: existing physical footprints below16GiB, compressor below5GiB,
swap growth below32MiB per30seconds, disk above20GiB. Reserve8GiB, serialize
chains and preserve partial output on failure. Existing raw/native fits retain
their original capacities, budgets and ownership.

Preflight confirms both jobs are exact published output owners, all eight own
parent vectors share their original inodes, and actual predecessor processes
are detected. A phase fixture accepts both complete builds and rejects a
missing phase/build, changed neighbor capacity and changed regularization.
This is prepared/queued source reproduction, not a new ensemble or qualifying
score.

Evidence: experiments/upstream-split3-knn3-chains/{plan,
preflight-verification}.json. Code: scripts/upstream_split3_knn3_chains.py and
scripts/verify_upstream_split3_knn3_chains.py.

## Complete original source results (November20 UTC)

Both original two-phase source chains are complete and independently verified. The DNN32nm chain builds397,670 /396,169 weights across all17,770 items in its two phases; NADE12s builds413,623 /412,106. Every original source/binary/base/data/output binding is checked, every native query is finite with the complete expected shape, and every known projection matches exactly. Both qualifying target arrays remain zero; these are source completion results, not qualifying RMSE claims. Each independent report binds469 original lineage files.

The shared raw-GPU slot reservation ended after the native coordinator exited with no live native child; the final independent source proof followed20 seconds later. Recorded raw checkpoint pointers/source remained unchanged during the reservation, and the original50-epoch trajectory resumed normally. The fixed full KNN3-pair neural study is prepared under KNN3PAIR2_VIRTUAL_PLAN.md. Complete evidence: experiments/upstream-split3-knn3-chains/{s3-dnn32nm__knn3,s3-nade12s__knn3}-verification.json and verification-status.json; experiments/upstream-knn3-source-slot/released.json.
