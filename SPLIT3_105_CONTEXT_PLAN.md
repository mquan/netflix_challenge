# Next fixed context iteration: own105

Retain the verified own103 development cache exactly and append both completed,
source-published split3 attention columns together: `s3-attn16` and its ep08
snapshot. The independently replayed ridge diagnostic improves from
0.8626072996291720 to0.8625974317905872. This small ridge gain is not a global
blend or qualifying score.

The new cache has181 ridge columns and241 features. All retained columns,
new clipped/centered raw transforms, known labels, users and folds were checked.
The baseline vector is bound to the independent five-fold ridge replay.
The split3 completion proof binds the exact ProbeX projection, matching labels,
and full training-data disjointness scan.

`scripts/upstream_split3_105_context.py` retains the original64-hidden/rank16
context architecture, seeds, optimizer, stopping and five nested user folds.
The242 neural inputs partition into107 signal and135 context indices, with
both new attention features in the signal set. A synthetic actual one-epoch
fit/save/reload check passed with bitwise predictions.

The reference is the strongest completed own103 arm, the standalone NN at
0.8594692563654073 by independent compensated scoring. Evaluate NN and fixed
tree/NN mean as separate candidates; promote any lower comparable overall
score without an all-fold or minimum-gain veto. The tree91 component remains
the same verified own held-user predictions.

The controller uses the existing memory admission and shared fitting lock.
It cannot interrupt the current own103 full-known production/audits. Training
begins only after that production releases the shared lock and resource checks
pass. No qualifying outcomes, downloaded numerical features or additional
snapshot selection enter this iteration.

After actual five-fold completion and saved-model replay, decide on full-known
refitting from the completed known-ProbeX comparison. Preparing this cache does
not demonstrate a qualifying improvement or satisfy the strict0.85725 target.
