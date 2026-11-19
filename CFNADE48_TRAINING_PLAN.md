# Additional public CFNADE48 reproduction

The pinned dispatcher has a separate `cfnade-48` preset:160 hidden units,
rank48,95 epochs, with the canonical defaults including batch8, learning
rate0.0006, first-layer multiplier1.4, weight decay0.008, ordinal weight1,
RMSE weight0.1, context normalization0.5, milestones64/72 with gamma0.9,
and SWA starting at18. The existing320/96/95 source runs continue unchanged.

This additional full preset uses two independently initialized source-driver
phases, trainx→ProbeX and fulltrain→qual. Both must finish all95 epochs and
exit successfully. No partial snapshot is eligible for blending. The unchanged
generic source driver saves every epoch and the final prediction. The runtime
configuration is checked against the pinned preset definition. No source
hyperparameter, rank, epoch count or logical data split is reduced.

The already verified sparse-row backend is reused for this new model; there
is no migration of the existing CFNADE96 workers. Its original24-case parity
passes, and a separate8-case suite at160/48 also passes initialization plus7
epochs, all84 state arrays/RNG, direct/grouped/query predictions. The source
checkpoint protocol and generated array serialization are byte-identical to
the previously tested checkpoint implementation. Dependencies retain pinned
versions. All source/input/binary/configuration hashes are frozen in
`experiments/upstream-cfnade48/plan.json`.

New workers run at nice10, with3GiB constructor reservations per phase.
Admission requires all other native/root-venv worker groups plus reservations
below24GiB and free disk above28GiB. A sustained total above25GiB or disk below
16GiB stops only a new CF48 phase, retaining its exact optimizer/RNG checkpoint
for resume. Existing source fits have priority. Global RSS counts actual live
native and Python groups once, including the existing96 phases and final
blender workers. Models/checkpoints/output are isolated in the sparse-row
candidate tree (`preds_preset48`, `preset48-checkpoints`); current source trees
and primary prediction scans are unchanged.

Completion requires95-epoch checksum-bound checkpoints, all84 tensor checksums,
all95 snapshot vectors and final finite float32 vectors of the correct sizes,
and a successful native exit for each phase. The paired receipt discloses the
original source reference command and actual independent driver commands.
Source/input hashes are rechecked before final publication. Further known-user
fold evaluation and a separate blend integration remain required before any
qualifying-goal attribution. No qualifying outcomes or pretrained prediction
features enter fitting. Starting this model does not establish goal success.

Targeted Python checks cover two-phase constructor admission, global RSS group
accounting, and rejection of partial epochs or changed completion markers.
The active best development score remains0.8602161957664950; both recovered
qualifying groups still need independently verified RMSE below0.85725.
