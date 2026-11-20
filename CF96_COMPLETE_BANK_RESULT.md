# Complete own CF96 prediction bank verified

The completed original 320-hidden/rank-96 model ran all95 epochs in both
training phases. All196 original prediction files now pass original
finalizer hashes, complete float32 shape/finite checks, and an independent
whole-vector clipping and paired-alias replay.

The98 pairs reduce to96 distinct pairs. Terminal SWA equals epoch95 on both
queries; raw-SGD epoch95 equals terminal raw-SGD on both queries. Every other
epoch remains available without score-based selection. Both current84-array
checkpoint states, original data/binary/source/parity identities and complete
preserved resume traces are verified and hash-bound.

This admits prediction inputs, with no fitting or qualifying-label reads.
No prediction-inference replay from the final checkpoint is claimed. The
best recovered qualifying score remains0.8603920746 /0.8613737959 and the
strict below0.85725 target on both groups remains unmet.

The original1600-factor fit continues unchanged and paired391 H128 remains
queued behind it. A subsequent experiment must compare these complete
trajectories against existing scalar inputs and fit using known data only.

[Frozen protocol](CF96_COMPLETE_BANK_PLAN.md) ·
[Complete input inventory](experiments/upstream-cf96-complete-bank/manifest.json) ·
[Independent verification](experiments/upstream-cf96-complete-bank/independent-verification.json).
