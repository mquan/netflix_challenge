# GPU feasibility check for an additional own raw predictor

The current source reproduction continues through original CPU CF96 and split3
jobs, alongside the larger GPU blender. To test a more substantial source-model
addition, benchmark a full-item-width denoising autoencoder with17,770 inputs,
512 hidden units and17,770 outputs on CPU and MPS, using batches64 and256.

The synthetic benchmark performs actual masked denoising forward/backward and
AdamW steps, times synchronized work after warmup, and verifies actual saved-model
inference replay on each backend. It waits for the existing shared meta fitting
lock; it does not compete with the selected GPU study. No real dataset,
prediction feature, accuracy, original-source parity or qualifying result is
claimed by this benchmark. Timing under synthetic batches does not establish
full-data epoch time.

If measured throughput supports a full-data experiment, the next required work
is to build private trainx/fulltrain CSR histories from our canonical own data,
check all source/row bindings and query exclusion, and declare a complete own
training recipe before fitting. A possible50-epoch-per-phase recipe would need
durable optimizer/RNG checkpoints and epoch-boundary yielding so it can coexist
with shorter blending work. Its actual shape, budgets, snapshot eligibility
and independent query replay must be frozen and verified before use.

This is an additional own raw-model investigation. It does not change or claim
completion of the original95-epoch CF96 recipe, nor does it supply a replacement
for the existing measured score. No downloaded numerical predictions or
qualifying outcomes enter this feasibility check.

Code: scripts/benchmark_upstream_mps_autorec.py.
Launch: experiments/upstream-mps-autorec-benchmark/launch.json.

Update: the synchronized synthetic benchmark completed at4.39x/4.46x CPU
throughput for64/256-user batches, with all saved-model replays bitwise. Both
real histories then passed independent every-row verification, and the fixed
50/50 own recipe is now fitting. Actual first-epoch training times were41.5 /
41.2 seconds, excluding checkpoint/inference overhead. These timings are not
validation accuracy. See [MPS_AUTOREC_TRAINING_PLAN.md](MPS_AUTOREC_TRAINING_PLAN.md).
