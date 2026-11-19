# Accelerate the remaining original CFNADE96 training

The first own-training qualifying audit misses the target at0.8628067765 on
the worse group. The existing 91-predictor study and original1,000-factor source
job continue. The next performance task is to test whether the already proven
CFNADE48 sparse-row arithmetic also accelerates the full320-hidden/96-rank
trajectory. Capacity,95 epochs, seed, optimizer, SWA, source flags and data stay
fixed. This is a proposed execution-backend change, not a measured score gain.

## Preserve all current work

Do not edit the active CFNADE96/48 sources, original controllers, native trees,
checkpoints or existing fitting manifests. Build in a new private source/target
directory. The current CFNADE96 controller and native processes remain the
reference and fallback until both phase cutover proofs pass.

Reference: `experiments/upstream-cfnade-parallel/plan.json`, current source ID
`caa746c11ae0070b1615d5594e8edca924a348da88ae743b8938f8a5760cd636`,
data ID `c46870e9892edc2a8ad5a2550a146236e1fbfc8aee7c3f71422bf73ccbf12538`.
Use the actual latest committed checkpoint at capture time, not an assumed
epoch number. The existing CFNADE48 sparse-row source is the candidate kernel;
the original checkpointed96 source remains the comparison implementation.

## Prove the full-capacity kernel before any production cutover

1. Build a separate320/96 stage driver and native parity executable with the
   unchanged original95-epoch configuration and preserved Cargo.lock.
2. Compare original and candidate implementations at320/96 on the existing
   synthetic fixture cases, including active source flags. Require bitwise
   equality of every one of the84 parameter/Adam/SWA arrays and complete RNG
   state after initialization and each of seven epochs. Record actual timings.
3. If parity fails, preserve the mismatch and investigate; do not admit a merely
   close model or change precision, rank, epoch count or optimizer to pass.

## Convert checkpoint identity without changing learning state

After code/parity are frozen, capture each current complete original checkpoint
into a private destination. Rehash all84 tensor files, the state manifest,
latest pointer and owner. Preserve tensors, epoch, timestep, SWA count, RNG
seed/stream/word position and swapped-state flag exactly.

Use an explicit conversion receipt for changing only the source-implementation
identity to the candidate's actual frozen source identity. Update the private
state/owner identity and its manifest MD5/pointer accordingly. Preserve the
entire original checkpoint and its original identity as evidence. Do not claim
the converted checkpoint came from an uninterrupted original execution.

## Require actual full-data trajectory equality

For each phase, resume the private candidate and require the regenerated latest
epoch prediction to reproduce the original native NPY SHA256 exactly. Advance
the candidate just one additional epoch and exit after its complete checkpoint.
Wait for the original running reference to reach that same epoch and capture
its complete checkpoint before rolling cleanup can remove it.

Require the next-epoch native prediction hash, all84 tensor hashes, timestep,
SWA counter, RNG state and swapped-state flag to match exactly, excluding only
the disclosed source-implementation identity. This full-data check complements
the synthetic parity checks. A changed prediction or state disqualifies cutover.

## Resume through all95 epochs only after the proof passes

Use resource admission for one private trial phase at a time on the32-GiB
coordinator. Preserve the active native and development workers. After both
full-data proofs pass, retire only the owned old CFNADE96 controller through
its existing termination path and independently rehash its fallback checkpoints.
Continue the exact candidate trajectories from their verified complete states.

Full completion still requires all95 native epoch outputs and terminal vectors
for both phases, finite float32 predictions, terminal/epoch95 equality, complete
checkpoint and source/data/config bindings, and a truthful resumed-execution
receipt. Export the same-trajectory raw96r variant only after epoch95, under the
existing canonical raw-export proof. No partial model becomes production
eligible, and qualifying outcomes never enter fitting or model selection.

Any later blend candidate must pass the known-ProbeX development gate, refit
and replay its saved production models, freeze its final vector, and then
independently audit both recovered groups. The required result remains strictly
below0.85725 on both groups.
