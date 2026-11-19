# Item-oriented own raw predictor: feasibility benchmark

The bounded-memory v2 benchmark completed all four actual saved-model replays.
MPS steps take0.0779124s /0.1493196s for item batches8/32, versus CPU
0.2498753s /0.3723881s:3.2071x /2.4939x speed ratios. Batch32 has about2.09x
the MPS item throughput of batch8. These are synthetic timings, not real-data
epoch times or accuracy. The original failed allocation-cap trial is preserved.

The next full-data recipe will use128 hidden units, batch32, bounded0.30 MPS
fraction, AdamW0.0005/0.0001, seed2026 and50 full epochs per phase. Loss covers
observed ratings hidden by50% input corruption, centered at3.6. Retain snapshots
5/10/20/30/40/50 and require actual saved-model full query replay. Use one
atomic optimizer/RNG checkpoint per phase, preserving the previous complete
checkpoint until the new pointer is durable; keep all snapshot model weights.
Prepare lossless transposed histories and independently check every original
tuple and query exclusion before fitting. Start after the current raw50/50
trajectory completes to preserve disk headroom. Data/trainer implementation
remains the next action; **real-data item-oriented training is not running**.

The following describes the benchmark protocol:

Benchmark a distinct own raw masked-denoising model with480,189 user inputs,
128 tanh hidden units and480,189 reconstructed user ratings. Each example is
an item rather than a user. The model has123,408,701 parameters; it reads no
public numerical features or qualifying outcomes. This is an additional own
recipe, not exact original native-preset parity.

The synthetic benchmark covers dense full-width forward/backward/AdamW steps,
four warmup steps and12 synchronized measured steps for CPU/MPS and item
batches8/32. Every successful run must reload its actual saved weights and
repeat inference bitwise. Throughput is not real-data epoch timing, predictive
accuracy or a completed raw training run.

The original GPU trial stopped in AdamW at its bounded3.20GiB allocation cap,
trying to allocate another236MiB. Both CPU runs completed and actually replayed
their saved models. Preserve that failure. A separate v2 benchmark retains
the same architecture/numerical recipe and uses a bounded0.30 MPS memory
fraction instead of0.15. It never disables the allocator limit. Results remain
pending until the actual v2 reports complete.

If feasibility passes, the next concrete implementation is a lossless
item-oriented CSR of the original two training histories, separate all-tuple
and query-exclusion checks, then a fixed full-data two-phase raw recipe with
durable optimizer/RNG checkpoints and actual saved-model query replays.
Serialize it after the current raw50/50 trajectory to preserve disk headroom;
declare checkpoint/snapshot storage and budgets before fitting. No actual
item-oriented raw fitting is claimed by this benchmark or plan.

Code: scripts/benchmark_upstream_mps_item_autorec{,_v2}.py.
Reports: experiments/upstream-mps-item-autorec-benchmark{,-v2}/.
