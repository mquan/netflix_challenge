# Grouped prediction candidate: verified, not promoted

The active own-model best development result remains0.8602161957664950.
Qualifying success is still unverified. Current production and training trees
are unchanged by this separate experiment.

The existing mapped backend invokes its eviction accounting from individual
prediction rows. The public parallel factor model also uses the default row
prediction helper. A separate candidate reuses the existing Tx per-user
prediction implementation and moves grouped eviction to once per completed
user. The direct and subscore APIs retain eviction. Training update code,
source hyperparameters, factor counts, epochs, logical chunks and random
number ordering are unchanged. All modifications are disclosed in
`experiments/upstream-grouped-storage/storage.patch`.

Parallel checks passed24 cases (factors0/1/17/1000/4000/16000,1/16 logical
chunks, original sequential epochs enabled/disabled), initialization plus20
epochs, at2/4 physical workers and heap/mapped storage. Every learned/cache bit,
direct/grouped/component prediction and actual library generation output
matches the original source. Final signatures match across all four execution
modes. The serial288-case suite also passes all four heap/mapped/cached/lazy
modes through4800 factors and5 epochs, including the72MB persistence check.
No claim about the feasibility or quality of full-size16000-factor training
follows from these synthetic checks.

Timing uses a separate driver:200000 synthetic query rows,1000 factors and20000
padded users, giving80MB user allocations that cross the64MB eviction threshold.
One freshly trained synthetic model is evaluated by the actual library in row
and grouped modes. Three alternating-order repeats have identical prediction
bits and hashes in both heap and mapped runs. On this concurrently busy host,
median mapped inference is0.261126125s row versus0.221339125s grouped (1.1798x).
Heap is0.130990250s row versus0.221000500s grouped (0.5927x).

The result is a tradeoff, not evidence of a universal or full-data speedup.
The workload concentrates queries on32 active users and does not reproduce the
full Netflix user/day distribution or memory pressure. The queued1000-factor
training watcher58553 and its original backend are therefore retained. Their
source proof and zero-qualifying sentinel were independently revalidated after
this work. Promotion requires more representative timing and a separate
explicit source-bound training launch. Existing uncheckpointed fits continue.

Reproduction: run `scripts/upstream_grouped_storage.py`, then
`scripts/upstream_grouped_benchmark.py` in the project venv. These write only
`artifacts/upstream-grouped-storage` and its own report directory. Scripts and
proof/source digests are now frozen; use a new experiment for further changes.
