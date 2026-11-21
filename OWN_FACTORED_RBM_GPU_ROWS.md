# Factored RBM user-batch reader prototype

This extends the [separate GPU study plan](OWN_FACTORED_RBM_GPU_PLAN.md) and
[prediction prototype](OWN_FACTORED_RBM_GPU_PREDICTION.md). It adds an array-only
reader and synthetic integration checks. No production arrays were scanned,
no actual input provenance was newly established, and no training job launched.

`scripts/upstream_factored_rbm_rows.py` accepts separately supplied training and
query arrays. It requests training ratings and masks, but never query ratings.
It checks exact user counts, native contiguous user grouping, positive-state
rating bounds and coordinate bounds. It preserves the original per-user rating
order and sorts only the unique training dates needed for bias slots.

Source `rx.rs` skips masked rows in positive states, day slots and initial
visible counts, but includes their item addresses in the conditional union.
The reader preserves that distinction. It unions all training/query addresses
per user, deduplicates conditionals and retains original query row IDs.
Query-day lookup uses `-1` when no positive training date exists. Shuffled
batches carry both local coordinates and global user/day/query indices.
Training batches reject users without positive histories; prediction supports
them. The batch's user count is never reduced by dropping history rows.

Raw arrays may be read-only memory maps. Rating validation uses bounded blocks;
selected user histories are gathered in full. Global user offsets, query row
permutation, visible counts and unique day metadata remain resident, so actual
full-data peak memory and model/checkpoint storage still require measurement
and resource admission. This is not a constant-memory or already-admitted
production loader, and it does not itself read files or verify their hashes.

Six new checks cover source order, masked-label isolation, conditional unions,
day mappings, original query order, cold users, invalid counts/grouping/masks,
and a shuffled batch through Torch CD-3 gradients, parameter updates and query
prediction against separate scalar equations. Together with the earlier13,
these are19 CPU-only synthetic tests. Independent full-source/data/split/map
verification, full raw-row replay and known-holdout disjointness remain required.

Pending work remains explicit: production file/source admission, model and RNG
initialization, epoch scheduling and durable terminal models, guarded full15+15
execution, actual MPS replay/throughput/memory checks, independent full-query
model replay, and a separately frozen known-label blend before recovered audit.
The original native1200 and all existing queues remain unchanged. No new score
or contribution to the below.85725/both-groups objective is claimed here.
