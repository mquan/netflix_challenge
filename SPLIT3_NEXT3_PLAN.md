# Next exact original split3 producers

Reproduce three still-missing published producers from the pinned source:

| Producer | Original phases | Original parameters | Fixed published pairs |
| --- | --- | --- | --- |
| s3-attn64 |12/12 |D64, seed7, EPSAVE1; source default12 epochs |terminal and epoch5 |
| s3-rbm200 |25/25 |H200, LR0.025, EPSAVE1 |terminal |
| s3-tx100r |22/22 |F100, REG_I0.012, REG_U0.08, EPSAVE1 |terminal and epoch3 |

All five selected pairs are exact declared outputs in the public metadata-only
coverage audit. No public numerical features are loaded. Each original command
must equal the source-resolved registry command, including its original four
Rayon and four OpenBLAS threads. The attention default12 is separately checked
in the pinned dispatcher source, without adding a new EPOCHS override.

The isolated source/output tree reuses the exact pinned fit binary and verified
split3/fulltrain/zero-target qualifying data. The fixed canonical known-ProbeX
row map and complete training-disjointness proof remain bound. No ingest job
that imports recovered qualifying labels is executed.

Preserve complete12/12,25/25 and22/22 native traces, every declared output,
all native side outputs and all factors. Project each fixed published probe_3
vector onto every known-ProbeX row. A separate watcher checks source/binary/data,
both full traces, every output hash, finite complete vectors, zero qualifying
targets and every known projection before marking a producer complete. Source
availability or queue admission is not trained-feature or score completion.

Controller61826 and independent watcher61827 are live. Admission uses two stable
memory samples, existing native/meta physical footprints below16GiB, compressor
below5GiB, swap growth below32MiB per30seconds and disk above20GiB. Reserve8GiB
for the next native model and full-phase data. Jobs are serialized; preserve all
partial artifacts on any failure. Existing CPU tree and raw GPU work continue.
No capacity, data or epoch budget is reduced to fit the queue.

Evidence: experiments/upstream-split3-next3/{plan,preflight-verification}.json.
Code: scripts/upstream_split3_next3.py and
scripts/verify_upstream_split3_next3.py. No new blend/qualifying result is claimed.
