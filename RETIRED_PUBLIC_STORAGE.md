# Retired public-cache storage recovery

These transactions reclaim disk for our original native temporal model's
unchanged20GiB admission guard. They do not contribute features or weights
to our own experiments or scores.

The first Gram-cache archive preserved eight original files and recovered
298,172,982 logical bytes. The separate public-feature archive preserved
263 files and recovered957,053,852 logical bytes. Every file is retained
as a zstd archive beside its historical path, with original size, SHA256,
permissions, ownership and timestamps in its transaction catalog. A separate
verification reopened every feature archive and decoded all original bytes.

Only closed, untracked, unflagged, single-link files under explicit retired
public-cache directories qualify. Actual own input bindings are excluded.
The uncompressed copy is unlinked only after full decompression reproduces
its original SHA256 and size, the original file is rechecked, and its archive
and metadata are cataloged. Own datasets, training inputs, saved models,
native checkpoints and source are untouched.

The prediction-cache transaction initially stopped during schema inventory:
older public Probe vectors have1,408,395 rows. No originals were changed.
Its log/script/failure receipt are preserved. A separate v2 transaction
includes that public vector shape and uses the same exact-byte checks.
It stops at20.5GiB free space or when all eligible files are processed;
it does not lower any training resource guard.

That v2 archive is now complete:539 files,637,861,376 logical bytes saved.
A separate pass reopened and fully decoded all539 archives successfully.
It exhausted its eligible files at about20.17GiB free space, enough for
the original temporal admission guard, without reaching the20.5GiB target.

Subsequent temporal memory growth caused swap to consume the disk reserve.
Under the user's earlier authorization to remove downloaded public-bank
results, two separate transactions intentionally discarded the remaining
362 eligible untracked downloaded public qualifying prediction caches,
4,079,252,024 logical bytes. They retained pre-deletion SHA/metadata catalogs
and did not remove any earlier archives, source, canonical data or own
numerical values. These intentionally discarded caches are not lossless
archives and cannot be restored through the commands below. All original
native writers and their owners were paused while disk headroom recovered.
[Native memory recovery](TX100_MEMORY_RECOVERY.md).

Actual transaction evidence:

- `experiments/upstream-retired-public-stats-archive/verification.json`
- `experiments/upstream-retired-public-features-archive/verification.json`
- `experiments/upstream-retired-public-features-archive/independent-verification.json`
- `experiments/upstream-retired-public-prediction-archive/failure.json`
- `experiments/upstream-retired-public-prediction-archive-v2/plan.json`

Restore exact original bytes, subject to the native checkpoint disk reserve:

```sh
.venv/bin/python scripts/upstream_public_stats_archive.py --restore-all
.venv/bin/python scripts/upstream_retired_public_features_archive.py --restore-all
.venv/bin/python scripts/upstream_retired_public_prediction_archive_v2.py --restore-all
```

Restore checks the archive and decoded original SHA256 before replacing its
historical path. The verified archive remains available after restoration.
Do not remove these archives unless intentionally discarding the retired
public caches; they are the retained original bytes.
