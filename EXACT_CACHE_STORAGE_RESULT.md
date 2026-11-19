# Preserve every cache path while sharing exact duplicate storage

Four inactive cache pairs were independently identified from immutable manifest
hashes, then compared byte-for-byte in full before any filesystem change:
source-extension versus extension-context design/features, and context-blend
versus voting-blend design/features. All candidate files had zero open handles.
Only each redundant target inode was atomically replaced with a hardlink to its
existing byte-identical canonical file. Every original path and value remains.

The producer verifies equal hashes, file lengths, permissions, owners and
filesystem device, excludes changed/open candidates, fsyncs directory updates,
then rehashes both retained paths and every original manifest. A separate
checker rehashes every complete file again, verifies shared storage and NPY
shapes/dtypes, and confirms original permissions/owners and manifest hashes.
Distinct caches, original datasets, predictions, model weights and checkpoints
were not removed or changed. Training and scoring recipes are unchanged.

Logical duplicate bytes shared: **4,981,972,736**. Observed free-space increase:
**4,981,575,680 bytes**, ending at27,947,999,232 free bytes at the receipt time.
Concurrent training/checkpoint writes affect free space, so this observed net
increase is not an isolated extent measurement. The additional headroom helps
the queued complete native recipes and raw snapshot retention proceed.

Evidence: experiments/upstream-exact-cache-links/{plan,receipt,
independent-verification}.json. Source: scripts/upstream_exact_cache_links.py
and scripts/verify_upstream_exact_cache_links.py. No new accuracy claim.
