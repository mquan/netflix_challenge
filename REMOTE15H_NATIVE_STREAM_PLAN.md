# Preserve complete own remote query outputs with bounded storage

The remote branch now publishes a hashed two-part archive of the complete
native outputs of its three original assigned producers: DNN24, attention32
and exponential RBM200. They retain original8/8,12/12 and25/25 phase budgets.
The previous compact imports contain only canonical known and qualifying
queries; the full1,760,494-row native holdout remains necessary for later
matched larger-population studies.

Stream both exact release parts in order without storing their2.95GB
compressed payload locally. Verify every part byte count and SHA256,
every declared tar member's complete plaintext byte count and SHA256,
safe member paths, complete member inventory and original producer receipt
output bindings. Materialize all96 native probe/qualifying query vectors
(878,916,288 bytes), original metadata and binaries. The full training
prediction members are hashed but remain preserved in the original remote
release rather than materialized locally. Do not discard any own output.

Require disk floor18.75GiB plus both original CF checkpoint staging reserves
before import. Bounded stream buffers and one-vector-at-a-time checks must
avoid holding a complete archive or prediction bank in memory. Preserve
partial evidence on failure and never overwrite an existing destination.
Only declared small AppleDouble metadata may be ignored; no unlisted data,
unsafe paths, symlinks or duplicate members may enter the destination.

Independently recompute the complete canonical map, original training/query
disjointness, all native vector shapes/finiteness and every previous compact
known/qualifying value from the full native query vectors. Admit no new
ensemble or score from transport. Keep the running own352 study frozen;
any new predictor batch requires a separate known-only declaration.
