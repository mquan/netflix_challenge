# Preserve the completed study's exact RAM inputs while freeing its owner

The full matched study has completed both arms and all ten independent
saved-pipeline replays. The original130-feature,1,760,494-row voting table
is still held by its idle RAM owner. Its0.878GiB footprint makes the next
full16GiB neural reservation exceed the24GiB limit by about0.457GiB.
Do not shrink the model, release unpreserved inputs or change fitted code.

Archive every original little-endian float32 byte in feature-major order,
in ten lossless zstd parts of thirteen complete features each. Require the
entire payload and each feature's full SHA256 to match the original native
generation proof. Decode every archived byte independently, checking part
hashes and the complete original915,456,880-byte payload hash. Keep source,
native-generation and completed-model proofs unchanged.

Before releasing RAM, require the original complete two-arm/ten-fold proof,
no active matched-study fitting/replay consumer, the same idle source owner,
all archive decode checks, and enough disk for original checkpoint reserves.
Use the owner's existing explicit release-file contract only after a durable
archive completion receipt. No source changes, signals or numeric edits to
the original owner. Keep the archive and provide verified exact restoration.

A restored regular-file table must reproduce the original complete payload
hash, shape and dtype. A supplemental reader may replace the original live
owner transport with this exact file, retaining all original numeric block
operators and source/data checks. Historical live-owner proofs stay intact;
archive provenance explicitly supplements that ephemeral transport gate.
No new model or score is claimed by archival.
