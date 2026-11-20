# Completed original1600 build cache preserved in a verified archive

The original1200 producer requires20.5GiB free before preparation/build;
available space was just below that threshold. The independently completed
original1600 producer's unused build intermediates were preserved in a
lossless archive, recovering209MiB net and raising free space above20.5GiB.

All1,269 regular single-link files under its exact nonsymlink
`target/release/build` directory were inventoried with original metadata and
SHA256. No open users were observed. Every archived member was then decoded
and fully rehashed against the original before any file was removed.
The archive, parent directory and recovery evidence were fsynced before
unlink. All original directory structure remains. The pinned executable,
source, model outputs, predictions, checkpoints and source-closure proofs
were preserved; protected file hashes match before and after.

Archive: `artifacts/upstream-original1600-build-cache.tar.gz`, SHA256
`42b1a96aa4f98a14526db916370add5d4c73e70ebf8845e7fa0bc2e2db69ec60`.
Its members use repository-relative paths. They can be restored from the
repository root with `tar -xzf artifacts/upstream-original1600-build-cache.tar.gz`.
The plan retains exact original metadata, including nanosecond timestamps,
for restoration beyond the tar metadata if needed. Restoring cached files
does not restore their old inode identities; numerical proof hashes and the
pinned executable were not changed by this operation.

[Complete inventory](experiments/upstream-original1600-build-cache-archive/plan.json) ·
[Full archive payload verification before removal](experiments/upstream-original1600-build-cache-archive/archive-verification.json) ·
[Protected-file and recovered-space check](experiments/upstream-original1600-build-cache-archive/verification.json).

The admitted original1200 build cache was later preserved the same way:
1,269 regular single-link files,334,729,267 original bytes,116,470,812
archive bytes, recovering about210MiB net. The archive is
`artifacts/upstream-original1200-build-cache.tar.gz`, SHA256
`a0f9f9f9122e092ee7903f8e4ba3c5467b3fc2ca9de3b672ce3b9a1d813b76ef`.
Every member fully replayed before unlink; archive/evidence/parents were
fsynced. The admitted executable, build/admission receipts and frozen source
proofs match before and after. This permitted original1200 native admission
without discarding predictions, models, checkpoints or raw training data.
[Original1200 inventory](experiments/upstream-original1200-build-cache-archive/plan.json) ·
[Full archive replay](experiments/upstream-original1200-build-cache-archive/archive-verification.json) ·
[Protected-file check](experiments/upstream-original1200-build-cache-archive/verification.json).
