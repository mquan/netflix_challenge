# Exact cache storage sharing

Six inactive duplicate cache pairs now use APFS copy-on-write storage. Every
logical path remains present with the original complete bytes, permissions,
owner, group, and modification time. The pairs retain separate inodes and
independent future writes; these are not hard links.

The duplicated logical payload totals **10,405,832,500 bytes (10.41 GB /
9.69 GiB)**. Measured free space rose by **10,404,577,280 bytes**, from
19,288,215,552 to29,692,792,832 bytes, approximately **18.0 to27.7 GiB**.
Ongoing native writes can affect the observed free-space delta.

| Inactive cache copies | Duplicate bytes shared |
|---|---:|
| Phase-bank and recovered-temporal-bank qualifying design | 3,628,464,856 |
| Their qualifying features | 2,490,343,932 |
| Their known-query design | 1,453,075,360 |
| Remote-batch and audit-format design | 1,435,024,736 |
| Phase-bank and recovered-temporal-bank known features | 997,297,104 |
| Remote-batch and audit-format features | 401,626,512 |

Each pair passed complete SHA256 equality and a full byte comparison before
replacement. The operation checked unchanged file identities, zero open
handles, one link per file, no flags, no extended attributes or ACL, and a
compatible filesystem and metadata. A temporary clone was fully rehashed and
checked before atomic replacement; source and destination were rehashed again
afterward, and both actual NumPy mmap paths were opened. A prior disposable
same-filesystem fixture proved that writing through a clone changes only the
clone, with separate source and destination inodes.

Datasets, trained models, predictions, checkpoints, and active files were
preserved. No raw ratings or numerical research payload was published. The
extra headroom clears the next native producer's20 GiB admission threshold;
its other scheduling and memory requirements remain in force.

Evidence: [execution](experiments/upstream-cow-cache-storage/verification.json),
[separate complete-byte verification](experiments/upstream-cow-cache-storage/independent-verification.json),
[declared operations](experiments/upstream-cow-cache-storage/plan.json),
[full candidate hashes](experiments/upstream-old-cache-storage/full-hash-candidates.json),
[write-isolation fixture](experiments/upstream-storage-compression-fixture/clone-verification.json).
