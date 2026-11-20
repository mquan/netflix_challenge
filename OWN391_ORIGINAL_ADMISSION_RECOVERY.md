# Restore the original H128 admission rule

The initial exact-process recovery added a compressor-below4.5GiB admission
rule absent from the original H128 queue. Read-only cache eviction released
2.088GiB from33 immutable matrix inodes; global compressor subsequently fell
below the original5GiB cap but stayed above the extra4.5GiB threshold.
Idle compressed anonymous pages need not decompress merely because cache
pages were released. No artificial memory pressure or model-state access
is used to change that metric.

This separate revision restores the ORIGINAL compressor-below5GiB admission
rule, with three stable observations,16GiB reservation credited for already
allocated paused RAM, and the original nonblocking GPU lane/recheck. Every
continuous guard remains unchanged: own16GiB, research24GiB physical/RSS,
compressor5GiB, swap growth128MiB from the original last preguard sample,
and disk18.75GiB. The narrow compressor margin may retrigger the guard;
no claim of safe continuation is made before actual admission.

Only the former waiting supervisor is retired through its signal handler,
after verifying it has never continued the worker and owns no learning
child. Original process71312 remains stopped and is adopted with full
PID/start/command/PGID identity. All former source/plans/failure receipts,
numerical recipes, tensors, optimizer and RNG are preserved. CF35985 stays
stopped. No fitting worker is restarted. Full original saved-model/scaler
replay and separate canonical verification remain mandatory before adding
completion evidence for existing downstream queues.

Evidence: `experiments/upstream-own391-neural-original-admission/`.
