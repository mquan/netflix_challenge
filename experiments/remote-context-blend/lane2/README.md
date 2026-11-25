# Lane 2: TX300 on a second split (known phase), 2026-11-25

Declared in the plan amendment "Lane 2". Probex known rows (1,128,164), 5 original user folds,
global ridge λ=1000 fitted on outer-training rows only. No qualifying labels read.
Bank v2 already contains A (`s3-tx300r`, split-3, seed 42), so (ii) uses bank v2 without A
and (iii) uses the full bank v2.

Arm C (split-3s, seed 20261124) finished 2026-11-25T03:46:57Z, exit 0, peak footprint 52.5 GB.
Arm B (split-3, seed 20261124) finished its known phase at epoch 24 (probe_3 0.87832) and is now in fulltrain→qual.

| | A | B | C |
|---|---|---|---|
| raw probex RMSE | 0.877325 | 0.877316 | 0.877273 |
| 1-column ridge, pooled | 0.877197 | 0.877169 | 0.877155 |

| comparison | Δ pooled | folds better |
|---|---|---|
| (i) C vs B, standalone | −0.000014 | 3/5 |
| (ii) bank−A + C vs bank−A + B | −0.000019 | 5/5 |
| (iii) bank + C vs bank + B | −0.000019 | 5/5 |
| bank + B vs bank (0.863155) | −0.000024 | 5/5 |
| bank + C vs bank | −0.000043 | 5/5 |
| bank + B + C vs bank + B | −0.000020 | 5/5 |

Correlations on probex: A–B 0.99559, A–C 0.99142, B–C 0.99504. The second split decorrelates
C from A more than a seed change does, and it adds about 1.8× B's gain on the bank. The absolute
gain is small (−0.000043). It is consistent on every fold but far under the 0.001 bar that the
earlier rules used.

Frozen known-side package: release asset `rcb-lane2-tx300-known.tar.gz` (830,013,342 B, sha256
`037c9642046aad869c8f8b297417fee7bb7eeac1e8860167f734b8ea2b0c98c3`). It holds C's probe_3s and train_3s vectors, 24 epoch
snapshots, log, start/end receipts and row map, plus B's probe_3 and train_3 vectors. Members:
`known-package-members.sha256`. B's qual vector (= C's qual by declaration) follows when B's
second phase ends. Summary: `summary.json`, script `scripts/rcb_lane2_compare.py`.

## Arm B final (qual phase), 2026-11-25
B exited 0 at 09:07:21Z; peak footprint 55.3 GB. Qual vector `s3-tx300r.qual.npy` sha256
`169d16948ca7cb5f275f10ca4204df64b87731130a0464ad362d3ae8a9189747` (2,817,131 float32, finite, raw range
−2.96..8.33 before clipping); fulltrain `13803aa7…8039`. Correlation with A's qual vector 0.99594, matching the
known-side A–B 0.99559. By declaration this is also C's qual vector. Release asset `rcb-lane2-tx300-B-final.tar.gz`
(724,189,228 B, sha256 `3dc518a762cbd4db3297d03f32493f541332c95ef695f596e8772949e17e2f08`): qual and fulltrain
vectors, 24 qual/fulltrain epoch snapshots, log, receipts. Members: `B-final-package-members.sha256`.
