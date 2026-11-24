# Lane 1 — CF-NADE residual neighbourhoods

Base: own CFNADE48. Training-row predictions were exported from its epoch-95 checkpoint, and the probex and qual replays
are byte-exact (plan amendment). Chains run in `artifacts/rcb-r1` (`lane1.sh`, tmux `cf-lanes:lane1`).

## Missing inputs for the preferred rank-96 base (own CF96)

None of the own CF96 vectors are on this machine. Only the upstream reference `cfnade-96*` vectors are here, and they
are excluded. To run the same four chains on CF96, upload to `remote-context-blend-20261124`:

| file (into `preds_new/`) | rows | phase | role |
|---|---|---|---|
| `cfnade-96.trainx.npy` | 99,352,343 | known (trainx→probex) | training-row predictions |
| `cfnade-96.probex.npy` | 1,128,164 | known | query predictions |
| `cfnade-96.fulltrain.npy` | 100,480,507 | qualifying (fulltrain→qual) | training-row predictions |
| `cfnade-96.qual.npy` | 2,817,131 | qualifying | query predictions |

All four must be float32 in canonical native row order, from the same terminal model (state the SWA terminal or
epoch), with sha256 for each file. Once they arrive, `lane1.sh` reruns with `cfnade-96` in place of `cfnade-48`.
