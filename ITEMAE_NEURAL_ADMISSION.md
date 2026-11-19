# Keep the full-size neural test within its measured memory budget

The narrow raw MPS model can overlap the CPU tree, but the full-size virtual
neural study has a substantially larger peak. Its original admission helper
counts native jobs without the concurrent CPU tree, so it could admit a neural
fit that pushes combined RSS above the existing24GiB guard and interrupts both
long-running fits. Earlier full-size neural workers reached about14GiB RSS.

Before pair10 arrived or any neural fitting/preparation began, admission
controller68129 verified pending coordinator63754 had no child and was still
waiting at its declared5/10 boundary. It holds only that idle coordinator while
the CPU tree fit, saved-model replay and supervision remain live. Raw GPU
training, original attention64 and both CFNADE96 phases continue unchanged.
No fitting worker was stopped, no recipe changed and no budget reduced.

After all actual CPU tree workers/replay/supervision finish, pair10 must be
ready and two stable readings must show other jobs below8GiB RSS and compressor
below5GiB. This reserves16GiB for the full neural peak under the original24GiB
combined limit. The same original coordinator is then released; its original
GPU epoch lock, source checks, capacities, data, folds, seeds, stopping and
actual saved-model replays still apply. Raw epochs and neural fitting remain
serialized on that GPU lock. This is extra admission, not a new model recipe.

The boundary fixture rejects a live CPU tree even at low current RSS, excessive
other-job RSS, excessive compressor use and a missing pair. It admits a ready
pair after the tree and safe peak headroom. The live status confirms actual
CPU worker60801/supervisor62158 and the intentionally held pending coordinator.

Evidence: experiments/upstream-itemae-neural-admission/plan.json and
experiments/upstream-itemae-neural-admission-fixture/verification.json.
Implementation: scripts/upstream_itemae_neural_admission.py.
No new ensemble, qualifying or official Test result is claimed.

At23:47 UTC, both actual epoch5/10 pairs are ready and independently verified.
The same pending test remains held for the live CPU tree and its required
peak reserve. Raw training continues at11; no checkpoint or capacity changes.
