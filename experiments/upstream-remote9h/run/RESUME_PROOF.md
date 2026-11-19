# CFNADE48 resume proof — 2026-11-19T06:06:54Z

- Actual start (UTC): 2026-11-19T06:03:28Z; training deadline 2026-11-19T14:33:28Z; return by 15:03:28Z
- Package: release remote9h-cfnade48-20261119 / cfnade48-resume.tar.gz,
  sha256 0399ae6ed06f2529e302f6f090bed8125e91cf4e51e4f445fe8390ccd1be5c63 (matches tracked metadata; archive kept chmod 444)
- verify-bundle: PASS (all package members, 36 data arrays, zero qual targets, runtime config, native 160/48 parity: 8 cases, init + 7 epochs)
- source_id e14530b5ade496efb3980a64f98442d35c9a606960add79149ce67b057677d94
- data_id   55ce479fc16447a826d013288b2ccb3e1eec39a07d7783f3d860a3f6e33e1fe0
- config: CfNadeConfig n_hidden 160, rank 48, n_epochs 95, seed 42 (unchanged; `cfnade48-stage --describe`)
- Binaries (copied, ran as-is, no rebuild):
  - bin/cfnade48-stage  0a239de36f5ce926ae6463788f6c88c651be65b903c6e4a90ed0639ebfbf68b2
  - bin/cfnade48-parity 4d3becd04a39a467a822449c5030add89853dcb77437b8e9c261acd9a6a0c2db
- Commands: `experiments/upstream-remote9h/run/launch-cf48.sh x|qual` (caffeinate -i → remote12h_supervisor.py
  --name remote9h-cf48-{x,qual} --deadline 2026-11-19T14:33:28Z --cwd $P/native-source -- $P/bin/cfnade48-stage {x,qual});
  RAYON_NUM_THREADS=2, OMP/OPENBLAS/MKL/VECLIB=1, CF_EXIT_AFTER_CHECKPOINT/CF_PROOF_* unset; tmux session remote9h-cf48
- native-source/data → symlink to ~/workspace/netflix_upstream_repro/artifacts/upstream-repro/fresh/data (not copied)

## Restore
| phase | start epoch | log | regenerated prediction sha256 | manifest expected | match |
|---|---|---|---|---|---|
| trainx→probex | 31 | `Exact CFNADE checkpoint restored: trainx epoch 31` | e8fe6d09f9d04836e49a153c03b168c4556713c90614ca859824371ee00e5ab1 | e8fe6d09…5ab1 | exact |
| fulltrain→qual | 30 | `Exact CFNADE checkpoint restored: fulltrain epoch 30` | 92a8192d52b163445c65390ef72f16d3a4b8cfcdb73dc5eab39d46ffea1205b9 | 92a8192d…05b9 | exact |

Copied latest-epoch predictions were moved first to `$P/baseline/*.baseline.npy` (hashes equal to the manifest's).

**Deviation, disclosed:** the package shipped without `checkpoints/{trainx,fulltrain}/owner.json`, and the native code
refuses unowned directories that have no owner file (`refuse unowned checkpoint contents`). I wrote each owner.json
verbatim from that epoch's `state.json["identity"]`. The binary then checked it against the identity it computes
from env/data/config (strict equality) and against state.json, so the check could not have been forced to pass.
I changed no checkpoint arrays, state or pointers.

## Live processes at proof time
```
95402 95402 95394 3949920       01:04 /Users/quan/workspace/netflix_remote9h/artifacts/upstream-remote9h/cfnade48-package/bin/cfnade48-stage qual
95404 95404 95390 3851024       01:04 /Users/quan/workspace/netflix_remote9h/artifacts/upstream-remote9h/cfnade48-package/bin/cfnade48-stage x
95390 95388 95388  17712       01:04 /opt/homebrew/Cellar/python@3.12/3.12.9/Frameworks/Python.framework/Versions/3.12/Resources/Python.app/Contents/MacOS/Python /Users/quan/workspace/netflix_remote9h
95397 95388 95390   1648       01:04 caffeinate -i /Users/quan/workspace/netflix_upstream_repro/.venv/bin/python /Users/quan/workspace/netflix_remote9h/scripts/remote12h_supervisor.py --name remote9h-
95394 95392 95392  17776       01:04 /opt/homebrew/Cellar/python@3.12/3.12.9/Frameworks/Python.framework/Versions/3.12/Resources/Python.app/Contents/MacOS/Python /Users/quan/workspace/netflix_remote9h
95396 95392 95394   1632       01:04 caffeinate -i /Users/quan/workspace/netflix_upstream_repro/.venv/bin/python /Users/quan/workspace/netflix_remote9h/scripts/remote12h_supervisor.py --name remote9h-
```
Coordinator jobs: none on this machine; none touched.
