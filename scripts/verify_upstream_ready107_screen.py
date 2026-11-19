#!/usr/bin/env python3
"""Independent five-user-fold solve/replay of the fixed own107 ridge comparison."""
from pathlib import Path
import upstream_ready107_screen as experiment
import verify_upstream_ready_public_schema as replay

if __name__=='__main__':
    replay.experiment=experiment;replay.main()
    path=experiment.REPORT/'ridge-verification.json';proof=experiment.u.read(path)
    proof.update(wrapper_sha256=experiment.u.sha(Path(__file__)),
        scope='Independent five-fold statistics, ridge solves and stored-vector replay for the fixed all-two native addition. Known-ProbeX diagnostic only.')
    experiment.u.write(path,proof)
