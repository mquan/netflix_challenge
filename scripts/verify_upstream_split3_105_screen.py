#!/usr/bin/env python3
"""Independent five-user-fold solve/replay for the fixed own105 split3 addition."""
from pathlib import Path
import upstream_split3_105_screen as experiment
import verify_upstream_ready_public_schema as replay

if __name__=='__main__':
    replay.experiment=experiment;replay.main()
    path=experiment.REPORT/'ridge-verification.json';proof=experiment.u.read(path)
    proof.update(wrapper_sha256=experiment.u.sha(Path(__file__)),
        scope='Independently recomputed five-fold ridge statistics, solves and prediction replay for both fixed split3 attention columns. Known-ProbeX diagnostic only.')
    experiment.u.write(path,proof)
