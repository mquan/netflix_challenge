#!/usr/bin/env python3
"""Separate complete five-user-fold solve/replay for both original KNN3 pairs."""
from pathlib import Path
import upstream_knn3pair2_screen as experiment
import verify_upstream_ready_public_schema as replay

if __name__=='__main__':
    replay.experiment=experiment;replay.main()
    path=experiment.REPORT/'ridge-verification.json';proof=experiment.u.read(path)
    proof.update(wrapper_sha256=experiment.u.sha(Path(__file__)),scope='Independent five-fold statistics, ridge solves and stored-vector replay for the fixed complete original KNN3 terminal addition. Known-ProbeX diagnostic only.')
    experiment.u.write(path,proof)
