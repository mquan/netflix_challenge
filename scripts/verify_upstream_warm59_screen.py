#!/usr/bin/env python3
"""Independent five-fold statistics, solves and replay for the fixed warm59 batch."""
from pathlib import Path
import upstream_warm59_screen as experiment
import verify_upstream_ready_public_schema as replay

if __name__=='__main__':
    replay.experiment=experiment;replay.main()
    path=experiment.REPORT/'ridge-verification.json';proof=experiment.u.read(path)
    proof.update(wrapper_sha256=experiment.u.sha(Path(__file__)),
        scope='Independent five-fold statistics, ridge solves and full stored-vector replay for all59 completed captured snapshot pairs. No full95 CF96 completion, nonlinear or qualifying score claimed.')
    experiment.u.write(path,proof)
