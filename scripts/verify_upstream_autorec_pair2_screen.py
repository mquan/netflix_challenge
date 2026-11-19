#!/usr/bin/env python3
"""Independent complete five-fold statistics and solves for raw snapshots5/10."""
from pathlib import Path
import upstream_autorec_pair2_screen as experiment
import verify_upstream_ready_public_schema as replay

if __name__=='__main__':
    replay.experiment=experiment;replay.main()
    path=experiment.REPORT/'ridge-verification.json';p=experiment.u.read(path)
    p.update(wrapper_sha256=experiment.u.sha(Path(__file__)),scope='Independent five-fold statistics, solves and every stored prediction replay for fixed own raw snapshot5/10 additions. No whole50-epoch completion, nonlinear or qualifying score claimed.')
    experiment.u.write(path,p)
