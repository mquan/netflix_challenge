#!/usr/bin/env python3
"""Independent five-user-fold solve and prediction replay for own CF96 epoch10."""
from pathlib import Path
import upstream_cf96_prefix10_screen as experiment
import verify_upstream_ready_public_schema as replay

if __name__=='__main__':
    replay.experiment=experiment;replay.main()
    path=experiment.REPORT/'ridge-verification.json';proof=experiment.u.read(path)
    proof.update(wrapper_sha256=experiment.u.sha(Path(__file__)),full95_model_complete=False,
        scope='Independent five-fold statistics, ridge solves and vector replay for one fixed source-published CF96 epoch10 pair. No95-epoch completion or qualifying score claimed.')
    experiment.u.write(path,proof)
