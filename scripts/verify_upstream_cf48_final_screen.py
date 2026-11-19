#!/usr/bin/env python3
"""Independent fold-statistic/prediction replay of our final CF48 snapshot addition."""
from pathlib import Path
import upstream_cf48_final_screen as experiment
import verify_upstream_ready_public_schema as replay


if __name__=='__main__':
    replay.experiment=experiment;replay.main()
    path=experiment.REPORT/'ridge-verification.json';proof=experiment.u.read(path)
    proof.update(wrapper_sha256=experiment.u.sha(Path(__file__)),full95_model_complete=True,
        scope='Independent control/all-one known-ProbeX ridge replay using our completed original95-epoch CF48 terminal pair. No nonlinear or qualifying result.')
    experiment.u.write(path,proof)
