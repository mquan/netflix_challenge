#!/usr/bin/env python3
"""Replay fixed all-nine and control arms with the unchanged independent solver."""
from pathlib import Path
import upstream_cf48_prefix_screen as experiment
import verify_upstream_ready_public_schema as replay


def main():
    # The solver reads experiment paths explicitly; no original module or data
    # file changes on disk. Its independent five-fold solve remains unchanged.
    replay.experiment=experiment
    replay.main()
    path=experiment.REPORT/'ridge-verification.json';proof=experiment.u.read(path)
    proof.update(wrapper_sha256=experiment.u.sha(Path(__file__)),full95_model_complete=False,
        scope='Independent control/all-nine known-ProbeX ridge replay. Our completed epoch25 snapshot pair is used; original95-epoch job is not claimed complete. No nonlinear or qualifying promotion.')
    experiment.u.write(path,proof)


if __name__=='__main__':main()
