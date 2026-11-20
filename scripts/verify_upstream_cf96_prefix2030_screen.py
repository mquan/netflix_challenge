#!/usr/bin/env python3
"""Separate complete five-user-fold solve/replay for both original ownCF96 prefixes."""
from pathlib import Path
import upstream_cf96_prefix2030_screen as experiment
import verify_upstream_ready_public_schema as replay

if __name__=='__main__':
    replay.experiment=experiment;replay.main()
    path=experiment.REPORT/'ridge-verification.json';proof=experiment.u.read(path)
    proof.update(wrapper_sha256=experiment.u.sha(Path(__file__)),scope='Independent five-fold statistics, ridge solves and stored-vector replay for the fixed complete ownCF96 epoch20/30 prefix addition; full95/95 remains incomplete. Known-ProbeX diagnostic only.')
    experiment.u.write(path,proof)
