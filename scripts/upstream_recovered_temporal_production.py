#!/usr/bin/env python3
"""Run the first fixed85 production fit from our recovered complete temporal pairs."""
import argparse
from pathlib import Path
import threading
import upstream_production as core
import upstream_recovered_temporal_bank as bank

u=core.u
CORE_PATHS=list(core.code_hashes())
RUN=u.ROOT/'artifacts/upstream-phase-production'
REPORT=u.ROOT/'experiments/upstream-phase-production'
OPERATIONAL=u.ROOT/'experiments/upstream-production/status.json'
ORIGINAL_WRITE=u.write


def code_hashes():
    paths=[u.ROOT/name for name in CORE_PATHS]+[Path(__file__)]
    return {**bank.code_hashes(),**{str(p.relative_to(u.ROOT)):u.sha(p) for p in paths}}


def publish(path,value):
    # The unchanged numerical runner historically described its mapped input.
    # Correct only this adapter's lineage text BEFORE either report is frozen;
    # all fitting, replay, prediction hashing and audit operations stay original.
    if Path(path) in [REPORT/'plan.json',REPORT/'report.json']:
        value=dict(value,temporal_provenance=bank.PROVENANCE,
            paired_bank_manifest_sha256=u.sha(bank.REPORT/'manifest.json'))
        if Path(path)==REPORT/'report.json':
            value['scope']='Fresh own-trained paired85 bank and full-known-ProbeX-fitted weights. '+bank.PROVENANCE+' Frozen before the recovered qualifying audit.'
    return ORIGINAL_WRITE(path,value)


def configure():
    bank.configure()
    core.RUN,core.REPORT,core.bank,core.code_hashes=RUN,REPORT,bank,code_hashes
    core.__file__=str(Path(__file__).resolve())
    u.write=publish


def relay(stopping):
    while not stopping.wait(5):
        path=REPORT/'status.json'
        if path.exists():
            state=u.read(path);state['scheduler']='recovered-own-temporal-production';u.write(OPERATIONAL,state)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--family',choices=['tree','nn']);parser.add_argument('--replay',action='store_true');args=parser.parse_args()
    configure()
    if args.family:core.worker(args.family,args.replay)
    else:
        stopping=threading.Event();thread=threading.Thread(target=relay,args=(stopping,),daemon=True);thread.start()
        try:core.run()
        finally:stopping.set();thread.join()
