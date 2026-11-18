#!/usr/bin/env python3
"""Run the frozen 85-column production recipe using the disclosed phase bank."""
import argparse
from pathlib import Path
import threading
import time
import upstream_production as core
import upstream_phase_bank as bank

u=core.u
CORE_FILE=Path(core.__file__)
CORE_PATHS=list(core.code_hashes())
RUN=u.ROOT/'artifacts/upstream-phase-production'
REPORT=u.ROOT/'experiments/upstream-phase-production'
OPERATIONAL=u.ROOT/'experiments/upstream-production/status.json'


def code_hashes():
    paths=[u.ROOT/name for name in CORE_PATHS]+[Path(__file__),Path(bank.__file__),Path(bank.phase.__file__)]
    return {str(p.relative_to(u.ROOT)):u.sha(p) for p in paths}


def configure():
    bank.configure()
    core.RUN,core.REPORT,core.bank,core.code_hashes=RUN,REPORT,bank,code_hashes
    # The unchanged core launches child workers through its own __file__ global.
    # Point those children to this adapter, which repeats the same bindings.
    core.__file__=str(Path(__file__).resolve())


def relay(stop):
    while not stop.wait(5):
        path=REPORT/'status.json'
        if path.exists():
            state=u.read(path);state['scheduler']='independent-phase-production'
            u.write(OPERATIONAL,state)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--family',choices=['tree','nn']);parser.add_argument('--replay',action='store_true');args=parser.parse_args()
    configure()
    if args.family:core.worker(args.family,args.replay)
    else:
        stopping=threading.Event();thread=threading.Thread(target=relay,args=(stopping,),daemon=True);thread.start()
        try:core.run()
        finally:
            stopping.set();thread.join()
            if (REPORT/'status.json').exists():
                state=u.read(REPORT/'status.json');state['scheduler']='independent-phase-production';u.write(OPERATIONAL,state)
