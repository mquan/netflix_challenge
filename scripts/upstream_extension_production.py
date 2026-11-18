#!/usr/bin/env python3
"""Fit/replay/freeze the fixed 91-column extension with the unchanged core."""
import argparse
from pathlib import Path
import threading
import upstream_production as core
import upstream_extension_bank as bank
import upstream_source_extension as development

u=core.u
CORE_PATHS=list(core.code_hashes())
RUN=u.ROOT/'artifacts/upstream-extension-production'
REPORT=u.ROOT/'experiments/upstream-extension-production'
OPERATIONAL=u.ROOT/'experiments/upstream-production/status.json'


def code_hashes():
    paths=[u.ROOT/name for name in CORE_PATHS]+[Path(__file__),Path(bank.__file__),Path(development.__file__),
        Path(bank.parent.__file__),Path(bank.parent.phase.__file__),Path(__file__).with_name('upstream_extension_run.py')]
    return {str(p.relative_to(u.ROOT)):u.sha(p) for p in paths}


def configure():
    core.RUN,core.REPORT,core.bank,core.voting,core.code_hashes=RUN,REPORT,bank,development,code_hashes
    core.__file__=str(Path(__file__).resolve())


def relay(stop):
    while not stop.wait(5):
        path=REPORT/'status.json'
        if path.exists():
            state=u.read(path);state['scheduler']='fresh-source-extension-production';u.write(OPERATIONAL,state)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--family',choices=['tree','nn']);parser.add_argument('--replay',action='store_true');args=parser.parse_args();configure()
    if args.family:core.worker(args.family,args.replay)
    else:
        stopping=threading.Event();thread=threading.Thread(target=relay,args=(stopping,),daemon=True);thread.start()
        try:core.run()
        finally:
            stopping.set();thread.join()
            if (REPORT/'status.json').exists():
                state=u.read(REPORT/'status.json');state['scheduler']='fresh-source-extension-production';u.write(OPERATIONAL,state)
