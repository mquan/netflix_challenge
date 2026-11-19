#!/usr/bin/env python3
"""Bind the current development gate while preserving frozen production code."""
import argparse
from pathlib import Path
import threading
import upstream_extension_production as inherited

core=inherited.core
original_hashes=inherited.code_hashes


def code_hashes():
    paths=[Path(__file__),Path(__file__).with_name('upstream_extension_run_current.py')]
    return {**original_hashes(),**{str(p.relative_to(inherited.u.ROOT)):inherited.u.sha(p) for p in paths}}


def configure():
    inherited.configure();core.code_hashes=code_hashes;core.__file__=str(Path(__file__).resolve())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--family',choices=['tree','nn']);parser.add_argument('--replay',action='store_true')
    args=parser.parse_args();configure()
    if args.family:core.worker(args.family,args.replay)
    else:
        stopping=threading.Event();thread=threading.Thread(target=inherited.relay,args=(stopping,),daemon=True);thread.start()
        try:core.run()
        finally:
            stopping.set();thread.join()
            if (inherited.REPORT/'status.json').exists():
                state=inherited.u.read(inherited.REPORT/'status.json');state['scheduler']='fresh-source-extension-production';inherited.u.write(inherited.OPERATIONAL,state)
