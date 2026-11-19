#!/usr/bin/env python3
"""Use the tested idle handoff with the additional AEX128 delegation."""
from pathlib import Path
import signal
import upstream_primary_handoff as inherited
import upstream_primary_guarded_aex as successor

u=successor.u
REPORT=successor.REPORT


def prepare():
    REPORT.mkdir(parents=True,exist_ok=True)
    files=[Path(__file__),Path(inherited.__file__),Path(successor.__file__),
        Path(successor.original.__file__),successor.original.REPORT/'plan.json',
        Path(inherited.transfer.__file__),Path(successor.core.__file__)]
    result=dict(code_sha256={str(p.relative_to(u.ROOT)):u.sha(p) for p in files},
        original_launch=u.read(inherited.PRIMARY/'launch.json'),source_commit=u.PIN,
        delegated_source_jobs=sorted(successor.DELEGATED),qualifying_labels_read=False,
        policy='Same tested idle-only scheduler transition. Hold all parallel heap recipes, existing large serial models, CF48/96/96r and already-running independent AEX128; complete imported native receipts are checked before holds.')
    path=REPORT/'plan.json'
    if path.exists() and u.read(path)!=result:raise ValueError('Existing AEX-aware handoff plan changed')
    u.write(path,result);return result


def run():
    def interrupted(signum,frame):raise KeyboardInterrupt('Signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted)
    inherited.REPORT=REPORT;inherited.successor=successor;inherited.prepare=prepare
    inherited.run()


if __name__=='__main__':
    try:run()
    except BaseException as error:
        if not isinstance(error,BlockingIOError):u.write(REPORT/'status.json',dict(state='incomplete_preserved',error=repr(error)))
        raise
