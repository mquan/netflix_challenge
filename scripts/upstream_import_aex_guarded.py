#!/usr/bin/env python3
"""Route completed AEX reuse to the primary scheduler with AEX delegation."""
from pathlib import Path
import upstream_import_aex as inherited
import upstream_primary_handoff_aex as handoff

u=inherited.u
REPORT=u.ROOT/'experiments/upstream-import-aex-guarded'
original_configure=inherited.configure


def configure():
    original_configure()
    inherited.transfer.PRIMARY_FRAGMENT='scripts/upstream_primary_guarded_aex.py'


def run():
    REPORT.mkdir(parents=True,exist_ok=True)
    files=[Path(__file__),Path(inherited.__file__),Path(handoff.__file__),
        Path(handoff.successor.__file__),Path(inherited.transfer.__file__)]
    policy=dict(code_sha256={str(p.relative_to(u.ROOT)):u.sha(p) for p in files},
        predecessor_handoff_report=str(handoff.REPORT),
        scheduler_command_fragment='scripts/upstream_primary_guarded_aex.py',
        qualifying_labels_read=False)
    path=REPORT/'routing-plan.json'
    if path.exists() and u.read(path)!=policy:raise ValueError('AEX reuse routing changed')
    u.write(path,policy)
    inherited.REPORT=REPORT;inherited.handoff.REPORT=handoff.REPORT;inherited.configure=configure
    inherited.run()


if __name__=='__main__':
    try:run()
    except BaseException as error:
        if not isinstance(error,BlockingIOError):u.write(REPORT/'queue-status.json',dict(state='incomplete_preserved',error=repr(error)))
        raise
