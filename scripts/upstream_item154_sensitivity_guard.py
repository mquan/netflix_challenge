#!/usr/bin/env python3
"""Resource supervisor for the frozen item154 realization diagnostic."""
import datetime as dt
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import upstream_repro as u

REPORT=u.ROOT/'experiments/upstream-item154-sensitivity-guard'

def main():
    REPORT.mkdir(exist_ok=False)
    command=[sys.executable,str(u.ROOT/'scripts/upstream_item154_realization_sensitivity.py')]
    with (REPORT/'worker.log').open('x') as log:
        child=subprocess.Popen(['nice','-n','10',*command],cwd=u.ROOT,stdout=log,
            stderr=subprocess.STDOUT,start_new_session=True)
        high=0
        try:
            while child.poll() is None:
                output=subprocess.check_output(['ps','-axo','pid=,rss=,command='],text=True)
                total=0;own=0
                for line in output.splitlines():
                    parts=line.strip().split(None,2)
                    if len(parts)!=3:continue
                    pid,rss,cmd=parts
                    if int(pid)==child.pid:own=int(rss)*1024
                    if str(u.ROOT) in cmd or 'target/release/fit ' in cmd:
                        total+=int(rss)*1024
                free=shutil.disk_usage(u.ROOT).free
                high=high+1 if total>24*2**30 or own>7*2**30 else 0
                u.write(REPORT/'status.json',dict(state='running',pid=child.pid,
                    own_rss_bytes=own,research_rss_bytes=total,disk_free_bytes=free,
                    observed=dt.datetime.now(dt.timezone.utc).isoformat()))
                if high>=3 or free<16*2**30:
                    raise RuntimeError('Resource guard exceeded; diagnostic preserved')
                time.sleep(5)
        except BaseException:
            os.killpg(child.pid,signal.SIGTERM);child.wait();raise
        if child.returncode:raise RuntimeError('Diagnostic failed; original evidence preserved')
        u.write(REPORT/'completion.json',dict(state='complete',returncode=child.returncode))

if __name__=='__main__':main()
