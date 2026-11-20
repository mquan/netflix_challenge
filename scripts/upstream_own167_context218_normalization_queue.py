#!/usr/bin/env python3
"""Guard complete normalization parity using compressed-inclusive footprints."""
import datetime as dt
import fcntl
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import upstream_repro as u
import upstream_cf96_memory_resume as memory

REPORT=u.ROOT/'experiments/upstream-own167-context218-normalization-queue'
WORKER=u.ROOT/'scripts/upstream_own167_context218_normalization.py'

def footprints(extra=None):
    result={}
    for line in subprocess.check_output(['ps','-axo','pid=,rss=,command='],text=True).splitlines():
        parts=line.strip().split(None,2)
        if len(parts)!=3:continue
        pid,rss,cmd=parts;pid=int(pid)
        native='/target/release/' in cmd.split()[0]
        if not (native or pid==extra or (str(u.ROOT) in cmd and int(rss)>64*1024)):continue
        summary=subprocess.run(['vmmap','-summary',str(pid)],text=True,capture_output=True)
        match=re.search(r'^Physical footprint:\s+([\d.]+)([KMG])',summary.stdout,re.M)
        if not match:
            if subprocess.run(['ps','-p',str(pid),'-o','pid='],text=True,capture_output=True).stdout.strip():raise RuntimeError('Cannot account for live process footprint')
            continue
        result[str(pid)]=dict(bytes=float(match.group(1))*{'K':2**10,'M':2**20,'G':2**30}[match.group(2)],rss_bytes=int(rss)*1024,command=cmd)
    return result

def main():
    REPORT.mkdir(exist_ok=False)
    frozen={WORKER:u.sha(WORKER),Path(__file__):u.sha(Path(__file__))}
    def check_frozen():
        for p,digest in frozen.items():
            if u.sha(p)!=digest:raise ValueError('Frozen normalization implementation changed')
    u.write(REPORT/'plan.json',dict(worker_sha256=frozen[WORKER],supervisor_sha256=frozen[Path(__file__)],
        rule='Unchanged actual522-column/full-eligible parity. Reserve8GiB measured temporary footprint, require24GiB total compressed-inclusive research footprint,14GiB own RSS and footprint,24GiB RSS,16GiB disk floor. No data/model reduction.'))
    def interrupted(signum,frame):raise KeyboardInterrupt('Normalization supervisor signal')
    signal.signal(signal.SIGTERM,interrupted)
    lock=(u.ROOT/'experiments/upstream-source-extension/run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    previous=memory.memory();good=0
    while good<2:
        check_frozen()
        workers=footprints();current=memory.memory();total=sum(v['bytes'] for v in workers.values())
        good=good+1 if total+8*2**30<24*2**30 and current['swap_used_bytes']-previous['swap_used_bytes']<64*2**20 else 0
        u.write(REPORT/'status.json',dict(state='waiting_for_exact_complete_parity_headroom',workers=workers,memory=current,good_samples=good))
        previous=current
        if good<2:time.sleep(10)
    baseline=current
    check_frozen()
    with (REPORT/'worker.log').open('x') as log:
        child=subprocess.Popen([sys.executable,str(WORKER)],cwd=u.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        high=0
        try:
            while child.poll() is None:
                check_frozen()
                workers=footprints(child.pid);current=memory.memory();own=workers.get(str(child.pid),dict(bytes=0,rss_bytes=0))
                total=sum(v['bytes'] for v in workers.values());rss=sum(v['rss_bytes'] for v in workers.values())
                high=high+1 if total>24*2**30 or rss>24*2**30 or own['bytes']>14*2**30 or own['rss_bytes']>14*2**30 or current['swap_used_bytes']-baseline['swap_used_bytes']>128*2**20 else 0
                u.write(REPORT/'status.json',dict(state='actual_complete_normalization_parity',pid=child.pid,workers=workers,memory=current,observed=dt.datetime.now(dt.timezone.utc).isoformat()))
                if high>=3 or shutil.disk_usage(u.ROOT).free<16*2**30:raise RuntimeError('Complete parity resource guard; evidence preserved')
                time.sleep(5)
        except BaseException:
            if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
            child.wait();raise
        if child.returncode:raise RuntimeError('Complete actual parity failed; evidence preserved')
    check_frozen()
    u.write(REPORT/'completion.json',dict(state='complete_guarded_actual522_input_normalization_parity',worker_sha256=u.sha(WORKER)))

if __name__=='__main__':main()
