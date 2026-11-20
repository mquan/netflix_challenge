#!/usr/bin/env python3
"""Guard complete original MF60 production alongside existing native jobs."""
import datetime as dt
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import upstream_repro as u
import upstream_research_footprints as resources
import upstream_cf96_memory_resume as memory

REPORT=u.ROOT/'experiments/upstream-split3-mf60-original-queue'
def stamp():return dt.datetime.now(dt.timezone.utc).isoformat()

def main():
    REPORT.mkdir(exist_ok=False);child=None
    files=[Path(__file__),u.ROOT/'scripts/upstream_split3_mf60_original.py',u.ROOT/'scripts/verify_upstream_split3_mf60_original.py',Path(resources.__file__)]
    frozen={str(p.relative_to(u.ROOT)):u.sha(p) for p in files}
    def check():
        for key,h in frozen.items():
            if u.sha(u.ROOT/key)!=h:raise ValueError('Original MF60 execution changed')
    u.write(REPORT/'plan.json',dict(created=stamp(),source_sha256=frozen,
        recipe='Original full MF60/20epochs per phase,train3/probe3 thenfulltrain/qual,all original outputs including full training predictions. Separate namespace,8GiB admission reserve,continuous total24GiB/native14GiB RSS+physical footprint guards,128MiB swap growth and16GiB disk floor. No automatic restart or qualifying access.'))
    def interrupted(signum,frame):raise KeyboardInterrupt('Original MF60 supervisor signal')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        # Admission also occurs in the original worker. Reserving here prevents
        # source preparation from racing a native allocation at the same moment.
        previous=memory.memory();good=0
        while good<2:
            check();workers=resources.footprints();current=memory.memory()
            admit=(sum(v['bytes'] for v in workers.values())+8*2**30<24*2**30 and current['compressor_bytes']<5*2**30 and
                current['swap_used_bytes']-previous['swap_used_bytes']<32*2**20 and shutil.disk_usage(u.ROOT).free>20*2**30)
            good=good+1 if admit else 0;previous=current
            u.write(REPORT/'status.json',dict(state='waiting_for_full_original_MF60_headroom',workers=workers,memory=current,good_samples=good,observed=stamp()))
            if good<2:time.sleep(30)
        baseline=current
        for phase,script in [('full-original-producer','upstream_split3_mf60_original.py'),('independent-native-verification','verify_upstream_split3_mf60_original.py')]:
            check();high=0
            with (REPORT/(phase+'.log')).open('x') as log:
                child=subprocess.Popen([sys.executable,str(u.ROOT/'scripts'/script)],cwd=u.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                u.write(REPORT/'launch.json',dict(phase=phase,pid=child.pid,observed=stamp()))
                while child.poll() is None:
                    check();workers=resources.footprints(child.pid);current=memory.memory()
                    # The native child is a descendant of the private worker's
                    # process group; no unrelated existing job is ever stopped.
                    owned=subprocess.run(['pgrep','-P',str(child.pid)],capture_output=True,text=True).stdout.split()
                    own=[workers[pid] for pid in [str(child.pid),*owned] if pid in workers]
                    high=high+1 if (sum(v['bytes'] for v in workers.values())>24*2**30 or sum(v['rss_bytes'] for v in workers.values())>24*2**30 or
                        sum(v['bytes'] for v in own)>14*2**30 or sum(v['rss_bytes'] for v in own)>14*2**30 or
                        current['swap_used_bytes']-baseline['swap_used_bytes']>128*2**20) else 0
                    u.write(REPORT/'status.json',dict(state=phase,pid=child.pid,workers=workers,memory=current,consecutive_high_samples=high,observed=stamp()))
                    if high>=3 or shutil.disk_usage(u.ROOT).free<16*2**30:raise RuntimeError('Original native resource guard exceeded; partial source evidence retained')
                    time.sleep(5)
                if child.returncode:raise RuntimeError('Original MF60 phase failed; actual evidence retained: '+phase)
                child=None
        proof=u.ROOT/'experiments/upstream-split3-mf60-original/s3-mf60-verification.json';p=u.read(proof)
        if p['state']!='complete_original_two_phase_split3_model':raise ValueError('Complete actual original source proof required')
        u.write(REPORT/'completion.json',dict(state='complete_original_full_MF60_and_independent_native_verification',observed=stamp(),verification_sha256=u.sha(proof),qualifying_labels_read=False))
    except BaseException as error:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        u.write(REPORT/'failure.json',dict(state='incomplete_preserved',error=repr(error),observed=stamp()));raise

if __name__=='__main__':main()
