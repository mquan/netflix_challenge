#!/usr/bin/env python3
"""Serial original-budget matched study and independent replay with live guards."""
import datetime as dt
import fcntl
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
import upstream_cf96_primary_lane as owners

REPORT=u.ROOT/'experiments/upstream-split3-matched-resume-queue'
def stamp():return dt.datetime.now(dt.timezone.utc).isoformat()

def main():
    REPORT.mkdir(exist_ok=False)
    scripts=[Path(__file__),u.ROOT/'scripts/upstream_split3_matched_data.py',u.ROOT/'scripts/upstream_split3_matched_development_inputs.py',u.ROOT/'scripts/upstream_split3_matched_resume.py',u.ROOT/'scripts/verify_upstream_split3_matched_resume.py',Path(resources.__file__)]
    frozen={str(p.relative_to(u.ROOT)):u.sha(p) for p in scripts}
    def check():
        for key,h in frozen.items():
            if u.sha(u.ROOT/key)!=h:raise ValueError('Frozen complete matched execution changed')
    u.write(REPORT/'plan.json',dict(created=stamp(),source_sha256=frozen,rule='Seven original folds preserved; three missing expanded folds only after independently complete originalTX22/22. Same complete numerical matched study; exclusive GPU lock;11GiB admission reserve,total24GiB/worker14GiB compressed-inclusive and RSS guards,swap growth128MiB,disk16GiB floor. No automatic restarts or reduced feature/row/epoch/seed budgets.'))
    child=None
    def interrupted(signum,frame):raise KeyboardInterrupt('Matched supervisor signal')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        lock=(u.ROOT/'experiments/upstream-source-extension/run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
        previous=memory.memory();good=0
        while good<2:
            check();workers=resources.footprints();current=memory.memory()
            finished=u.ROOT/'experiments/upstream-tx100-serial-memory-recovery/completion.json'
            ready=False
            if finished.exists():
                done=u.read(finished);native=u.read(finished.parent/'plan.json')['original_processes']['tx']
                ready=(done.get('state')=='complete_original_TX22_22_then_resumed_same_CF_native_and_owner' and not owners.same_owner(native,owners.process(native['pid'])))
            admitted=(ready and sum(v['bytes'] for v in workers.values())+11*2**30<24*2**30 and
                current['swap_used_bytes']-previous['swap_used_bytes']<64*2**20 and shutil.disk_usage(u.ROOT).free>16*2**30)
            good=good+1 if admitted else 0
            u.write(REPORT/'status.json',dict(state='waiting_for_full_original_temporal_completion_and_gpu_recovery_headroom',workers=workers,memory=current,good_samples=good,observed=stamp()))
            previous=current
            if good<2:time.sleep(10)
        baseline=current
        phases=[('prepare-preserved-fold-recovery',['upstream_split3_matched_resume.py']),
            ('three-missing-expanded-fits',['upstream_split3_matched_resume.py','--worker']),('independent-ten-fold-replay',['verify_upstream_split3_matched_resume.py'])]
        for phase,args in phases:
            check();high=0;command=[sys.executable,str(u.ROOT/'scripts'/args[0]),*args[1:]]
            with (REPORT/(phase+'.log')).open('x') as log:
                child=subprocess.Popen(['nice','-n','10',*command],cwd=u.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                u.write(REPORT/'launch.json',dict(phase=phase,pid=child.pid,command=command,observed=stamp()))
                while child.poll() is None:
                    check();workers=resources.footprints(child.pid);current=memory.memory();own=workers.get(str(child.pid),dict(bytes=0,rss_bytes=0))
                    high=high+1 if (sum(v['bytes'] for v in workers.values())>24*2**30 or sum(v['rss_bytes'] for v in workers.values())>24*2**30 or
                        own['bytes']>14*2**30 or own['rss_bytes']>14*2**30 or current['swap_used_bytes']-baseline['swap_used_bytes']>128*2**20) else 0
                    u.write(REPORT/'status.json',dict(state=phase,pid=child.pid,workers=workers,memory=current,consecutive_high_samples=high,observed=stamp()))
                    if high>=3 or shutil.disk_usage(u.ROOT).free<16*2**30:raise RuntimeError('Full matched resource guard exceeded; original partial artifacts preserved')
                    time.sleep(5)
                if child.returncode:raise RuntimeError('Full matched phase failed; actual artifacts preserved: '+phase)
                child=None
        proof=u.ROOT/'experiments/upstream-split3-matched-resume/independent-verification.json';result=u.read(proof)
        if result['state']!='complete_both_matched_arms_original_statistics_scalers_and_actual_saved_model_replays':raise ValueError('Complete actual replay admission required')
        u.write(REPORT/'completion.json',dict(state='complete_preserved_original_matched_study_and_all_actual_replays',observed=stamp(),verification_sha256=u.sha(proof),arms=result['arms'],qualifying_labels_read=False))
    except BaseException as error:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        u.write(REPORT/'failure.json',dict(state='incomplete_preserved',error=repr(error),observed=stamp()));raise

if __name__=='__main__':main()
