#!/usr/bin/env python3
"""Guard the complete bounded own330 neural-input admission beside native work."""
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

REPORT=u.ROOT/'experiments/upstream-own330-neural-inputs-queue'


def stamp():return dt.datetime.now(dt.timezone.utc).isoformat()


def safe(workers,current,baseline,free,reserve=0):
    return (sum(v['bytes'] for v in workers.values())+reserve<24*2**30 and
            sum(v['rss_bytes'] for v in workers.values())+reserve<24*2**30 and
            current['swap_used_bytes']-baseline['swap_used_bytes']<=128*2**20 and
            free>=18.75*2**30)


def main():
    REPORT.mkdir(exist_ok=False);child=None
    files=[Path(__file__),u.ROOT/'scripts/upstream_own330_neural_inputs.py',
           u.ROOT/'scripts/verify_upstream_own330_neural_inputs.py',
           u.ROOT/'scripts/upstream_unused_native_pairs.py',
           u.ROOT/'OWN330_NEURAL_PLAN.md',
           u.ROOT/'experiments/upstream-unused-native-pairs/verification.json',Path(resources.__file__)]
    frozen={str(p.relative_to(u.ROOT)):u.sha(p) for p in files}
    def check():
        for rel,digest in frozen.items():
            if u.sha(u.ROOT/rel)!=digest:raise ValueError('Frozen complete diagnostic changed')
    def interrupted(signum,frame):raise KeyboardInterrupt('Bounded diagnostic supervisor signal')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    u.write(REPORT/'plan.json',dict(created=stamp(),source_sha256=frozen,
            rule='Full own330406design/556nonlinear operators, all canonical rows and fixed164 additions. Bounded regular NPY row reads avoid multiGiB mapped resident pages. Reserve1.5GiB;24GiB total RSS/physical,1.5GiB child cap,128MiB swap-growth,18.75GiB disk floor. Two stable admission samples with compressor<5GiB. Full independent retained/new input bit replay; no fitting. No neural or qualifying fitting; no automatic retry. Guard suspends only this dedicated CPU child group, preserving RAM.'))
    try:
        previous=memory.memory();good=0
        while good<2:
            check();workers=resources.footprints();current=memory.memory()
            admitted=(safe(workers,current,previous,shutil.disk_usage(u.ROOT).free,1.5*2**30)
                      and current['compressor_bytes']<5*2**30
                      and current['swap_used_bytes']-previous['swap_used_bytes']<32*2**20)
            good=good+1 if admitted else 0;previous=current
            u.write(REPORT/'status.json',dict(state='waiting_for_bounded_complete_ridge_headroom',
                    workers=workers,memory=current,good_samples=good,observed=stamp()))
            if good<2:time.sleep(10)
        baseline=current
        for phase,script in [('complete-neural-operators','upstream_own330_neural_inputs.py'),
                             ('independent-complete-replay','verify_upstream_own330_neural_inputs.py')]:
            check()
            with (REPORT/(phase+'.log')).open('x') as log:
                child=subprocess.Popen([sys.executable,'-u',str(u.ROOT/'scripts'/script)],
                        cwd=u.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                u.write(REPORT/'launch.json',dict(phase=phase,pid=child.pid,command=child.args,observed=stamp()))
                while child.poll() is None:
                    check();workers=resources.footprints(child.pid);current=memory.memory()
                    own=workers.get(str(child.pid),dict(bytes=0,rss_bytes=0))
                    if not safe(workers,current,baseline,shutil.disk_usage(u.ROOT).free) or max(own['bytes'],own['rss_bytes'])>1.5*2**30:
                        raise RuntimeError('Bounded full-data diagnostic resource guard; child RAM preserved')
                    u.write(REPORT/'status.json',dict(state=phase,pid=child.pid,workers=workers,memory=current,observed=stamp()))
                    time.sleep(2)
            if child.returncode:raise RuntimeError('Full ridge phase failed without automatic retry: '+phase)
            child=None
        proof=u.ROOT/'experiments/upstream-own330-neural-inputs/independent-verification.json'
        result=u.read(proof)
        if result['state']!='complete_all_original330_design_and_nonlinear_input_bits_replayed':
            raise ValueError('Full independent actual ridge proof required')
        u.write(REPORT/'completion.json',dict(state='complete_original330_neural_input_operators_and_independent_all_bit_replay',
                verification_sha256=u.sha(proof),nonlinear_columns=556,nn_inputs=557,observed=stamp(),
                neural_models_fitted=0,qualifying_labels_read=False,no_global_score_promotion=True))
    except BaseException as error:
        if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGSTOP)
        u.write(REPORT/'failure.json',dict(error=repr(error),child_pid=child.pid if child else None,
                original_child_RAM_preserved_if_live=True,observed=stamp()))
        raise


if __name__=='__main__':main()
