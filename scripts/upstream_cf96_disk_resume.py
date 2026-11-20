#!/usr/bin/env python3
"""Resume both exact full CF96 trajectories after verified disk-guard interruption."""
from pathlib import Path
import os
import shutil
import signal
import subprocess
import time
import upstream_cfnade_parallel as native
import upstream_remote_cf96 as checker
import upstream_cf96_primary_lane as owners
import upstream_knn3pair2_virtual_queue as resources
u=native.u
ORIGINAL_REPORT=native.REPORT
REPORT=u.ROOT/'experiments/upstream-cf96-disk-resume'
SUPERVISOR=92667
ORIGINAL_ADMIT=native.admit


def other_rss():
    rows=subprocess.check_output(['ps','-axo','pid=,pgid=,command='],text=True).splitlines()
    own={int(row.split(None,2)[1]) for row in rows if str(native.BASE/'target/release/cfnade-stage') in row}
    return resources.model.base.preceding.resources.other_rss(own)


def main():
    if REPORT.exists():raise ValueError('Existing exact disk recovery preserved')
    REPORT.mkdir()
    owner=owners.process(SUPERVISOR)
    if not owner or 'T' not in owner['state'] or str(Path(native.__file__)) not in owner['command'] or owners.children(SUPERVISOR):
        raise ValueError('Only the observed original idle paused supervisor may hand off')
    original_plan=ORIGINAL_REPORT/'plan.json';plan=u.read(original_plan)
    initial={};largest=0
    for phase,train in native.PHASES.items():
        query='probex' if phase=='x' else 'qual';rows=1128164 if phase=='x' else 2817131
        pointer,state,directory=checker.phase_checkpoint(native.BASE/'parallel-checkpoints'/train,
            plan['source_id'],plan['data_id'],query,rows)
        size=sum(p.stat().st_size for p in directory.iterdir() if p.is_file());largest=max(largest,size)
        initial[phase]=dict(epoch=pointer['epoch'],tensors=len(state['files']),checkpoint_bytes=size,
            state_sha256=u.sha(directory/'state.json'),pointer=pointer,all84_full_hashes_and_finite_values_verified=True)
    floor=16*2**30+2*largest+2*2**30
    paths=[Path(__file__),Path(native.__file__),Path(checker.__file__),original_plan]
    source_hashes={str(p.relative_to(u.ROOT)):u.sha(p) for p in paths}
    u.write(REPORT/'recovery-plan.json',dict(created=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
        old_idle_supervisor=owner,source_sha256=source_hashes,initial_checkpoints=initial,
        initial_disk_free_bytes=shutil.disk_usage(u.ROOT).free,admission_disk_floor_bytes=floor,
        policy='Authoritative original guards stopped both native handles at exact74/70 checkpoints. Original idle supervisor paused, with no native child. Both full84-array Adam/SWA/RNG/source/data/config states independently rehashed. Resume SAME original binaries/commands/environment and full95/95 budget. Original numerical source untouched. Separate report/lock namespace; initial disk admission uses original16GiB continuous floor plus two measured full checkpoint writes and2GiB extra margin, after lossless CoW storage recovery. Original6GiB memory reserve,26GiB admission,27GiB continuous native RSS and16GiB disk stop guard retained. Other-RSS now includes actual research processes. No smaller model, restart fromzero, partial completion claim or qualifying outcomes.',
        qualifying_labels_read=False))
    shutil.copyfile(original_plan,REPORT/'plan.json')
    for query in ['probex','qual']:
        p=native.TREE/'preds_new'/('cfnade-96.phase-'+query+'.out')
        shutil.copyfile(p,REPORT/('before-resume-'+query+'.out'))
    def admitted(voting,other,active,free):
        return voting=='complete' and other+sum(active)+6*2**30<26*2**30 and free>floor
    native.REPORT=REPORT;native.admit=admitted;native.common.other_rss=other_rss
    try:
        native.run()
        # The original supervisor remains the canonical finalizer. Restoring
        # complete phase receipts lets its existing loop finish without fitting.
        for phase in native.PHASES:
            if not native.phase_complete(phase):raise ValueError('Full original95 phase proof missing')
            shutil.copyfile(REPORT/(phase+'-complete.json'),ORIGINAL_REPORT/(phase+'-complete.json'))
        if not owners.same_owner(owner,owners.process(SUPERVISOR)) or owners.children(SUPERVISOR):
            raise ValueError('Original finalizer owner changed')
        os.kill(SUPERVISOR,signal.SIGCONT)
        while owners.same_owner(owner,owners.process(SUPERVISOR)):time.sleep(5)
        final=ORIGINAL_REPORT/'native-receipt.json';record=u.read(final)
        if record['plan_sha256']!=u.sha(original_plan):raise ValueError('Canonical final source receipt differs')
        for name,digest in record['outputs'].items():
            if u.sha(native.TREE/name)!=digest:raise ValueError('Canonical complete source output differs')
        u.write(REPORT/'recovery-completion.json',dict(state='complete_exact95_95_and_original_finalizer',
            original_receipt_sha256=u.sha(final),initial_checkpoints=initial,qualifying_labels_read=False))
    finally:
        if owners.same_owner(owner,owners.process(SUPERVISOR)) and 'T' in owners.process(SUPERVISOR)['state']:
            os.kill(SUPERVISOR,signal.SIGCONT)

if __name__=='__main__':
    try:main()
    except BaseException as error:
        REPORT.mkdir(parents=True,exist_ok=True)
        u.write(REPORT/'recovery-failure.json',dict(error=repr(error),all_original_states_preserved=True));raise
