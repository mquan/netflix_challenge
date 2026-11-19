#!/usr/bin/env python3
"""Resume original CF96 after temporal output completion and actual memory admission."""
import datetime as dt
import fcntl
import re
import subprocess
import sys
import time
from pathlib import Path
import upstream_repro as u

REPORT=u.ROOT/'experiments/upstream-cf96-memory-resume'
SCRIPT=u.ROOT/'scripts/upstream_cfnade_parallel.py'


def parse_memory(vm,swap):
    page=int(re.search(r'page size of (\d+) bytes',vm).group(1))
    compressed=int(re.search(r'Pages occupied by compressor:\s+(\d+)\.',vm).group(1))*page
    used=float(re.search(r'used =\s+([\d.]+)M',swap).group(1))*2**20
    return dict(compressor_bytes=compressed,swap_used_bytes=used)


def admitted(previous,current,footprint):
    # Footprint includes compressed/swapped anonymous pages; RSS alone does not.
    return (current['compressor_bytes']<10*2**30 and footprint<14*2**30 and
            current['swap_used_bytes']-previous['swap_used_bytes']<64*2**20)


def memory():
    return parse_memory(subprocess.check_output(['vm_stat'],text=True),
        subprocess.check_output(['sysctl','vm.swapusage'],text=True))


def live(pid,token):
    row=subprocess.run(['ps','-p',str(pid),'-o','stat=,command='],capture_output=True,text=True).stdout.strip()
    return bool(row and not row.split()[0].startswith('Z') and token in row)


def footprints():
    rows=subprocess.check_output(['ps','-axo','pid=,command='],text=True).splitlines();result={}
    for row in rows:
        pid,command=row.strip().split(None,1);executable=command.split()[0]
        native='/target/release/' in executable
        fitting='upstream_' in command and any(flag in command for flag in ['--worker ','--family '])
        if not (native or fitting):continue
        completed=subprocess.run(['vmmap','-summary',pid],capture_output=True,text=True)
        match=re.search(r'^Physical footprint:\s+([\d.]+)([KMG])',completed.stdout,re.M)
        if not match:
            if not subprocess.run(['ps','-p',pid,'-o','pid='],capture_output=True,text=True).stdout.strip():continue
            raise RuntimeError('Cannot account for live worker physical footprint: '+pid)
        result[pid]=dict(bytes=float(match.group(1))*{'K':2**10,'M':2**20,'G':2**30}[match.group(2)],command=command)
    return result


def preflight():
    plan=u.read(u.ROOT/'experiments/upstream-cfnade-parallel/plan.json')
    if u.sha(SCRIPT)!=plan['runner_sha256']:raise ValueError('Original CF96 controller changed')
    for pid in [35987,41719,41720]:
        if live(pid,'cfnade'):raise ValueError('Original yielded CF96 process still live')
    checkpoint_root=u.ROOT/'artifacts/upstream-cfnade-checkpoint/parallel-checkpoints'
    states={}
    for split in ['trainx','fulltrain']:
        base=checkpoint_root/split;pointer=u.read(base/'latest.json');directory=base/f"epoch-{pointer['epoch']:05}"
        state=u.read(directory/'state.json')
        if (u.sha(directory/'state.json','md5')!=pointer['manifest_md5'] or len(state['files'])!=84 or
                state['identity']['source']!=plan['source_id'] or state['identity']['data']!=plan['data_id'] or
                u.read(base/'owner.json')!=state['identity']):raise ValueError('Original fallback checkpoint differs')
        for name,digest in state['files'].items():
            if u.sha(directory/name,'md5')!=digest:raise ValueError('Original fallback tensor differs')
        states[split]=dict(epoch=pointer['epoch'],state_sha256=u.sha(directory/'state.json'),all84_tensors_verified=True)
    return states


def run():
    REPORT.mkdir(parents=True,exist_ok=True)
    with (REPORT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);states=preflight();previous=memory();good=0
        while True:
            ready=(u.ROOT/'experiments/upstream-phase-bank/manifest.json').exists()
            current=memory();owned=footprints() if ready else {};total=sum(v['bytes'] for v in owned.values())
            good=good+1 if ready and admitted(previous,current,total) else 0
            u.write(REPORT/'status.json',dict(state='waiting_for_temporal_bank_and_memory',observed=dt.datetime.now(dt.timezone.utc).isoformat(),
                original_phase_bank_ready=ready,memory=current,owned_footprint_bytes=total,owned_workers=owned,
                consecutive_good_samples=good,preserved_checkpoints=states))
            if good>=2:break
            if not ready and not live(46595,'scripts/upstream_phase_bank.py --watch'):
                raise RuntimeError('Original phase-bank watcher exited without a complete bank')
            previous=current;time.sleep(30)
        import upstream_phase_bank as bank
        bank.build();states=preflight()
        with (u.ROOT/'experiments/upstream-cfnade-parallel/run.lock').open('a') as shared:
            fcntl.flock(shared,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (REPORT/'child.log').open('a') as log:
            child=subprocess.Popen([sys.executable,'-u',str(SCRIPT)],cwd=u.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        u.write(REPORT/'launch-child.json',dict(pid=child.pid,command=child.args,script_sha256=u.sha(SCRIPT),
            observed=dt.datetime.now(dt.timezone.utc).isoformat(),checkpoints=states,admission=current,owned_footprint_bytes=total))
        u.write(REPORT/'status.json',dict(state='original_cf96_controller_resumed',child_pid=child.pid))
        code=child.wait();u.write(REPORT/'status.json',dict(state='original_controller_exited',returncode=code,child_pid=child.pid))
        if code:raise RuntimeError('Resumed original controller failed; no automatic retry')


if __name__=='__main__':
    try:run()
    except Exception as error:
        u.write(REPORT/'status.json',dict(state='incomplete_preserved',error=repr(error)));raise
