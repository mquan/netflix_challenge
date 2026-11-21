"""Fixed local resource/ownership contract for the separate full-return adapter."""
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import upstream_repro as u
import upstream_complete_trajectory_ops as ops
import upstream_cf96_primary_lane as identity
import upstream_cf96_memory_resume as memory
import upstream_research_footprints as resources

ROOT=u.ROOT
PROTOCOL=ROOT/'experiments/upstream-remote-weekend-override-return-protocol/plan.json'
JOBS=('s3-rbm400pf','s3-tx300r','s3-rbm800p')
PHASES=('transport','independent_admission')
LIMIT=2*2**30
TOTAL=24*2**30
FLOOR=int(18.75*2**30)
META=64*2**20
ENV={k:'2' for k in ('OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}
ENV['PYTHONUNBUFFERED']='1'
MUST_BIND=('scripts/upstream_remote_weekend_override_runtime.py',
           'scripts/upstream_remote_weekend_override_worker.py',
           'scripts/upstream_remote_weekend_override_return_queue.py',
           'scripts/upstream_remote_weekend_override_handoff.py',
           'scripts/upstream_remote_weekend_override_context.py',
           'scripts/upstream_remote_weekend_override_policy.py',
           'scripts/upstream_remote_weekend_return.py',
           'scripts/verify_upstream_remote_weekend_return.py',
           'scripts/upstream_split3_pilot.py',
           'scripts/upstream_remote15h_native_stream.py',
           'experiments/upstream-remote-weekend/plan.json',
           'experiments/upstream-remote-weekend-owner-override-policy/remote-plan.json')


def read(path):return json.loads(Path(path).read_text())


def paths(job):
    if job not in JOBS:raise ValueError('Original full producer required')
    return (ROOT/'experiments/upstream-remote-weekend-override-return-queue'/job,
            ROOT/'experiments/upstream-remote-weekend-return'/job,
            ROOT/'artifacts/upstream-remote-weekend-return'/job)


def stamp():
    import datetime as dt
    return dt.datetime.now(dt.timezone.utc).isoformat()


def durable_directory(path):
    path.mkdir(parents=True,exist_ok=False)
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def write(path,value,exclusive=False):
    path=Path(path)
    if exclusive and path.exists():raise ValueError('Original evidence retained: '+str(path))
    temp=path.with_name(path.name+'.tmp')
    with temp.open('x') as f:
        f.write(json.dumps(value,indent=2)+'\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,path)
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def best_effort(path,value):
    try:write(path,value)
    except OSError:pass


def check_protocol():
    plan=read(PROTOCOL)
    if (plan['state']!='declared_full_weekend_owner_override_return_adapter' or plan['jobs']!=list(JOBS)
        or plan['phases']!=list(PHASES) or plan['environment']!=ENV or plan['worker_cap_bytes']!=LIMIT
        or plan['total_cap_bytes']!=TOTAL or plan['disk_floor_bytes']!=FLOOR
        or plan['numpy_version']!=importlib.metadata.version('numpy')
        or not set(MUST_BIND).issubset(plan['source_sha256'])):
        raise ValueError('Exact declared adapter runtime required')
    check_bindings(plan['source_sha256'])
    return plan


def check_bindings(bindings):
    for name,digest in bindings.items():
        p=Path(name)
        if p.is_absolute() or '..' in p.parts or ops.sha(ROOT/p)!=digest:
            raise ValueError('Original adapter source/preceding proof changed: '+name)


def safe(workers,current,baseline,free,reserve=0,disk_reserve=META):
    return (sum(w['bytes'] for w in workers.values())+reserve<TOTAL
        and sum(w['rss_bytes'] for w in workers.values())+reserve<TOTAL
        and current['compressor_bytes']<5*2**30
        and (baseline is None or current['swap_used_bytes']-baseline['swap_used_bytes']<=128*2**20)
        and free>=FLOOR+disk_reserve)


def ready(workers,current,baseline,free,reserve,disk_reserve):
    return safe(workers,current,baseline,free,reserve,disk_reserve) and current['compressor_bytes']<4.5*2**30


def continuation_ready(workers,pid,current,baseline,free,disk_reserve):
    own=workers.get(str(pid))
    if not own or max(own['bytes'],own['rss_bytes'])>LIMIT:return False
    padded=dict(workers);padded[str(pid)]=dict(own,bytes=LIMIT,rss_bytes=LIMIT)
    return ready(padded,current,baseline,free,0,disk_reserve)


def same_child(captured,actual):
    return bool(actual and actual['pid']==captured['pid'] and actual['started']==captured['started']
        and actual['pgid']==captured['pid'] and actual['command'] in (captured['command'],captured.get('worker_command')))


def stop(captured):
    if not captured or not same_child(captured,identity.process(captured['pid'])):return False
    os.killpg(captured['pid'],signal.SIGSTOP);return True


def worker_command(job,phase,lane_fd):
    if job not in JOBS or phase not in PHASES:raise ValueError('Original full return phase required')
    return [sys.executable,'-u',str(ROOT/'scripts/upstream_remote_weekend_override_worker.py'),
            '--job',job,'--phase',phase,'--lane-fd',str(lane_fd)]


def output_remaining(job):
    from upstream_remote_weekend_override_policy import REMOTE_PLAN_SHA
    policy=ROOT/'experiments/upstream-remote-weekend-owner-override-policy/remote-plan.json'
    if ops.sha(policy)!=REMOTE_PLAN_SHA:raise ValueError('Pinned policy plan changed')
    spec=next(j for j in read(policy)['jobs'] if j['name']==job)
    _,report,art=paths(job);used=0;seen=set()
    for folder in (report,art):
        if not folder.exists():continue
        for p in folder.rglob('*'):
            if p.is_symlink():raise ValueError('Owned regular materialized files required')
            if p.is_file():
                s=p.stat();key=(s.st_dev,s.st_ino)
                if key not in seen:seen.add(key);used+=s.st_size
    return max(META,len(spec['published_columns'])*40*2**20+128*2**20-used)


def validate_lane(lane_fd):
    lock=ROOT/'experiments/full-upstream-reproduction/runner.lock'
    a,b=os.fstat(lane_fd),lock.stat()
    if (a.st_dev,a.st_ino)!=(b.st_dev,b.st_ino) or not os.get_inheritable(lane_fd):
        raise ValueError('Inherited original primary-lane descriptor required')
    with lock.open('a') as probe:
        try:fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:pass
        else:
            fcntl.flock(probe,fcntl.LOCK_UN);raise ValueError('Primary lane must already be locked')
    fcntl.flock(lane_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)  # Must share the actual admitted lock description.


class Guard:
    def __init__(self,job,phase,lane_fd):
        if phase not in PHASES:raise ValueError('Original complete return phase required')
        self.job,self.phase=job,phase;self.queue,_,_=paths(job);self.plan=check_protocol()
        queue_plan=read(self.queue/'plan.json')
        if queue_plan['job']!=job or queue_plan['protocol_sha256']!=ops.sha(PROTOCOL):
            raise ValueError('Exact source-bound producer queue required')
        self.owner=read(self.queue/'monitor.json');self.captured=read(self.queue/(phase+'-ownership.json'))
        self.gate=read(self.queue/(phase+'-gate.json'));self.baseline=read(self.queue/'first-scientific-baseline.json')
        self.baseline_sha=ops.sha(self.queue/'first-scientific-baseline.json');self.last=0.
        actual=identity.process(os.getpid());command=worker_command(job,phase,lane_fd)
        if (not same_child(self.captured,actual) or self.captured['pid']!=os.getpid() or actual['command']!=' '.join(command)
            or self.gate['pid']!=os.getpid() or self.gate['phase']!=phase or self.gate['command']!=command
            or self.gate['protocol_sha256']!=ops.sha(PROTOCOL)
            or self.gate['queue_plan_sha256']!=ops.sha(self.queue/'plan.json')
            or self.gate['owner_sha256']!=ops.sha(self.queue/'monitor.json') or self.gate['baseline_sha256']!=self.baseline_sha
            or any(os.environ.get(k)!=v for k,v in ENV.items())):
            raise ValueError('Exact live admitted source-bound worker gate required')
        validate_lane(lane_fd)
        self.check()

    def check(self,extra_disk_bytes=0):
        while True:
            now=time.monotonic();free=0;error=None;reserve=max(META,extra_disk_bytes)
            try:
                free=shutil.disk_usage(ROOT).free
                reserve=max(reserve,output_remaining(self.job))
            except OSError as exc:error=repr(exc)
            if not error and now-self.last<3 and free>=FLOOR+reserve:return
            check_protocol();check_bindings(read(self.queue/'plan.json')['source_sha256'])
            check_bindings(read(self.queue/'prerequisites.json')['source_sha256'])
            check_bindings(self.gate['predecessor_sha256'])
            if ops.sha(self.queue/'first-scientific-baseline.json')!=self.baseline_sha:
                raise ValueError('Original first-scientific baseline changed')
            try:
                workers,current=resources.footprints(os.getpid()),memory.memory();own=workers.get(str(os.getpid()))
                valid=bool(not error and own and max(own['bytes'],own['rss_bytes'])<=LIMIT
                    and safe(workers,current,self.baseline,free,disk_reserve=reserve)
                    and identity.same_owner(self.owner,identity.process(self.owner['pid']))
                    and identity.same_owner(self.gate['watchdog'],identity.process(self.gate['watchdog']['pid'])))
                sample=dict(workers=workers,memory=current,free_bytes=free,disk_error=error)
            except Exception as exc:valid=False;sample=dict(measurement_error=repr(exc),free_bytes=free)
            if valid:self.last=now;return
            best_effort(self.queue/(self.phase+'-worker-resource-request.json'),dict(phase=self.phase,child=self.captured,
                sample=sample,extra_disk_reserve=reserve,baseline_sha256=self.baseline_sha,observed=stamp()))
            self.last=0.  # Always remeasure both memory domains after a same-frame continuation.
            if not stop(self.captured):raise RuntimeError('Original dedicated return worker identity disappeared')
            # The same frame/RAM/stream and inherited lane remain until original readmission.
