"""Resume an exact preserved weekend native on its original96GiB host.

Run this file from a separate recovery checkout. It never edits the original
helper, prepares data, launches a model, resets a baseline or changes a recipe.
Every continuation after a new resource stop requires an explicit request.
"""
import argparse
import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import sys
import time

PLAN_SHA='2b8194eba4356db9a67af6d2b69fdb71816d2ba0b95df939b1cf122e86319514'
TREES={'s3-rbm400pf':'upstream-remote-weekend-r2','s3-tx300r':'upstream-remote-weekend-r4','s3-rbm800p':'upstream-remote-weekend-r3'}
DEADLINE=dt.datetime(2026,11,23,14,tzinfo=dt.timezone.utc)
GIB=2**30
RESERVE=64*2**20
# Immutable execution metadata from ee38c3ec91e8160250d159c423e1581640c96700
ORIGINAL_CONTROL_SHA={
    "s3-rbm400pf": {
        "ownership": "3d4bea9b5f3101b9c448320f748a417e1c97e0de792cd904fc663f4c29247f98",
        "owner": "c4e8e6de183fbeb564d3d59f47569f468b5424264251b81fe811270af32f7874",
        "launch": "096a070cd3f283a6910be474b43372af9b988a78afdb08b590a957ab1f5bc9d5",
        "launch-gate": "2fb2e70b63283a583d5181426f64613ae8fe397f11511ec3f4f6894cf7884072",
        "native-dispatch": "f40b51a309b1ca17d8f7cae2ee5a393ea18ddf060e8ca49a369c623ef9c033fb",
        "baseline": "6b35707a364909fbf9a3fe506093eaa03996af668717aaf41f8de7fa8dd36d29",
        "failure": "63e75b1c3fdb409630e8e3c07eee209baec1b92303d7b759a92abd5ae04ea960",
        "guard-sample": "52ffb3a253acfdf4af2fc560068917597d14fdcfeddb3bdafb38cd0884e8740f"
    },
    "s3-tx300r": {
        "ownership": "90d726ffccd9bd098eabc241bea623d0c226429ad45d966362192515af629c21",
        "owner": "ecd429b69ecb8844b05dd03b3b0b01b61c9ba442b0d2ebfe7e501c811e8817ac",
        "launch": "f0dfa8593dca3363cb490ac426338747884fc0f6921d843545fe41574c95bab1",
        "launch-gate": "df3da5cd3e17c13233d25b137fa99542a719876af7f7bf670d8ee5fe2fe8ffc1",
        "native-dispatch": "0a0b4e1073197e1c4a0ab571967d8b7df35266e15b86a41d261f5577cd835364",
        "baseline": "7cf964c731334daf5bfe3d97a8d11e76cc8e8975abc8966b39c25ed7d5d8542e",
        "failure": "b84486a1570f09eacf6fd9a03c7d072ff251991885e9b4baf1ee8e20473905a3",
        "guard-sample": "f5e58fdcdee81652f3adb84f272fef54198854599b4301c6d586ca4a8cfdf1dc"
    },
    "s3-rbm800p": {
        "ownership": "279409f9702fe066237d7afd5fc7ac47e536d5dcbabec9cbace3d302a38381c8",
        "owner": "4f7ab79ec2719a0d48a822db4c7a24455461fbad24e90395164775d39279342f",
        "launch": "405de8d01d3a3a3b8ba5b73800e3c8ff933cfbe6c819bc122b220114848bf661",
        "launch-gate": "6c1cf50af3aa4f386fd901792a7a397ac4407130ef6f5f8d5ad083054b378f23",
        "native-dispatch": "e8725bb2914d231c697ecb8ea01ce50aaa0f3eff95197b46e4d501e07c627143",
        "baseline": "7cf964c731334daf5bfe3d97a8d11e76cc8e8975abc8966b39c25ed7d5d8542e",
        "failure": "d9512fc7099a70467399b11062259f6ea0cb65721847420736b81c6b071b6def",
        "guard-sample": "fe2fbe9f055b9f77eb494a74dcb6112f467c1ba6a9267fa0d642b9793f0fdf81"
    }
}


def now():return dt.datetime.now(dt.timezone.utc)
def read(path):return json.loads(Path(path).read_text())
def identity(s):return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]


def require_host():
    if sys.platform!='darwin' or int(subprocess.check_output(['sysctl','-n','hw.memsize'],text=True))<96*GIB:
        raise ValueError('Adoption is only for the original96GiB remote host; never run locally')


def sha(path,guard=lambda:None):
    digest=hashlib.sha256();path=Path(path)
    with path.open('rb') as stream:
        before=os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):raise ValueError('Original bound regular file required')
        if sys.platform=='darwin':fcntl.fcntl(stream.fileno(),48,1)
        while True:
            guard();block=stream.read(2**20)
            if not block:break
            digest.update(block)
        if identity(before)!=identity(os.fstat(stream.fileno())) or identity(before)!=identity(path.stat()):
            raise ValueError('Original file identity changed during full hash')
    return digest.hexdigest()


def durable(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.staging')
    if path.exists() or temporary.exists():raise ValueError('Existing evidence retained; no overwrite/retry')
    with temporary.open('x') as stream:
        json.dump(value,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.link(temporary,path);sync(path.parent);temporary.unlink();sync(path.parent)


def sync(path):
    fd=os.open(path,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def status(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    with temporary.open('w') as stream:
        json.dump(value,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path);sync(path.parent)


def exact(captured,actual):
    return bool(actual and actual['pid']==captured['pid'] and actual['started']==captured['started']
        and actual['pgid']==captured['pid'] and actual['command']==captured['native_command'])


def stop(original,captured):
    if not exact(captured,original.identity.process(captured['pid'])):return False
    os.killpg(captured['pid'],signal.SIGSTOP);return True


def capacity(workers,current,baseline,native_pid,owner_pid,admission=False):
    own=workers.get(str(owner_pid));native=workers.get(str(native_pid)) if native_pid else None
    if not own or max(own['bytes'],own['rss_bytes'])>2*GIB:return False
    if native_pid and (not native or max(native['bytes'],native['rss_bytes'])>64*GIB):return False
    reserved={pid:dict(row) for pid,row in workers.items()}
    if admission:
        reserved[str(owner_pid)].update(bytes=2*GIB,rss_bytes=2*GIB)
        if native_pid:reserved[str(native_pid)].update(bytes=64*GIB,rss_bytes=64*GIB)
    return (sum(row['bytes'] for row in reserved.values())<80*GIB
        and sum(row['rss_bytes'] for row in reserved.values())<80*GIB
        and current['compressor_bytes']<(10 if admission else 12)*GIB
        and current['swap_used_bytes']-baseline['swap_used_bytes']<=128*2**20)


def load_original(root):
    root=Path(root).expanduser().resolve();plan_path=root/'experiments/upstream-remote-weekend/plan.json'
    if sha(plan_path)!=PLAN_SHA:raise ValueError('Exact original frozen weekend declaration required')
    plan=read(plan_path);helper=root/'scripts/upstream_remote_weekend.py'
    if sha(helper)!=plan['controller_sha256']['scripts/upstream_remote_weekend.py']:
        raise ValueError('Original training helper changed')
    sys.path.insert(0,str(root/'scripts'))
    spec=importlib.util.spec_from_file_location('_frozen_weekend_training_helper',helper)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if module.ROOT!=root:raise ValueError('Original helper must remain in original checkout')
    return module


def attest_runtime(original,tree,captured,job):
    actual=original.identity.process(captured['pid'])
    if not exact(captured,actual):raise ValueError('Exact original native PID/start/group/command required')
    result=subprocess.run(['/usr/sbin/lsof','-a','-p',str(captured['pid']),'-d','cwd,txt','-Ffn'],text=True,capture_output=True)
    if result.returncode or result.stderr.strip():raise ValueError('Original native cwd/executable attestation unavailable')
    kind=None;cwd=[];text=[]
    for line in result.stdout.splitlines():
        if line.startswith('f'):kind=line[1:]
        elif line.startswith('n'):
            if kind=='cwd':cwd.append(line[1:])
            elif kind=='txt':text.append(line[1:])
    if len(cwd)!=1 or not os.path.samefile(cwd[0],tree):raise ValueError('Original native cwd differs')
    if not any(Path(path).exists() and os.path.samefile(path,tree/'target/release/fit') for path in text):
        raise ValueError('Actual loaded fit binary must be the original admitted executable')
    expected=dict(job['environment'],VECLIB_MAXIMUM_THREADS='1')
    raw=subprocess.check_output(['ps','eww','-p',str(captured['pid']),'-o','command='],text=True)
    observed={key:re.findall(r'(?:^|\s)'+re.escape(key)+r'=([^\s]+)',raw) for key in expected}
    if any(observed[key]!=[str(value)] for key,value in expected.items()):raise ValueError('Preserved native environment differs from original launch')
    return dict(identity=actual,cwd=cwd[0],original_binary_vnode_matches=True,original_environment_matches=True)


def shapes(job):
    name=job['name'];result={f'preds_3/{column}.{query}.npy':(count,) for column in job['published_columns']
        for query,count in (('probe_3',1760494),('qual',2817131))}
    result.update({f'preds_3/{name}.{split}.npy':(count,) for split,count in (('train_3',98720013),('fulltrain',100480507))})
    if job['kind']=='rbm':
        width=5*int(job['environment'].get('NFACT',job['environment']['H']))
        result.update({f'preds_3/{name}.ifeat.{split}.npy':(17770,width) for split in ('train_3','fulltrain')})
    return result


class Session:
    def __init__(self,root,job):
        self.original=load_original(root);self.root=self.original.ROOT
        self.plan=read(self.original.PLAN);self.job=next(row for row in self.plan['jobs'] if row['name']==job)
        self.tree=self.root/'artifacts'/TREES[job]/'source';self.base=self.tree.parent;self.name=job
        self.report=self.root/'experiments/upstream-remote-weekend/recovery-v1'/job
        self.last=0.;self.data_identities={};self.pause_number=0;self.inputs_verified=False

    def original_evidence(self):
        files={key:self.base/(self.name+'-'+key+'.json') for key in ('ownership','owner','launch','launch-gate','native-dispatch','baseline','failure','guard-sample')}
        for key,path in files.items():
            if sha(path)!=ORIGINAL_CONTROL_SHA[self.name][key]:raise ValueError('Original pre-recovery execution expectation differs: '+key)
        records={key:read(path) for key,path in files.items()};prep_path=self.base/'preparation.json';prep=read(prep_path)
        captured,launch,gate,failure=records['ownership'],records['launch'],records['launch-gate'],records['failure']
        if (failure['error']!="RuntimeError('Weekend resource/watchdog guard; actual child RAM preserved')"
                or failure['actual_child_RAM_preserved'] is not True or failure['no_automatic_retry'] is not True
                or failure['child']!=captured or launch['pid']!=captured['pid'] or launch['job']!=self.name
                or launch['command']!=self.job['command'] or launch['environment']!=self.job['environment']
                or captured['native_command']!=' '.join(self.job['command']) or launch['plan_sha256']!=PLAN_SHA
                or launch['preparation_sha256']!=sha(prep_path) or prep['plan_sha256']!=PLAN_SHA
                or prep['state']!='prepared_own_data_only' or prep['qualifying_targets_zero'] is not True
                or prep['source_pin']!=self.plan['source_pin'] or prep['known_rows_absent_from_training'] is not True
                or prep['qualifying_labels_read'] is not False or prep['public_prediction_values_read'] is not False
                or launch['original_swap_baseline']!=records['baseline']
                or dt.datetime.fromisoformat(launch['deadline'])!=DEADLINE
                or gate['pid']!=captured['pid'] or gate['command']!=self.job['command']
                or gate['owner_sha256']!=sha(files['owner'])
                or records['native-dispatch']['plan_sha256']!=PLAN_SHA
                or records['native-dispatch']['environment']!=self.job['environment']):
            raise ValueError('Original launch, source/preparation, guard stop and first baseline must all agree')
        for old in (records['owner'],gate['watchdog']):
            if self.original.identity.same_owner(old,self.original.identity.process(old['pid'])):
                raise ValueError('Original owner and watchdog must both be terminal before adoption')
        actual=self.original.identity.process(captured['pid'])
        if not exact(captured,actual) or 'T' not in actual['state']:raise ValueError('Original preserved stopped RAM required; never replace a model')
        if (self.base/(self.name+'-receipt.json')).exists():raise ValueError('Existing native completion retained; inspect/export instead')
        self.captured=captured;self.baseline=records['baseline'];self.preparation=prep
        self.bindings={str(path):sha(path) for path in list(files.values())+[prep_path,self.original.PLAN,Path(__file__).resolve()]}
        for relative,digest in self.plan['controller_sha256'].items():self.bindings[str(self.root/relative)]=digest
        log=self.base/(self.name+'.log');payload=log.read_bytes();self.log_prefix=(len(payload),hashlib.sha256(payload).hexdigest(),identity(log.stat())[:2])
        trace=[int(value) for value in re.findall(rb'^Epoch\s+(\d+)\s*\|',payload,re.M)]
        full=list(range(1,self.job['epochs_per_phase']+1))*2
        if trace!=full[:len(trace)] or len(trace)>len(full):raise ValueError('Original complete-budget epoch prefix required; never restart or skip epochs')
        return dict(original_records=records,preparation_sha256=sha(prep_path),runtime=attest_runtime(self.original,self.tree,captured,self.job),
            original_log_prefix_bytes=len(payload),original_log_prefix_sha256=self.log_prefix[1],original_epoch_prefix=trace,original_baseline=records['baseline'])

    def check(self):
        for path,digest in self.bindings.items():
            if sha(path)!=digest:raise ValueError('Original source/control/baseline changed: '+path)
        for path,bound in self.data_identities.items():
            if identity(Path(path).stat())!=bound:raise ValueError('Original full-hashed source/data/binary identity changed')
        log=self.base/(self.name+'.log')
        if identity(log.stat())[:2]!=self.log_prefix[2] or log.stat().st_size<self.log_prefix[0]:raise ValueError('Original append-only training log changed')
        with log.open('rb') as stream:
            if hashlib.sha256(stream.read(self.log_prefix[0])).hexdigest()!=self.log_prefix[1]:raise ValueError('Preserved original log prefix differs')
        if not self.original.identity.same_owner(self.watch_identity,self.original.identity.process(self.watch_identity['pid'])):
            raise RuntimeError('Original recovery watchdog unavailable')

    def disk_remaining(self):
        total=0
        for relative,shape in shapes(self.job).items():
            expected=4
            for dimension in shape:expected*=dimension
            expected+=256;path=self.tree/relative
            if path.exists() and (path.is_symlink() or not path.is_file()):raise ValueError('Original regular native outputs required')
            total+=max(0,expected-(path.stat().st_size if path.exists() else 0))
        return total+RESERVE

    def measured(self,admission=False,terminal=False):
        workers=self.original.resources.footprints(os.getpid());current=self.original.memory.memory()
        native_pid=None if terminal else self.captured['pid']
        allowed=capacity(workers,current,self.baseline,native_pid,os.getpid(),admission)
        required=max(32*GIB if admission and not terminal else 20*GIB,20*GIB+self.disk_remaining())
        free=shutil.disk_usage(self.tree).free
        return allowed and free>=required,dict(workers=workers,memory=current,free_bytes=free,required_free_bytes=required)

    def admission(self,terminal=False):
        good=0
        while good<3:
            self.check()
            if not terminal:
                actual=self.original.identity.process(self.captured['pid'])
                if not exact(self.captured,actual):raise RuntimeError('Original preserved native handle disappeared')
                if 'T' not in actual['state']:
                    stop(self.original,self.captured);raise RuntimeError('Unexpected external continuation; original RAM preserved')
                if now()>=DEADLINE:raise RuntimeError('Original training deadline reached; preserve incomplete RAM, exclude producer')
            try:allowed,sample=self.measured(admission=True,terminal=terminal)
            except Exception as error:allowed=False;sample={'measurement_error':repr(error)}
            good=good+1 if allowed else 0
            try:status(self.report/'status.json',dict(state='waiting_original_stable_admission',stable_samples=good,sample=sample,observed=now().isoformat()))
            except OSError:pass
            if good<3:time.sleep(10)

    def hash_guard(self):
        if time.monotonic()-self.last>=10:self.admission(terminal=not exact(self.captured,self.original.identity.process(self.captured['pid'])));self.last=time.monotonic()

    def verify_inputs(self):
        expected={str(self.root/path):digest for path,digest in self.plan['controller_sha256'].items()}
        for base,bindings in ((self.tree,self.plan['source_files_sha256']),(self.tree/'data',self.plan['data_sha256']),
                              (self.tree/'data',self.plan['split3_sha256']),(self.tree,self.preparation['binary_sha256'])):
            for relative,digest in bindings.items():
                path=Path(relative)
                if path.is_absolute() or '..' in path.parts:raise ValueError('Safe original relative input required')
                expected[str(base/path)]=digest
        expected[str(self.base/'probex-row-map.npy')]=self.plan['row_map_sha256']
        for path,digest in expected.items():
            if sha(path,self.hash_guard)!=digest:raise ValueError('Original source/data/binary hash mismatch: '+path)
            self.data_identities[path]=identity(Path(path).stat())
        np=self.original.np
        targets=np.load(self.tree/'data/qual/ratings.npy',mmap_mode='r',allow_pickle=False)
        if targets.dtype!=np.int8 or targets.shape!=(2817131,) or np.any(targets):raise ValueError('Original zero qualifying targets required')
        self.inputs_verified=True

    def continue_once(self,number):
        if not self.inputs_verified:raise ValueError('Complete original input admission required before any native continuation')
        self.admission();attest_runtime(self.original,self.tree,self.captured,self.job);self.check()
        path=self.report/('initial-continuation.json' if number==0 else 'continuation-%03d.json'%number)
        durable(path,dict(native=self.captured,owner=self.owner,watchdog=self.watch_identity,
            original_baseline_sha256=sha(self.base/(self.name+'-baseline.json')),execution_plan_sha256=sha(self.report/'plan.json'),
            continuation_number=number,observed=now().isoformat(),same_model_RAM_preserved=True,recipe_and_environment_changed=False))
        actual=self.original.identity.process(self.captured['pid'])
        if (now()>=DEADLINE or not exact(self.captured,actual) or 'T' not in actual['state']
                or not self.original.identity.same_owner(self.watch_identity,self.original.identity.process(self.watch_identity['pid']))):
            raise ValueError('Exact stopped native/live watchdog and original deadline required at signal')
        os.killpg(self.captured['pid'],signal.SIGCONT)
        # A durable intent is not evidence that the signal succeeded. Other
        # jobs may use only this separate observed-running receipt or a fully
        # verified terminal400 completion as their ordering prerequisite.
        for _ in range(100):
            actual=self.original.identity.process(self.captured['pid'])
            if actual is None:return  # Complete outputs must independently qualify.
            if not exact(self.captured,actual):raise ValueError('Original native identity changed after continuation signal')
            if 'T' not in actual['state']:
                durable(self.report/('initial-running-observation.json' if number==0 else 'running-observation-%03d.json'%number),
                    dict(observed_native=actual,original_native=self.captured,execution_plan_sha256=sha(self.report/'plan.json'),
                        continuation_intent_sha256=sha(path),observed=now().isoformat(),actual_running_state_observed=True))
                return
            time.sleep(.05)
        raise RuntimeError('No running state observed after the single continuation signal; preserve RAM, do not retry')

    def monitor(self):
        while True:
            self.check();actual=self.original.identity.process(self.captured['pid'])
            if actual is None:return
            if not exact(self.captured,actual):raise ValueError('Original native identity differs')
            if now()>=DEADLINE:
                stop(self.original,self.captured);raise RuntimeError('Original deadline: incomplete native preserved and excluded')
            try:allowed,sample=self.measured()
            except Exception as error:allowed=False;sample={'measurement_error':repr(error)}
            if not allowed or 'T' in actual['state']:
                if not stop(self.original,self.captured):continue
                self.pause_number+=1
                try:durable(self.report/('pause-%03d.json'%self.pause_number),dict(native=self.captured,sample=sample,observed=now().isoformat(),explicit_request_required=True))
                except OSError:pass
                request=self.report/('continue-request-%03d.json'%self.pause_number)
                while not request.exists():
                    self.check()
                    if now()>=DEADLINE:raise RuntimeError('Deadline reached while original native remains preserved')
                    actual=self.original.identity.process(self.captured['pid'])
                    if not exact(self.captured,actual) or 'T' not in actual['state']:raise RuntimeError('Original paused handle changed without owned continuation')
                    try:status(self.report/'status.json',dict(state='same_native_RAM_preserved_explicit_continuation_request_required',pause_number=self.pause_number,observed=now().isoformat()))
                    except OSError:pass
                    time.sleep(10)
                value=read(request)
                if value['execution_plan_sha256']!=sha(self.report/'plan.json') or value['owner']!=self.owner or value['native']!=self.captured:
                    raise ValueError('Exact same-owner/source/native explicit continuation request required')
                self.continue_once(self.pause_number)
            try:status(self.report/'status.json',dict(state='same_original_full_budget_native_training',native=actual,observed=now().isoformat()))
            except OSError:pass
            time.sleep(10)

    def finalize(self):
        self.admission(terminal=True);self.verify_inputs();self.check()
        log=self.base/(self.name+'.log');self.original.trace_check(log.read_text(),self.job['epochs_per_phase'])
        if log.stat().st_mtime>DEADLINE.timestamp():raise ValueError('Native log extends beyond original training deadline')
        expected=shapes(self.job);out=self.tree/'preds_3';np=self.original.np
        if {p.name for p in out.glob(self.name+'*') if p.is_file()}-{self.name+'.out'}!={Path(name).name for name in expected}:
            raise ValueError('Every original epoch/terminal/training/factor output required')
        outputs={}
        for relative,shape in expected.items():
            self.hash_guard();path=self.tree/relative
            if path.stat().st_mtime>DEADLINE.timestamp():raise ValueError('Native output was written beyond original training deadline')
            value=np.load(path,mmap_mode='r',allow_pickle=False)
            if value.dtype!=np.float32 or value.shape!=shape or path.stat().st_size!=value.offset+value.nbytes:
                raise ValueError('Original full finite f32 output schema required')
            for lo in range(0,len(value),262144):
                self.hash_guard()
                if not np.isfinite(value[lo:lo+262144]).all():raise ValueError('Every original native output element must be finite')
            del value;outputs[relative]=sha(path,self.hash_guard)
        for query in ('probe_3','qual'):
            final=np.load(out/(self.name+'.'+query+'.npy'),mmap_mode='r',allow_pickle=False)
            last=np.load(out/('%s_ep%02d.%s.npy'%(self.name,self.job['epochs_per_phase'],query)),mmap_mode='r',allow_pickle=False)
            if not np.array_equal(final.view(np.uint32),last.view(np.uint32)):raise ValueError('Complete terminal and last epoch must match every bit')
        copied=out/(self.name+'.out')
        if copied.exists():
            if sha(copied)!=sha(log):raise ValueError('Original native log output differs')
        else:
            with copied.open('xb') as stream:stream.write(log.read_bytes());stream.flush();os.fsync(stream.fileno())
            sync(out)
        if sha(copied)!=sha(log):raise ValueError('Complete original log copy must match every byte')
        outputs[str(copied.relative_to(self.tree))]=sha(copied)
        code=Path(__file__).resolve().read_bytes()
        adoption=dict(execution_plan=read(self.report/'plan.json'),execution_plan_sha256=sha(self.report/'plan.json'),
            owner=self.owner,watchdog=self.watch_identity,controller_sha256=hashlib.sha256(code).hexdigest(),controller_utf8=code.decode(),
            evidence={p.name:read(p) for p in sorted(self.report.glob('*.json')) if p.name not in ('status.json','plan.json')},
            terminal_original_identity_absent=True,orphan_exit_status_observable=False,native_exit_code=None,
            original_launch_and_baseline_preserved=True,models_restarted=0,qualifying_labels_read=False)
        receipt=self.base/(self.name+'-receipt.json')
        durable(receipt,dict(state='native_complete_pending_export',observed=now().isoformat(),job=self.name,outputs=outputs,
            log_sha256=sha(log),launch_sha256=sha(self.base/(self.name+'-launch.json')),same_process_adoption=adoption))
        durable(self.report/'completion.json',dict(state='complete_same_native_full_original_epochs_and_all_outputs_pending_independent_export',receipt_sha256=sha(receipt),orphan_exit_code=None,qualifying_labels_read=False))


def watchdog(root,job,fd):
    require_host()
    session=Session(root,job);plan=read(session.report/'plan.json');owner=read(session.report/'owner.json');captured=plan['original_native']
    if identity(os.fstat(fd))[:2]!=identity((session.base/'operation.lock').stat())[:2]:raise ValueError('Original inherited operation lock descriptor required')
    durable(session.report/'watchdog-ready.json',dict(watchdog=session.original.identity.process(os.getpid()),owner=owner,execution_plan_sha256=sha(session.report/'plan.json')))
    try:
        while not (session.report/'owner-release.json').exists():
            if not session.original.identity.same_owner(owner,session.original.identity.process(owner['pid'])):
                stopped=stop(session.original,captured)
                try:durable(session.report/'watchdog-recovery.json',dict(native=captured,actual_RAM_preserved=stopped,original_operation_lock_retained=stopped,observed=now().isoformat()))
                except (OSError,ValueError):pass
                while exact(captured,session.original.identity.process(captured['pid'])):time.sleep(2)
                return
            time.sleep(2)
    finally:os.close(fd)


def require400_first(root,job):
    if job=='s3-rbm400pf':return
    first=Path(root)/'experiments/upstream-remote-weekend/recovery-v1/s3-rbm400pf';confirmed=first/'initial-running-observation.json'
    if confirmed.exists():
        value=read(confirmed);first_plan=read(first/'plan.json')
        if (value['actual_running_state_observed'] is not True or 'T' in value['observed_native']['state']
                or not exact(first_plan['original_native'],value['observed_native'])
                or value['execution_plan_sha256']!=sha(first/'plan.json')
                or value['continuation_intent_sha256']!=sha(first/'initial-continuation.json')):
            raise ValueError('Actual original400pf running observation must bind its plan and continuation intent')
    elif (first/'completion.json').exists():
        done=read(first/'completion.json');receipt=Path(root)/'artifacts'/TREES['s3-rbm400pf']/'s3-rbm400pf-receipt.json'
        if (done['state']!='complete_same_native_full_original_epochs_and_all_outputs_pending_independent_export'
                or done['receipt_sha256']!=sha(receipt)):
            raise ValueError('Original400pf full terminal proof required')
    else:raise ValueError('Preserve five completed400pf epochs: genuinely continue400pf first')


def run(root,job):
    require_host()
    session=Session(root,job)
    if session.report.exists():raise ValueError('Existing recovery namespace retained; use explicit continue or inspect')
    if now()>=DEADLINE:raise ValueError('Original training deadline already reached')
    require400_first(session.root,job)
    package=Path(__file__).resolve().parents[1]/'experiments/upstream-remote-weekend-adoption-protocol/verification.json'
    proof=read(package)
    if (proof['state']!='verified_remote_adoption_synthetic_protocol_only'
            or proof['source_sha256']['scripts/upstream_remote_weekend_adopt.py']!=sha(Path(__file__).resolve())):
        raise ValueError('Reviewed recovery package source required')
    lock=(session.base/'operation.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BaseException:lock.close();raise
    try:original_evidence=session.original_evidence();session.report.mkdir(parents=True)
    except BaseException:lock.close();raise
    sync(session.report);sync(session.report.parent);sync(session.report.parent.parent)
    session.owner=session.original.identity.process(os.getpid())
    durable(session.report/'owner.json',session.owner)
    durable(session.report/'plan.json',dict(state='declared_exact_original_stopped_native_adoption_without_restart',created=now().isoformat(),job=job,
        original_native=session.captured,original_evidence=original_evidence,original_bindings=session.bindings,deadline=DEADLINE.isoformat(),
        controller_sha256=sha(Path(__file__).resolve()),protocol_verification_sha256=sha(package),
        caps=dict(native_bytes=64*GIB,controller_bytes=2*GIB,total_research_bytes=80*GIB,compressor_bytes=12*GIB,admission_compressor_bytes=10*GIB,swap_growth_bytes=128*2**20),
        additional_continuations_require_explicit_requests=True,qualifying_labels_read=False))
    watch=None
    def interrupted(signum,frame):raise KeyboardInterrupt('Same-process adoption owner interrupted')
    signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGTERM,interrupted)
    try:
        with (session.report/'watchdog.log').open('x') as stream:
            watch=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--root',str(session.root),'--job',job,'--watch-fd',str(lock.fileno())],
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lock.fileno(),))
        session.watch_identity=session.original.identity.process(watch.pid)
        if not session.watch_identity or watch.poll() is not None:raise RuntimeError('Actual recovery watchdog required before native continuation')
        while not (session.report/'watchdog-ready.json').exists():
            if watch.poll() is not None:raise RuntimeError('Original recovery watchdog failed before readiness')
            time.sleep(.1)
        ready=read(session.report/'watchdog-ready.json')
        if (not session.original.identity.same_owner(ready['watchdog'],session.watch_identity)
                or ready['watchdog']['pgid']!=ready['watchdog']['pid'] or ready['owner']!=session.owner
                or ready['execution_plan_sha256']!=sha(session.report/'plan.json')):
            raise ValueError('Source-bound actual watchdog readiness required before continuation')
        durable(session.report/'watchdog-gate.json',dict(watchdog=session.watch_identity,owner=session.owner,execution_plan_sha256=sha(session.report/'plan.json')))
        session.admission();session.verify_inputs()
        durable(session.report/'input-admission.json',dict(state='complete_original_source_data_binary_rowmap_and_zero_target_admission',identities=session.data_identities,qualifying_labels_read=False))
        session.continue_once(0);session.monitor();session.finalize()
        durable(session.report/'owner-release.json',dict(state='original_native_terminal_all_outputs_verified_before_operation_lock_release'))
    except BaseException as error:
        stopped=stop(session.original,session.captured)
        try:durable(session.report/'failure.json',dict(error=repr(error),original_native=session.captured,actual_RAM_preserved=stopped,no_automatic_retry=True,observed=now().isoformat()))
        except (OSError,ValueError):pass
        if not exact(session.captured,session.original.identity.process(session.captured['pid'])):
            try:durable(session.report/'owner-release.json',dict(state='no_live_original_native_at_terminal_recovery_failure'))
            except (OSError,ValueError):pass
        raise
    finally:lock.close()


def request_continuation(root,job):
    require_host()
    session=Session(root,job);plan=read(session.report/'plan.json');owner=read(session.report/'owner.json');gate=read(session.report/'watchdog-gate.json')
    captured=plan['original_native'];current=read(session.report/'status.json')
    actual=session.original.identity.process(captured['pid'])
    if (now()>=DEADLINE or current['state']!='same_native_RAM_preserved_explicit_continuation_request_required'
            or not session.original.identity.same_owner(owner,session.original.identity.process(owner['pid']))
            or not session.original.identity.same_owner(gate['watchdog'],session.original.identity.process(gate['watchdog']['pid']))
            or not exact(captured,actual) or 'T' not in actual['state']):
        raise ValueError('Exact live owner/watchdog and original resource-stopped native required')
    durable(session.report/('continue-request-%03d.json'%current['pause_number']),dict(owner=owner,native=captured,
        execution_plan_sha256=sha(session.report/'plan.json'),explicit_same_process_continuation_request=True,observed=now().isoformat()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--job',choices=TREES,required=True)
    parser.add_argument('--adopt',action='store_true');parser.add_argument('--continue',dest='continue_native',action='store_true');parser.add_argument('--watch-fd',type=int)
    args=parser.parse_args()
    if args.watch_fd is not None:watchdog(args.root,args.job,args.watch_fd)
    elif args.adopt:run(args.root,args.job)
    elif args.continue_native:request_continuation(args.root,args.job)
    else:parser.error('Explicit --adopt or --continue required; import never signals or launches anything')
