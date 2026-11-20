#!/usr/bin/env python3
"""Isolated full-budget original producers; never reads recovered qual labels."""
import argparse
import datetime as dt
import hashlib
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import tarfile
import signal
import sys
import upstream_cf96_primary_lane as identity
import upstream_cf96_memory_resume as memory
import upstream_research_footprints as resources
import numpy as np
from upstream_split3_pilot import arrays, projection, assert_disjoint, keys

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'experiments/upstream-remote-weekend/plan.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        stream.write(json.dumps(obj, indent=2) + '\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def environment(knobs=None):
    keep = {'PATH', 'HOME', 'TMPDIR', 'USER', 'LOGNAME', 'SHELL', 'LANG',
            'LC_ALL', 'CARGO_HOME', 'RUSTUP_HOME', 'SDKROOT',
            'MACOSX_DEPLOYMENT_TARGET', 'DYLD_LIBRARY_PATH'}
    env = {k: v for k, v in os.environ.items() if k in keep}
    env.update(RAYON_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
               VECLIB_MAXIMUM_THREADS='1')
    env.update(knobs or {})
    return env


def check_files(root, files):
    for rel, expected in files.items():
        if sha(root / rel) != expected:
            raise ValueError('Immutable file differs: ' + str(root / rel))


def vector(path, count):
    a = np.load(path, mmap_mode='r', allow_pickle=False)
    if a.shape != (count,) or a.dtype != np.float32 or not np.isfinite(a).all():
        raise ValueError('Invalid float32 prediction: ' + str(path))
    return a


def trace_check(log, epochs):
    trace = [int(n) for n in re.findall(r'^Epoch\s+(\d+)\s*\|', log, re.M)]
    if trace != list(range(1, epochs + 1)) * 2:
        raise ValueError('Both complete original epoch traces required')


def verify_inputs(tree):
    plan = read(PLAN)
    check_files(ROOT, plan['controller_sha256'])
    receipt = read(tree.parent / 'preparation.json')
    if receipt['plan_sha256'] != sha(PLAN):
        raise ValueError('Frozen plan changed')
    check_files(tree, plan['source_files_sha256'])
    check_files(tree / 'data', plan['data_sha256'])
    check_files(tree / 'data', plan['split3_sha256'])
    check_files(tree, receipt['binary_sha256'])
    if np.any(np.load(tree / 'data/qual/ratings.npy', mmap_mode='r')):
        raise ValueError('Qualifying targets must be zero')
    return plan


def prepare(source, data, tree):
    plan = read(PLAN)
    check_files(ROOT, plan['controller_sha256'])
    if tree.exists() or (tree.parent / 'preparation.json').exists():
        raise ValueError('Existing preparation preserved; use verify-inputs')
    if shutil.disk_usage(tree.parent).free < 20 * 2**30:
        raise ValueError('Need at least 20 GiB free before preparation')
    check_files(source, plan['source_files_sha256'])
    check_files(data, plan['data_sha256'])
    if np.any(np.load(data / 'qual/ratings.npy', mmap_mode='r')):
        raise ValueError('Qualifying targets must be zero')
    tree.mkdir()
    for rel in plan['source_files_sha256']:
        target = tree / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / rel, target)
    (tree / 'data').mkdir()
    for split in ['train', 'probe', 'fulltrain', 'qual', 'probex']:
        (tree / 'data' / split).symlink_to(data / split, target_is_directory=True)
    with (tree.parent / 'build.log').open('x') as log:
        subprocess.run(['cargo', 'build', '--release', '--locked', '--features',
                        'blas', '--bin', 'fit', '--bin', 'split_3'], cwd=tree,
                       env=environment(), stdout=log, stderr=subprocess.STDOUT, check=True)
    with (tree.parent / 'split3.log').open('x') as log:
        subprocess.run(['./target/release/split_3'], cwd=tree, env=environment(),
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    check_files(tree / 'data', plan['split3_sha256'])
    source_rows = arrays(tree / 'data/probe_3', labels=True)
    known = arrays(tree / 'data/probex', labels=True)
    mapping = projection(source_rows, known)
    assert_disjoint(np.sort(keys(*known[:3])), arrays(tree / 'data/train_3'))
    rowmap = tree.parent / 'probex-row-map.npy'
    np.save(rowmap, mapping)
    if sha(rowmap) != plan['row_map_sha256']:
        raise ValueError('Exact coordinator row map differs')
    write(tree.parent / 'preparation.json', dict(state='prepared_own_data_only',
          observed=stamp(), plan_sha256=sha(PLAN), source_pin=plan['source_pin'],
          binary_sha256={f'target/release/{n}': sha(tree / 'target/release' / n)
                         for n in ['fit', 'split_3']},
          row_map_sha256=sha(rowmap), known_rows_absent_from_training=True,
          qualifying_targets_zero=True, qualifying_labels_read=False,
          public_prediction_values_read=False))
    verify_inputs(tree)


def capacity(workers, current, baseline, own=None, reserve=0):
    return (sum(w['bytes'] for w in workers.values())+reserve < 80*2**30 and
            sum(w['rss_bytes'] for w in workers.values())+reserve < 80*2**30 and
            (own is None or max(own['bytes'],own['rss_bytes']) <= 64*2**30) and
            current['compressor_bytes'] < 12*2**30 and
            current['swap_used_bytes']-baseline['swap_used_bytes'] <= 128*2**20)


def same_child(captured,actual):
    return bool(actual and actual['pid']==captured['pid'] and actual['started']==captured['started'] and
        actual['pgid']==captured['pid'] and actual['command'] in [captured['command'],captured.get('native_command')])


def bootstrap(tree,name,owner_pid):
    # No model starts until ownership and a live watchdog are acknowledged.
    plan=read(PLAN);check_files(ROOT,plan['controller_sha256'])
    job=next(j for j in plan['jobs'] if j['name']==name)
    owner=read(tree.parent/(name+'-owner.json'))
    if owner['pid']!=owner_pid:raise ValueError('Exact declared launch owner required')
    captured=identity.process(os.getpid());captured['native_command']=' '.join(job['command'])
    if captured['pgid']!=captured['pid']:raise ValueError('Dedicated bootstrap session required')
    write(tree.parent/(name+'-ownership.json'),captured)
    gate=tree.parent/(name+'-launch-gate.json')
    while True:
        if not identity.same_owner(owner,identity.process(owner_pid)):
            os.killpg(os.getpid(),signal.SIGSTOP);continue
        if gate.exists():
            declared=read(gate)
            if (declared['pid']!=os.getpid() or declared['command']!=job['command'] or
                declared['owner_sha256']!=sha(tree.parent/(name+'-owner.json')) or
                not identity.same_owner(declared['watchdog'],identity.process(declared['watchdog']['pid']))):
                raise ValueError('Exact live owner/watchdog/native launch gate required')
            write(tree.parent/(name+'-native-dispatch.json'),dict(pid=os.getpid(),command=job['command'],
                environment=job['environment'],plan_sha256=sha(PLAN),observed=stamp()))
            os.execvpe(job['command'][0],job['command'],environment(job['environment']))
        time.sleep(.1)


def watchdog(tree, name, owner_pid):
    owner=read(tree.parent/(name+'-owner.json'))
    if owner['pid'] != owner_pid: raise ValueError('Exact weekend owner required')
    release=tree.parent/(name+'-owner-release.json')
    while not release.exists():
        if not identity.same_owner(owner,identity.process(owner_pid)):
            if release.exists():return
            path=tree.parent/(name+'-ownership.json');captured=read(path) if path.exists() else None;stopped=False
            actual=identity.process(captured['pid']) if captured else None
            if captured and same_child(captured,actual):
                os.killpg(captured['pid'],signal.SIGSTOP);stopped=True
            write(tree.parent/(name+'-watchdog-recovery.json'),dict(child=captured,actual_child_RAM_preserved=stopped,observed=stamp()))
            return
        time.sleep(2)


def run_job(tree, name, deadline):
    plan=verify_inputs(tree);job=next(j for j in plan['jobs'] if j['name']==name)
    if int(subprocess.check_output(['sysctl','-n','hw.memsize'],text=True)) < 96*2**30:
        raise ValueError('This full weekend reservation requires the expected96GiB host; report actual hardware')
    out=tree/'preds_3';logpath=tree.parent/(name+'.log')
    if logpath.exists() or (out.exists() and list(out.glob(name+'*'))):
        raise ValueError('Existing native evidence preserved; never automatically restart')
    baseline=memory.memory();good=0
    write(tree.parent/(name+'-baseline.json'),baseline)
    while good<3:
        if (deadline-dt.datetime.now(dt.timezone.utc)).total_seconds()<job['minimum_remaining_training_hours']*3600:
            raise ValueError('Insufficient time for full original producer; no launch')
        workers,current=resources.footprints(os.getpid()),memory.memory()
        ready=(capacity(workers,current,baseline,reserve=64*2**30) and current['compressor_bytes']<10*2**30 and
               shutil.disk_usage(tree).free>=32*2**30)
        good=good+1 if ready else 0
        write(tree.parent/(name+'-status.json'),dict(state='waiting_original_full_capacity',good_samples=good,
              workers=workers,memory=current,observed=stamp()))
        if good<3:time.sleep(10)
    owner=identity.process(os.getpid());write(tree.parent/(name+'-owner.json'),owner)
    child=None;captured=None;watch=None
    def interrupted(signum,frame):raise KeyboardInterrupt('Weekend native monitor interrupted')
    signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGTERM,interrupted)
    try:
        with logpath.open('x') as log:
            with (tree.parent/(name+'-watchdog.log')).open('x') as watcher_log:
                watch=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--watch',str(os.getpid()),
                      '--tree',str(tree),'--job',name],cwd=ROOT,stdout=watcher_log,stderr=subprocess.STDOUT,start_new_session=True)
            child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--bootstrap',str(os.getpid()),
                '--tree',str(tree),'--job',name],cwd=tree,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            ownership=tree.parent/(name+'-ownership.json')
            while not ownership.exists():
                if child.poll() is not None or watch.poll() is not None:raise RuntimeError('Native launch handshake unavailable')
                time.sleep(.1)
            captured=read(ownership)
            if captured['pid']!=child.pid or not same_child(captured,identity.process(child.pid)):
                raise ValueError('Exact dedicated native bootstrap required')
            write(tree.parent/(name+'-launch.json'),dict(job=name,pid=child.pid,started=stamp(),deadline=deadline.isoformat(),
                  command=job['command'],environment=job['environment'],plan_sha256=sha(PLAN),
                  preparation_sha256=sha(tree.parent/'preparation.json'),original_swap_baseline=baseline))
            watcher=identity.process(watch.pid)
            if not watcher or watch.poll() is not None:raise RuntimeError('Live watchdog required before native dispatch')
            write(tree.parent/(name+'-launch-gate.json'),dict(pid=child.pid,command=job['command'],
                owner_sha256=sha(tree.parent/(name+'-owner.json')),watchdog=watcher))
            while child.poll() is None:
                if dt.datetime.now(dt.timezone.utc)>=deadline:
                    child.terminate()
                    try:child.wait(timeout=30)
                    except subprocess.TimeoutExpired:child.kill();child.wait()
                    write(tree.parent/(name+'-incomplete.json'),dict(state='deadline_interrupted_not_eligible',observed=stamp()))
                    raise RuntimeError('Explicit deadline reached; incomplete producer excluded')
                workers,current=resources.footprints(child.pid),memory.memory()
                if child.poll() is not None:break
                own=workers.get(str(child.pid))
                if (not own or not capacity(workers,current,baseline,own) or shutil.disk_usage(tree).free<20*2**30 or watch.poll() is not None):
                    write(tree.parent/(name+'-guard-sample.json'),dict(workers=workers,memory=current,observed=stamp()))
                    raise RuntimeError('Weekend resource/watchdog guard; actual child RAM preserved')
                check_files(ROOT,plan['controller_sha256'])
                write(tree.parent/(name+'-status.json'),dict(state='full_original_native_training',pid=child.pid,
                    workers=workers,memory=current,observed=stamp()));time.sleep(10)
            if child.returncode:raise RuntimeError('Native failed; do not retry')
        trace_check(logpath.read_text(),job['epochs_per_phase'])
        shutil.copyfile(logpath,out/(name+'.out'))
        outputs={str(p.relative_to(tree)):sha(p) for p in sorted(out.glob(name+'*')) if p.is_file()}
        if not set(job['original_registry_job']['outputs']).issubset(outputs):raise ValueError('Original outputs incomplete')
        for column in job['published_columns']:
            for query,count in [('probe_3',1760494),('qual',2817131)]:vector(tree/f'preds_3/{column}.{query}.npy',count)
        write(tree.parent/(name+'-receipt.json'),dict(state='native_complete_pending_export',observed=stamp(),job=name,
              outputs=outputs,log_sha256=sha(logpath),launch_sha256=sha(tree.parent/(name+'-launch.json'))))
        write(tree.parent/(name+'-status.json'),dict(state='complete_native_pending_independent_export',observed=stamp()))
        write(tree.parent/(name+'-owner-release.json'),dict(state='owned_native_terminal_after_full_original_trace_and_outputs'))
    except BaseException as error:
        stopped=False
        if child is not None and child.poll() is None:
            actual=identity.process(child.pid)
            if captured is None or same_child(captured,actual):
                os.killpg(child.pid,signal.SIGSTOP);stopped=True
        write(tree.parent/(name+'-failure.json'),dict(error=repr(error),child=captured,actual_child_RAM_preserved=stopped,
              no_automatic_retry=True,observed=stamp()))
        raise


def native_shape(relative,job):
    if not relative.endswith('.npy'):return None
    stem=Path(relative).name
    for query,count in [('probe_3',1760494),('qual',2817131)]:
        if stem.endswith('.'+query+'.npy'):return (count,)
    for split,count in [('train_3',98720013),('fulltrain',100480507)]:
        if stem==job['name']+'.'+split+'.npy':return (count,)
        if stem==job['name']+'.ifeat.'+split+'.npy' and job['kind']=='rbm':
            width=5*int(job['environment'].get('NFACT',job['environment']['H']))
            return (17770,width)
    raise ValueError('Unexpected native side output schema: '+relative)


def export(tree, name, dest):
    # Recheck source/data/binary, both traces, all outputs, then independently
    # reconstruct the known-row projection rather than trusting the saved map.
    plan = verify_inputs(tree)
    job = next(j for j in plan['jobs'] if j['name'] == name)
    receipt_path = tree.parent / (name + '-receipt.json')
    receipt = read(receipt_path)
    log = tree.parent / (name + '.log')
    launch_path = tree.parent / (name + '-launch.json')
    launch = read(launch_path)
    if (receipt['job'] != name or receipt['state'] != 'native_complete_pending_export'
            or receipt['log_sha256'] != sha(log)
            or receipt['launch_sha256'] != sha(launch_path)
            or launch['plan_sha256'] != sha(PLAN)
            or launch['preparation_sha256'] != sha(tree.parent / 'preparation.json')
            or launch['command'] != job['command'] or launch['environment'] != job['environment']):
        raise ValueError('Native receipt/launch binding differs')
    trace_check(log.read_text(), job['epochs_per_phase'])
    check_files(tree, receipt['outputs'])
    for relative in receipt['outputs']:
        if relative.endswith('.npy'):
            values=np.load(tree/relative,mmap_mode='r',allow_pickle=False)
            if values.dtype!=np.dtype('float32') or values.shape!=native_shape(relative,job):raise ValueError('Exact full native training/query/factor shape required')
            for lo in range(0,len(values),262144):
                if not np.isfinite(values[lo:lo+262144]).all():raise ValueError('Nonfinite native side output')
    if not set(job['original_registry_job']['outputs']).issubset(receipt['outputs']):
        raise ValueError('Original outputs incomplete')
    for rel in receipt['outputs']:
        if rel.endswith('.probe_3.npy'):
            vector(tree / rel, 1760494)
        elif rel.endswith('.qual.npy'):
            vector(tree / rel, 2817131)
    mapping = projection(arrays(tree / 'data/probe_3', labels=True),
                         arrays(tree / 'data/probex', labels=True))
    if not np.array_equal(mapping, np.load(tree.parent / 'probex-row-map.npy')):
        raise ValueError('Independent row projection differs')
    for split in ['probe_3', 'qual']:
        final = vector(tree / f'preds_3/{name}.{split}.npy', 1760494 if split == 'probe_3' else 2817131)
        last = vector(tree / f"preds_3/{name}_ep{job['epochs_per_phase']:02}.{split}.npy", len(final))
        if not np.array_equal(final.view(np.uint32), last.view(np.uint32)):
            raise ValueError('Terminal and last full epoch differ')
    if dest.exists():
        raise ValueError('Existing export preserved')
    dest.mkdir(parents=True)
    records = []
    for column in job['published_columns']:
        known = vector(tree / f'preds_3/{column}.probe_3.npy', 1760494)[mapping]
        qual = vector(tree / f'preds_3/{column}.qual.npy', 2817131)
        target = dest / (column + '.npz')
        np.savez_compressed(target, probex=known, qual=qual)
        with np.load(target, allow_pickle=False) as saved:
            for key, expected in [('probex', known), ('qual', qual)]:
                if not np.array_equal(saved[key].view(np.uint32), expected.view(np.uint32)):
                    raise ValueError('Lossless transfer differs')
        records.append(dict(column=column, file=target.name, sha256=sha(target),
                            native_probe_sha256=sha(tree / f'preds_3/{column}.probe_3.npy'),
                            native_qual_sha256=sha(tree / f'preds_3/{column}.qual.npy')))
    for path in [PLAN, receipt_path, launch_path, log, tree.parent / 'preparation.json',
                 tree.parent / 'build.log', tree.parent / 'split3.log', Path(__file__)]:
        shutil.copyfile(path, dest / path.name)
    native_files={}
    for relative,digest in receipt['outputs'].items():
        source=tree/relative;target=dest/'native'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        os.link(source,target)
        if sha(target)!=digest:raise ValueError('Every complete native payload must replay in return package')
        native_files[str(target.relative_to(dest))]=digest
    source_files={}
    for relative,digest in {**plan['source_files_sha256'],**read(tree.parent/'preparation.json')['binary_sha256']}.items():
        target=dest/'source'/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(tree/relative,target)
        if sha(target)!=digest:raise ValueError('Original source/binary return differs')
        source_files[str(target.relative_to(dest))]=digest
    write(dest / 'verification.json', dict(state='verified_complete_original_remote_weekend_producer',
          observed=stamp(), job=name, source_pin=plan['source_pin'], records=records,
          all_native_output_sha256=native_files,original_source_and_binary_sha256=source_files,
          source_data_binary_rehashed=True, both_full_original_epoch_traces=True,
          independent_known_projection=True, qualifying_labels_read=False,
          public_prediction_values_read=False, files_sha256={p.name: sha(p) for p in dest.iterdir() if p.is_file()},
          scope='Own numerical training outputs only. No blend or goal-success claim.'))


def archive(tree,name,dest):
    verify_inputs(tree)
    source=tree.parent/'exports'/name
    proof=read(source/'verification.json')
    if proof['state']!='verified_complete_original_remote_weekend_producer' or proof['job']!=name:
        raise ValueError('Full verified original export required')
    expected={**proof['files_sha256'],**proof['all_native_output_sha256'],**proof['original_source_and_binary_sha256'],
              'verification.json':sha(source/'verification.json')}
    files=[p for p in source.rglob('*') if p.is_file()]
    if any(p.is_symlink() for p in source.rglob('*')) or set(str(p.relative_to(source)) for p in files)!=set(expected):
        raise ValueError('Exact owned export files only; no raw dataset or unbound files')
    for relative,digest in expected.items():
        if sha(source/relative)!=digest:raise ValueError('Original verified export changed')
    total=sum(p.stat().st_size for p in files)
    if shutil.disk_usage(tree).free<20*2**30+total:raise ValueError('Reserve full archive write above20GiB disk floor')
    if dest.exists():raise ValueError('Existing immutable archive retained; no automatic repack')
    dest.mkdir(parents=True);buckets=[];bucket=[];size=0
    for relative in sorted(expected):
        count=(source/relative).stat().st_size
        if count>1024*2**20:raise ValueError('Unexpected single native file above1GiB; preserve original export')
        if bucket and size+count>1024*2**20:buckets.append(bucket);bucket=[];size=0
        bucket.append(relative);size+=count
    if bucket:buckets.append(bucket)
    records=[]
    for index,bucket in enumerate(buckets,1):
        target=dest/(name+'-part%02d.tar.gz'%index)
        with tarfile.open(target,'w:gz',compresslevel=3,dereference=True) as bundle:
            for relative in bucket:bundle.add(source/relative,arcname=name+'/'+relative,recursive=False)
        with target.open('rb') as stream:os.fsync(stream.fileno())
        members={}
        with tarfile.open(target,'r:gz') as bundle:
            for member in bundle:
                if not member.isfile() or not member.name.startswith(name+'/'):raise ValueError('Regular exact archive member required')
                relative=member.name[len(name)+1:]
                if relative not in bucket or relative in members:raise ValueError('Unexpected or duplicate archive member')
                digest=hashlib.sha256();stream=bundle.extractfile(member)
                for block in iter(lambda:stream.read(4*2**20),b''):digest.update(block)
                if digest.hexdigest()!=expected[relative]:raise ValueError('Full decoded native archive bytes differ')
                members[relative]=digest.hexdigest()
        if set(members)!=set(bucket):raise ValueError('Complete native archive membership required')
        records.append(dict(file=target.name,bytes=target.stat().st_size,sha256=sha(target),members_sha256=members))
        write(dest/'archive-progress.json',dict(state='verified_complete_archive_parts',parts=records,observed=stamp()))
    write(dest/'index.json',dict(state='complete_all_original_native_outputs_and_all_epoch_pairs_lossless_return',
          job=name,source_verification_sha256=sha(source/'verification.json'),plan_sha256=sha(PLAN),parts=records,
          all_original_files_retained=True,raw_dataset_uploaded=False,qualifying_labels_read=False,observed=stamp()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', nargs='?', choices=['prepare', 'verify-inputs', 'run-job', 'export', 'archive'])
    p.add_argument('--tree', type=Path, required=True)
    p.add_argument('--source-tree', type=Path)
    p.add_argument('--data-root', type=Path)
    p.add_argument('--job', choices=[j['name'] for j in read(PLAN)['jobs']])
    p.add_argument('--deadline-utc')
    p.add_argument('--output', type=Path)
    p.add_argument('--watch', type=int)
    p.add_argument('--bootstrap', type=int)
    a = p.parse_args()
    tree = a.tree.expanduser().resolve()
    if a.bootstrap:
        if not a.job:p.error('bootstrap requires job')
        bootstrap(tree,a.job,a.bootstrap);return
    if a.watch:
        if not a.job:p.error('watch requires job')
        watchdog(tree,a.job,a.watch);return
    if not a.action:p.error('action required')
    tree.parent.mkdir(parents=True, exist_ok=True)
    operation_lock = (tree.parent / 'operation.lock').open('a')
    fcntl.flock(operation_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if a.action == 'prepare':
        if not a.source_tree or not a.data_root:
            p.error('prepare requires --source-tree and --data-root')
        tree.parent.mkdir(parents=True, exist_ok=True)
        prepare(a.source_tree.expanduser().resolve(), a.data_root.expanduser().resolve(), tree)
    elif a.action == 'verify-inputs':
        verify_inputs(tree)
    elif a.action == 'run-job':
        if not a.job or not a.deadline_utc:
            p.error('run-job requires --job and --deadline-utc')
        deadline = dt.datetime.fromisoformat(a.deadline_utc.replace('Z', '+00:00'))
        if deadline.utcoffset() != dt.timedelta(0):
            p.error('deadline must explicitly use UTC')
        run_job(tree, a.job, deadline)
    else:
        if not a.job or not a.output:
            p.error('export requires --job and --output')
        action=export if a.action=='export' else archive
        action(tree, a.job, a.output.expanduser().resolve())
    print(a.action + ': complete', flush=True)


if __name__ == '__main__':
    main()
