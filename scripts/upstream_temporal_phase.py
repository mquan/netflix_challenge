#!/usr/bin/env python3
"""Pair our completed remote ProbeX phase with an independent native qual phase."""
import argparse
import datetime as dt
import fcntl
import hashlib
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import numpy as np
import upstream_repro as u
import upstream_exchange as x
import upstream_mapped_prepare as mapped
import upstream_mapped_job as original
from upstream_full_reproduce import group_rss, stop

JOB='tsvdx5-150o'
NAMES=[JOB, JOB+'_ep06', JOB+'_ep16', JOB+'_ep24']
BASE=mapped.BASE/'independent-qual'
REPORT=u.ROOT/'experiments/upstream-temporal-phase'
OPERATIONAL=original.REPORT/(JOB+'.status.json')
SOURCE='src/bin/tsvdx5-new.rs'
BINARY='tsvdx5-new-qual'
PRED='preds_independent_qual'


def driver(source):
    match=re.search(r'"tsvdx5-150o"\s*=>\s*(run_ord\([^;\n]+?\)),',source)
    if not match:raise ValueError('Missing original ordinal dispatch')
    body=source[source.index('fn run_ord('):source.index('\nfn run_nlpp(')]
    old='''    fit2!(
        TxModel, cfg, "rtg", job_name, SPLIT_NEW,
        save_train: true, save_subscores: true, save_probe_each_epoch: true
    );'''
    new='''    if std::env::args().any(|v| v == "--describe") { println!("{:?}",cfg); return; }
    netflix_prize::fit::<TxModel>(cfg,"rtg","fulltrain","qual",job_name,
        true,true,true,"preds_independent_qual",false,false);'''
    if body.count(old)!=1:raise ValueError('Original ordinal fit invocation changed')
    return ('use netflix_prize::{OrdinalHeadConfig,tx::{TxModel,TxConfig}};\n'
        +'fn main(){ let job_name="tsvdx5-150o"; '+match.group(1)+'; }\n'
        +body.replace(old,new))


def remote_columns():
    mp=u.ROOT/'experiments/upstream-remote-12h'/JOB/'transfer-manifest.json'
    manifest=u.read(mp);rp=u.ROOT/manifest['native_receipt'];receipt=u.read(rp)
    jobs=u.read(u.BASE/'registry.json')['jobs']
    if (manifest['source_commit']!=u.PIN or receipt['source_commit']!=u.PIN or
        manifest['job']!=JOB or receipt['job']!=JOB or
        manifest['native_receipt_sha256']!=u.sha(rp) or receipt['command']!=jobs[JOB]['command'] or
        manifest['native_command']!=receipt['command'] or not manifest['fresh_data_receipt']['qualifying_outcomes_zero']):
        raise ValueError('Invalid OUR fresh remote temporal receipt')
    for filename,digest in manifest['alignment_sha256'].items():
        if u.sha(u.FRESH/'data/probex'/filename)!=digest:raise ValueError('Remote phase row identities differ')
    values={};sources={str(mp.relative_to(u.ROOT)):u.sha(mp),str(rp.relative_to(u.ROOT)):u.sha(rp)}
    for name in NAMES:
        filename='preds_new/'+name+'.probex.npy'
        entries=[v for v in manifest['outputs'] if v['source']==filename]
        if len(entries)!=1:raise ValueError('Missing remote phase column '+name)
        entry=entries[0];path=u.ROOT/entry['npz']
        if (u.sha(path)!=entry['npz_sha256'] or entry['source_npy_sha256']!=receipt['outputs'].get(filename) or
            entry['receipt_bound_sha256']!=entry['source_npy_sha256'] or entry['owning_native_receipt_sha256']!=u.sha(rp)):
            raise ValueError('Remote prediction provenance differs')
        with np.load(path,allow_pickle=False) as archive:
            if archive.files!=['prediction']:raise ValueError('Unexpected remote array fields')
            value=archive['prediction']
        if value.shape!=(1128164,) or value.dtype!=np.dtype('float32') or not np.isfinite(value).all():
            raise ValueError('Invalid fresh remote prediction')
        if x.array_sha(value)!=entry['prediction_content_sha256']:raise ValueError('Remote content changed')
        sources[str(path.relative_to(u.ROOT))]=u.sha(path)
        values[name]=(value,entry['source_npy_sha256'])
    return values,sources


def prepare():
    proof,storage=original.verify_candidate();original.layout()
    BASE.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    source=mapped.pinned(SOURCE)
    target=mapped.TREE/'src/bin'/(BINARY+'.rs');text=driver(source)
    if target.exists() and target.read_text()!=text:raise ValueError('Phase driver changed')
    target.write_text(text)
    values,sources=remote_columns()
    pair=BASE/'paired';pair.mkdir(exist_ok=True)
    remote_storage={}
    for name,(value,digest) in values.items():
        path=pair/(name+'.probex.npy')
        if not path.exists():np.save(path,value)
        stored=np.load(path,allow_pickle=False)
        if stored.dtype!=value.dtype or stored.shape!=value.shape or not np.array_equal(stored.view(np.uint32),value.view(np.uint32)):
            raise ValueError('Re-encoded remote prediction values differ')
        remote_storage[name]=dict(original_native_npy_sha256=digest,reencoded_npy_sha256=u.sha(path),
                                 prediction_content_sha256=x.array_sha(value))
        sources[str(path.relative_to(u.ROOT))]=u.sha(path)
    env=u.environment();env.update(CARGO_TARGET_DIR=str(mapped.BASE/'target'),CARGO_NET_OFFLINE='true',
        TX_MMAP_DIR=str(BASE/'scratch'),TX_LAZY_DAY='1',RAYON_NUM_THREADS='2')
    (BASE/'scratch').mkdir(exist_ok=True)
    with (REPORT/'build.log').open('w') as log:
        subprocess.run([str(u.BASE/'toolchain/cargo/bin/cargo'),'build','--release','--offline','--bin',BINARY],
            cwd=mapped.TREE,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    binary=mapped.BASE/'target/release'/BINARY
    description=subprocess.check_output([str(binary),'--describe'],cwd=mapped.TREE,env=env,text=True).strip()
    oldlog=original.REPORT/(JOB+'.log')
    original_config=next(line for line in oldlog.read_text().splitlines() if line.startswith('TxConfig {'))
    if description!=original_config:raise ValueError('Independent phase config differs from live original recipe')
    original.verify_candidate()
    sources.update({str((u.FRESH/'data'/split/(key+'.npy')).relative_to(u.ROOT)):u.sha(u.FRESH/'data'/split/(key+'.npy'))
        for split in ['fulltrain','qual'] for key in ['user_idxs','item_idxs','dates','ratings','user_cnts','item_cnts','item_years','is_test']})
    plan=dict(source_pin=u.PIN,job=JOB,names=NAMES,canonical_reference_command=u.read(u.BASE/'registry.json')['jobs'][JOB]['command'],
        execution_command=[str(binary)],driver_sha256=u.sha(target),original_dispatcher_sha256=hashlib.sha256(source.encode()).hexdigest(),
        description=description,epochs=25,storage=storage,parity_sha256=u.sha(proof),binary_sha256=u.sha(binary),
        source_sha256=sources,remote_storage=remote_storage,script_sha256=u.sha(Path(__file__)),qualifying_outcomes_zero=True,
        provenance='ProbeX columns are OUR complete fresh remote source phase. The new fulltrain/qual phase calls exactly the pinned fit function used by fit2 phase2, with identical source-extracted config and original 25 epochs. Source math/storage parity proof disclosed. Not a second complete original pipeline job.')
    path=REPORT/'plan.json'
    if path.exists() and u.read(path)!=plan:raise ValueError('Frozen independent-phase plan changed')
    u.write(path,plan);return plan,env


def verify_receipt(verify_files=True):
    receipt=u.read(REPORT/'paired-receipt.json');plan=u.read(REPORT/'plan.json')
    if (receipt['state']!='complete' or receipt['plan_sha256']!=u.sha(REPORT/'plan.json') or
        receipt['job']!=JOB or receipt['source_commit']!=u.PIN or not receipt['qualifying_outcomes_zero'] or
        receipt['epochs_qual']!=25 or plan['script_sha256']!=u.sha(Path(__file__))):
        raise ValueError('Independent-phase completion binding differs')
    expected={name:dict(probex=str((BASE/'paired'/(name+'.probex.npy')).relative_to(u.ROOT)),
                        qual=str((mapped.TREE/PRED/(name+'.qual.npy')).relative_to(u.ROOT))) for name in NAMES}
    if receipt['pairs']!=expected or set(receipt['outputs'])!={path for pair in expected.values() for path in pair.values()}:
        raise ValueError('Independent-phase output identity differs')
    if verify_files:
        original.verify_candidate()
        if (u.sha(mapped.TREE/'src/bin'/(BINARY+'.rs'))!=plan['driver_sha256'] or
            u.sha(mapped.BASE/'target/release'/BINARY)!=plan['binary_sha256']):
            raise ValueError('Independent phase driver/binary changed')
        if plan['parity_sha256']!=u.sha(u.ROOT/'experiments/full-upstream-reproduction/mapped-parity/verification.json'):
            raise ValueError('Independent phase parity binding differs')
        for path,digest in {**plan['source_sha256'],**receipt['outputs']}.items():
            if u.sha(u.ROOT/path)!=digest:raise ValueError('Independent phase source/output changed: '+path)
    return receipt


def run():
    def interrupted(signum,frame):raise KeyboardInterrupt('Signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted)
    REPORT.mkdir(parents=True,exist_ok=True);BASE.mkdir(parents=True,exist_ok=True)
    with (BASE/'phase.lock').open('a') as ownlock:
        fcntl.flock(ownlock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan,env=prepare()
        if (REPORT/'paired-receipt.json').exists():verify_receipt();return
        # Acquired only after a separately reviewed handover stops the old phase.
        with (mapped.BASE/'training.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            out=mapped.TREE/PRED
            if out.exists() and any(out.iterdir()):raise ValueError('Unreceipted independent-phase outputs preserved')
            out.mkdir(exist_ok=True)
            disk=os.statvfs(BASE)
            if disk.f_bavail*disk.f_frsize<40*2**30:raise ValueError('Insufficient phase disk headroom')
            state=dict(state='training',job=JOB,phase='fulltrain-to-qual',independent_phase=True,
                started=dt.datetime.now(dt.timezone.utc).isoformat());child=None
            try:
                with (REPORT/'training.log').open('w') as log:
                    child=subprocess.Popen(plan['execution_command'],cwd=mapped.TREE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    high=0
                    while child.poll() is None:
                        own=group_rss(child.pid);free=os.statvfs(BASE);free=free.f_bavail*free.f_frsize
                        high=high+1 if own>20*2**30 else 0
                        state.update(pid=child.pid,rss_bytes=own,heartbeat=dt.datetime.now(dt.timezone.utc).isoformat(),disk_free_bytes=free)
                        u.write(REPORT/'status.json',state);u.write(OPERATIONAL,state)
                        if high>=3 or free<12*2**30:raise RuntimeError('Independent temporal resource guard; partials preserved')
                        time.sleep(30)
                if child.returncode:raise RuntimeError('Independent temporal phase failed')
                original.verify_candidate()
                outputs={};pairs={}
                for name in NAMES:
                    pair={}
                    for split,rows in [('probex',1128164),('qual',2817131)]:
                        path=BASE/'paired'/(name+'.'+split+'.npy') if split=='probex' else out/(name+'.'+split+'.npy')
                        a=np.load(path,mmap_mode='r',allow_pickle=False)
                        if a.shape!=(rows,) or a.dtype!=np.dtype('float32') or not np.isfinite(a).all():raise ValueError('Invalid paired phase output')
                        relative=str(path.relative_to(u.ROOT));outputs[relative]=u.sha(path);pair[split]=relative
                    pairs[name]=pair
                # The final epoch file plus successful exit proves all 25 updates ran.
                if not (out/(JOB+'_ep25.qual.npy')).exists():raise ValueError('Missing canonical final epoch')
                terminal=np.load(out/(JOB+'.qual.npy'));last=np.load(out/(JOB+'_ep25.qual.npy'))
                if not np.array_equal(terminal.view(np.uint32),last.view(np.uint32)):
                    raise ValueError('Final canonical temporal snapshot differs')
                record=dict(state='complete',source_commit=u.PIN,job=JOB,epochs_qual=25,
                    plan_sha256=u.sha(REPORT/'plan.json'),pairs=pairs,outputs=outputs,
                    qualifying_outcomes_zero=True,finished=dt.datetime.now(dt.timezone.utc).isoformat(),provenance=plan['provenance'])
                u.write(REPORT/'paired-receipt.json',record);verify_receipt()
                state.update(state='complete',paired_receipt_sha256=u.sha(REPORT/'paired-receipt.json'))
                u.write(REPORT/'status.json',state);u.write(OPERATIONAL,state)
            except BaseException as error:
                if child is not None:stop(child)
                state.update(state='incomplete_preserved',error=repr(error));u.write(REPORT/'status.json',state);u.write(OPERATIONAL,state);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true');args=parser.parse_args()
    if args.run:run()
    else:prepare()
