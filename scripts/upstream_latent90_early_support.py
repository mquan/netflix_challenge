#!/usr/bin/env python3
"""Extract exact source support columns after each original first phase finishes."""
import datetime as dt
import fcntl
from pathlib import Path
import re
import subprocess
import sys
import time
import numpy as np
import upstream_latent90_local as native

u=native.u
REPORT=u.ROOT/'experiments/upstream-latent90-early-support'
SPECS={'rbm50lat':[('ufeat','u',50)],'tx20lat':[('ufeat','u',20),('ifeat','i',20)]}
OUT=native.TREE/'preds_new/lat-early'

def stamp():return dt.datetime.now(dt.timezone.utc).isoformat()

def owner(pid):
    return subprocess.run(['ps','-p',str(pid),'-o','lstart=,command='],capture_output=True,text=True).stdout.strip()

def first_phase_prefix(text):
    marker='fulltrain => qual\n'
    if marker not in text:return None
    prefix=text.split(marker,1)[0]
    epochs=[int(v) for v in re.findall(r'^Epoch\s+(\d+)(?:\s|$)',prefix,re.M)]
    if epochs!=list(range(1,21)) or 'trainx => probex\n' not in prefix:
        raise ValueError('Original complete20-epoch first phase required')
    return prefix+marker

def prepare(pid):
    native.prepare()
    parent=native.REPORT/'plan.json'
    bindings={str(p.relative_to(u.ROOT)):u.sha(p) for p in [Path(__file__),parent,Path(native.__file__)]}
    identity=owner(pid)
    if not identity or 'upstream_latent90_local.py' not in identity:raise ValueError('Actual original native controller required')
    path=REPORT/'plan.json'
    if path.exists():
        plan=u.read(path)
        if plan['bindings']!=bindings or plan['native_owner']!=dict(pid=pid,identity=identity):raise ValueError('Frozen early-support plan changed')
        return plan
    REPORT.mkdir(parents=True)
    plan=dict(created=stamp(),bindings=bindings,native_owner=dict(pid=pid,identity=identity),
        source_pin=u.PIN,first_phase_epochs=20,feature_columns=90,
        rule='Unchanged original predfeat maps trainx factors onto BOTH query sets. Require complete original20-epoch first-phase log, closed factor files and full source/data/binary bindings. Extract to a separate lat-early directory while original two-phase fitting continues. No model capacity/epochs reduced.',
        scope='Complete paired support features can precede whole two-phase model completion. Do not treat them as completed native raw prediction pairs.',
        qualifying_labels_read=False,public_prediction_values_read=False)
    u.write(path,plan);return plan

def extract(name,prefix):
    receiptpath=REPORT/(name+'-receipt.json')
    if receiptpath.exists():
        receipt=u.read(receiptpath)
        if receipt['plan_sha256']!=u.sha(REPORT/'plan.json'):raise ValueError('Support plan changed')
        for rel,digest in {**receipt['factor_sha256'],**receipt['support_sha256']}.items():
            if u.sha(u.ROOT/rel)!=digest:raise ValueError('Completed support source/output changed')
        return receipt
    if OUT.exists() and list(OUT.glob(name+'.*.npy')):raise ValueError('Partial early extraction preserved')
    native.prepare()  # Recheck the unchanged full source, datasets and executables.
    if np.any(np.load(native.TREE/'data/qual/ratings.npy',mmap_mode='r')):raise ValueError('Nonzero qualifying outcomes')
    factors={};matrices={}
    for kind,tag,width in SPECS[name]:
        path=native.TREE/f'preds_new/{name}.{kind}.trainx.npy';a=np.load(path,mmap_mode='r')
        if a.shape!=(480189 if tag=='u' else 17770,width) or a.dtype!=np.float32 or not np.isfinite(a).all():raise ValueError('Invalid completed first-phase factors')
        factors[str(path.relative_to(u.ROOT))]=u.sha(path);matrices[tag]=a
    evidence=REPORT/(name+'-first-phase.log');evidence.write_text(prefix)
    command=['./target/release/predfeat','-p','pipeline-new.toml','--latent-users',name]
    if name=='tx20lat':command+=['--latent-items',name]
    command+=['--out','lat-early']
    with (REPORT/(name+'-predfeat.log')).open('x') as log:
        subprocess.run(command,cwd=native.TREE,env=native.model_environment({}),stdout=log,stderr=subprocess.STDOUT,check=True)
    outputs={};counts={}
    # Independent all-row gather replay: do not call the Rust extraction helper.
    for split,rows in [('probex',1128164),('qual',2817131)]:
        for kind,tag,width in SPECS[name]:
            idx=np.load(native.TREE/'data'/split/('user_idxs.npy' if tag=='u' else 'item_idxs.npy'),mmap_mode='r')
            if idx.shape!=(rows,):raise ValueError('Query row identity shape mismatch')
            for column in range(width):
                path=OUT/f'{name}.{tag}{column:02}.{split}.npy';values=np.load(path,mmap_mode='r')
                if values.shape!=(rows,) or values.dtype!=np.float32 or not np.isfinite(values).all():raise ValueError('Malformed support vector')
                for lo in range(0,rows,65536):
                    hi=min(lo+65536,rows);expected=matrices[tag][idx[lo:hi],column]
                    if not np.array_equal(values[lo:hi].view(np.uint32),expected.view(np.uint32)):
                        raise ValueError('Independent factor-row gather differs')
                outputs[str(path.relative_to(u.ROOT))]=u.sha(path)
        counts[split]=rows
    for rel,digest in factors.items():
        if u.sha(u.ROOT/rel)!=digest:raise ValueError('First-phase factor changed during extraction')
    result=dict(state='complete_original_first_phase_paired_supports',observed=stamp(),model=name,
        plan_sha256=u.sha(REPORT/'plan.json'),factor_sha256=factors,support_sha256=outputs,
        first_phase_log_sha256=u.sha(evidence),command=command,counts=counts,
        first_phase_epochs_complete=20,support_columns=sum(v[2] for v in SPECS[name]),
        all_support_rows_independently_replayed_bitwise=True,
        native_two_phase_completion_claimed=False,qualifying_labels_read=False,
        scope='Exact original-source support features from a completed first-phase model. Both query vectors complete and independently gathered; whole native two-phase training continues separately.')
    u.write(receiptpath,result);return result

def run(pid):
    plan=prepare(pid)
    with (REPORT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        complete={}
        while len(complete)<len(SPECS):
            for name in SPECS:
                if name in complete:continue
                log=native.REPORT/(name+'.log')
                prefix=first_phase_prefix(log.read_text()) if log.exists() else None
                if prefix is not None:
                    complete[name]=extract(name,prefix)
            if len(complete)==len(SPECS):break
            if owner(pid)!=plan['native_owner']['identity']:
                raise RuntimeError('Original native controller ended before all first-phase supports; outputs preserved, no restart')
            u.write(REPORT/'status.json',dict(state='waiting_for_original_first_phases',observed=stamp(),
                verified_models=list(complete),actual_native_controller_pid=pid))
            time.sleep(30)
        result=dict(state='complete_original90_paired_supports',observed=stamp(),
            plan_sha256=u.sha(REPORT/'plan.json'),models={name:u.sha(REPORT/(name+'-receipt.json')) for name in SPECS},
            feature_columns=90,query_vectors=180,all_rows_independently_replayed_bitwise=True,
            native_two_phase_completion_claimed=False,qualifying_labels_read=False)
        u.write(REPORT/'completion.json',result);print(result,flush=True)

if __name__=='__main__':run(int(sys.argv[1]))
