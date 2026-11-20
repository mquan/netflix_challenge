#!/usr/bin/env python3
"""Full original split3 inputs and user-disjoint matched training populations."""
import os
for key in ['OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS']:os.environ[key]='2'
import argparse
import hashlib
from multiprocessing import shared_memory, resource_tracker
from pathlib import Path
import numpy as np
import upstream_repro as u
import upstream_exchange as exchange
import upstream_snapshot_blend as blend
import upstream_split3_pilot as original

RUN=u.ROOT/'artifacts/upstream-split3-matched-data'
REPORT=u.ROOT/'experiments/upstream-split3-matched-data'
TREE=u.ROOT/'artifacts/upstream-split3-next3/source/data'
BANK=u.ROOT/'experiments/upstream-split3-full-raw-bank/manifest.json'
VOTES=u.ROOT/'experiments/upstream-split3-voting-stream/verification.json'
PARENT=u.ROOT/'artifacts/upstream-item154-mps-context'
ROWS=1760494
RAW=73
FEATURES=209
DESIGN=149

def digest(a):return hashlib.sha256(memoryview(np.ascontiguousarray(a)).cast('B')).hexdigest()

def partition(users,mapping,canonical_users,canonical_groups):
    """Extend original user partitions; no row of a held user becomes eligible."""
    if not np.array_equal(users[mapping],canonical_users):raise ValueError('Canonical user projection differs')
    unique=np.unique(users)
    outer=exchange.user_folds(unique)
    inner=exchange.user_folds(unique.astype(np.uint64)*np.uint64(7)+np.uint64(19))==0
    lookup=np.full(int(unique.max())+1,-1,dtype=np.int8)
    lookup[unique]=outer*2+inner.astype(np.int8)
    order=np.argsort(canonical_users,kind='stable');cu=canonical_users[order];cg=canonical_groups[order]
    if np.any((cu[1:]==cu[:-1])&(cg[1:]!=cg[:-1])):raise ValueError('Original partitions split a user')
    lookup[canonical_users]=canonical_groups
    groups=lookup[users]
    if not np.array_equal(groups[mapping],canonical_groups):raise ValueError('Original canonical folds must be preserved')
    canonical=np.zeros(len(users),dtype=bool);canonical[mapping]=True
    for fold in range(5):
        held_users=np.unique(canonical_users[canonical_groups//2==fold])
        if np.any(np.isin(users[groups//2!=fold],held_users)):raise ValueError('Held user leaked through additional labels')
    return groups,canonical

def masks(groups,canonical,arm,fold):
    if arm not in ['control','expanded'] or fold not in range(5):raise ValueError('Frozen arm/fold required')
    train=(groups//2!=fold)&(canonical if arm=='control' else True)
    return train,train&(groups%2==0),train&(groups%2==1)

def blocks(raw,context,voting):
    raw=np.clip(np.asarray(raw,dtype=np.float32),0,6)
    if raw.shape[1]!=RAW or context.shape!=(len(raw),5) or voting.shape!=(len(raw),130):raise ValueError('Full fixed feature schema required')
    first=raw[:,:62];mean=np.zeros((len(raw),1),dtype=np.float32)
    for j in range(62):mean[:,0]+=first[:,j]
    mean/=np.float32(62)
    centered=first.copy();centered-=mean
    z=blend.design(first,np.ones(62,dtype=bool),context,tuple(range(14)))
    z=np.column_stack([z[:,:-1],raw[:,62:],z[:,-1:]])
    f=np.column_stack([mean-3.6,centered,context,raw[:,62:]-3.6,voting]).astype(np.float32)
    if z.shape[1]!=DESIGN or f.shape[1]!=FEATURES:raise ValueError('Generalized original dimensions differ')
    return z,f

class Inputs:
    def __init__(self):
        proof=u.read(VOTES)
        if (proof['state']!='complete_original_full_two_phase_voting_features_in_ram' or
            not proof['all_existing_own_qualifying_payloads_bitwise_equal'] or proof['qualifying_labels_read']):
            raise ValueError('Both complete original feature phases and exact own qualifying parity required')
        status=__import__('subprocess').run(['ps','-p',str(proof['shared_memory']['owner_pid']),'-o','stat='],capture_output=True,text=True).stdout.strip()
        if not status or status.startswith('Z'):raise ValueError('Verified RAM owner must remain live')
        self.segment=shared_memory.SharedMemory(name=proof['shared_memory']['name'])
        resource_tracker.unregister(self.segment._name,'shared_memory')
        self.voting=np.ndarray((130,ROWS),dtype='<f4',buffer=self.segment.buf)
        if digest(self.voting)!=proof['probe3_feature_major_payload_sha256']:raise ValueError('Complete native RAM feature payload changed')
        bank=u.read(BANK)
        if bank['raw_columns']!=RAW:raise ValueError('Frozen complete raw bank width differs')
        self.raw=[np.load(u.ROOT/r['probe3'],mmap_mode='r') for r in bank['columns']]
        self.context=np.load(RUN/'context.npy',mmap_mode='r')
    def block(self,lo,hi):return blocks(np.column_stack([a[lo:hi] for a in self.raw]),self.context[lo:hi],self.voting[:,lo:hi].T)
    def close(self):
        del self.voting
        self.segment.close()

def prepare():
    if RUN.exists() or REPORT.exists():raise ValueError('Existing matched input preparation preserved')
    sources={}
    for p,state in [(BANK,'complete_full_probe3_own_raw_bank_all_native_pairs_verified'),
        (u.ROOT/'experiments/upstream-split3-full-population-v2/verification.json','complete_original_known_origin_and_all_split3_training_disjointness')]:
        proof=u.read(p)
        if proof['state']!=state or proof['qualifying_labels_read']:raise ValueError('Complete original source gates required')
        sources.update(proof['source_sha256']);sources[str(p.relative_to(u.ROOT))]=u.sha(p)
    probe=original.arrays(TREE/'probe_3',labels=True);known=original.arrays(u.FRESH/'data/probex',labels=True)
    mapping=original.projection(probe,known)
    canonical_users=np.load(PARENT/'users.npy');canonical_groups=np.load(PARENT/'groups.npy')
    if not np.array_equal(known[3],np.load(PARENT/'labels.npy')):raise ValueError('Exact original canonical labels required')
    groups,canonical=partition(probe[0],mapping,canonical_users,canonical_groups)
    train=original.arrays(TREE/'train_3')
    context=blend.context_features(*train,*probe[:3])
    paths=[Path(__file__),Path(exchange.__file__),Path(blend.__file__),Path(original.__file__),u.ROOT/'SPLIT3_FULL_HOLDOUT_PLAN.md']
    paths.extend(PARENT/(name+'.npy') for name in ['labels','users','groups'])
    paths.extend(TREE/stage/(name+'.npy') for stage in ['train_3','probe_3'] for name in ['user_idxs','item_idxs','dates'])
    paths.append(TREE/'probe_3/ratings.npy')
    sources.update({str(p.relative_to(u.ROOT)):u.sha(p) for p in paths})
    cache={}
    for key,h in sources.items():
        p=u.ROOT/key;s=p.stat();identity=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
        if identity not in cache:cache[identity]=u.sha(p)
        if cache[identity]!=h:raise ValueError('Original source changed: '+key)
    RUN.mkdir();REPORT.mkdir()
    for name,a in [('labels',probe[3]),('users',probe[0]),('groups',groups),('canonical',canonical),('mapping',mapping),('context',context)]:np.save(RUN/(name+'.npy'),a)
    counts=[]
    for arm in ['control','expanded']:
        for fold in range(5):
            training,fit,valid=masks(groups,canonical,arm,fold)
            counts.append(dict(arm=arm,fold=fold,training=int(training.sum()),inner_fit=int(fit.sum()),inner_valid=int(valid.sum()),canonical_held=int((canonical&(groups//2==fold)).sum())))
    u.write(REPORT/'manifest.json',dict(state='complete_original_full_holdout_metadata_and_user_disjoint_matched_populations',
        source_sha256=sources,prepared_sha256={p.name:u.sha(p) for p in RUN.iterdir()},rows=ROWS,canonical_rows=len(mapping),
        raw_columns=RAW,feature_columns=FEATURES,design_columns=DESIGN,counts=counts,
        all_canonical_user_groups_preserved=True,all_additional_rows_of_held_users_excluded=True,
        qualifying_labels_read=False,models_fitted=0))
    print('Full original holdout metadata prepared; all ten arm/fold user exclusions verified',flush=True)

def stats():
    manifest=u.read(REPORT/'manifest.json')
    if (REPORT/'statistics-verification.json').exists():raise ValueError('Existing statistics evidence preserved')
    for name,h in manifest['prepared_sha256'].items():
        if u.sha(RUN/name)!=h:raise ValueError('Prepared original metadata changed')
    inputs=Inputs();y=np.load(RUN/'labels.npy');groups=np.load(RUN/'groups.npy');canonical=np.load(RUN/'canonical.npy')
    grams=np.zeros((2,10,DESIGN,DESIGN));rhs=np.zeros((2,10,DESIGN));hashes=[hashlib.sha256(),hashlib.sha256()]
    try:
        for lo in range(0,ROWS,8192):
            hi=min(lo+8192,ROWS);z,f=inputs.block(lo,hi)
            hashes[0].update(z.tobytes(order='C'));hashes[1].update(f.tobytes(order='C'))
            if not np.isfinite(z).all() or not np.isfinite(f).all():raise ValueError('Nonfinite full original inputs')
            for arm in range(2):
                for g in range(10):
                    keep=(groups[lo:hi]==g)&(canonical[lo:hi] if arm==0 else True);a=z[keep]
                    grams[arm,g]+=a.T@a;rhs[arm,g]+=a.T@y[lo:hi][keep]
        np.savez(RUN/'stats.npz',gram=grams,rhs=rhs)
        u.write(REPORT/'statistics-verification.json',dict(state='complete_all_original_full_input_blocks_and_both_arm_group_statistics',
            manifest_sha256=u.sha(REPORT/'manifest.json'),voting_verification_sha256=u.sha(VOTES),stats_sha256=u.sha(RUN/'stats.npz'),
            design_payload_sha256=hashes[0].hexdigest(),feature_payload_sha256=hashes[1].hexdigest(),
            rows=ROWS,qualifying_labels_read=False,models_fitted=0))
    finally:inputs.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stats',action='store_true');args=parser.parse_args()
    stats() if args.stats else prepare()
