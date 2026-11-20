#!/usr/bin/env python3
"""Complete own CF95/TX50n additions with a fixed matched391 ridge control."""
import os
for key in ['OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS']:
    os.environ[key] = '2'
import argparse
import hashlib
from pathlib import Path
import shutil
import numpy as np
import upstream_own391_paired_data as parent
import upstream_cf96_complete_bank as cf
import upstream_research_footprints as resources
import upstream_cf96_memory_resume as memory

u = parent.u
REPORT = u.ROOT/'experiments/upstream-complete-cf-delta'
RUN = u.ROOT/'artifacts/upstream-complete-cf-delta'
DECL = u.ROOT/'OWN_COMPLETE_CF_DELTA_PLAN.md'
RAW = list(range(62)) + list(range(137,466))
COUNTS = dict(probex=1128164,qual=2817131)


class Guard:
    def __init__(self):
        self.baseline = memory.memory()
        self.check(reserve=3*2**30)

    def check(self,reserve=0):
        workers = resources.footprints(os.getpid())
        current = memory.memory()
        own = workers[str(os.getpid())]
        if (max(own['bytes'],own['rss_bytes']) > 3*2**30 or
            sum(w['bytes'] for w in workers.values())+reserve >= 24*2**30 or
            sum(w['rss_bytes'] for w in workers.values())+reserve >= 24*2**30 or
            current['compressor_bytes'] >= 5*2**30 or
            current['swap_used_bytes']-self.baseline['swap_used_bytes'] > 128*2**20 or
            shutil.disk_usage(u.ROOT).free < 18.75*2**30):
            raise RuntimeError('Standalone delta resource guard; all other jobs preserved')


def sources():
    pm = parent.REPORT/'manifest.json'
    pv = parent.REPORT/'independent-verification.json'
    m,v = u.read(pm),u.read(pv)
    cp = cf.REPORT/'manifest.json'
    cv = cf.REPORT/'independent-verification.json'
    c,proof = u.read(cp),u.read(cv)
    tp = u.ROOT/'experiments/upstream-remote15h-ext-return/s3-tx50n-v1/local-verification.json'
    tx = u.read(tp)
    if (v['state'] != 'complete_independent_matched_B391_all_operators_statistics_ridge_and_stored_predictions' or
        v['manifest_sha256'] != u.sha(pm) or v['source_sha256'] != m['source_sha256'] or
        not v['matched_B_known_and_qualifying'] or v['qualifying_labels_read'] is not False or
        proof['state'] != 'complete_independent_all98_own_cf96_pairs_and_exact_aliases' or
        proof['manifest_sha256'] != u.sha(cp) or proof['source_sha256'] != c['source_sha256'] or
        proof['distinct_pairs'] != 96 or proof['qualifying_labels_read'] is not False or
        tx['state'] != 'complete_local_compact_remote15h_extension_return_check' or
        tx['job'] != 's3-tx50n' or tx['source_pin'] != u.PIN or
        tx['original_full_epochs_each'] != 22 or len(tx['rows']) != 1 or
        tx['qualifying_labels_read'] is not False or tx['public_prediction_values_read'] is not False or
        not tx['original_recipe_trace_and_immutable_export_verified'] or
        not tx['local_full_known_map_and_training_disjointness_recomputed'] or
        not tx['native_qualifying_npy_bytes_reconstructed_exactly']):
        raise ValueError('Complete matched391, original95/95 and original22/22 proofs required')
    bindings = dict(m['source_sha256'])
    def bind(relative,digest):
        if relative in bindings and bindings[relative] != digest:
            raise ValueError('Conflicting original source expectation: '+relative)
        bindings[relative] = digest
    for closure in [proof['source_sha256'],tx['bindings']]:
        for k,d in closure.items():bind(k,d)
    for name,d in m['prepared_sha256'].items():bind(str((parent.RUN/name).relative_to(u.ROOT)),d)
    for p in [pm,pv,cp,cv,tp,DECL,Path(__file__),u.ROOT/'scripts/verify_upstream_complete_cf_delta.py',
              Path(resources.__file__),Path(memory.__file__)]:
        bind(str(p.relative_to(u.ROOT)),u.sha(p))
    added = [dict(id='complete-cf:'+r['name'],paired_files=r['paired_files']) for r in c['rows']]
    added += [dict(id='extension:'+r['name'],paired_files=r['paired_files']) for r in tx['rows']]
    if len(added) != 97 or len(RAW) != 391:
        raise ValueError('All97 distinct new source entries and391 existing scalar columns required')
    parent.check(bindings)
    return m,added,bindings


def vectors(row,query,lo,hi):
    values = np.load(u.ROOT/row['paired_files'][query],mmap_mode='r',allow_pickle=False)
    if values.shape != (COUNTS[query],) or values.dtype != np.dtype('<f4'):
        raise ValueError('Complete original scalar pair shape/dtype required')
    return np.clip(values[lo:hi],0,6).astype('<f4')


def reader(query,m):
    return parent.Inputs(query,m['original_rows'],m['temporal_rows'],m['new_rows'])


def admit():
    if REPORT.exists() or RUN.exists():raise ValueError('Existing delta study retained')
    guard = Guard()
    m,added,bindings = sources()
    REPORT.mkdir(); RUN.mkdir()
    u.write(REPORT/'plan.json',dict(source_sha256=bindings,candidates=added,
        parent_design_columns=467,old_scalar_columns=RAW,qualifying_labels_read=False,
        lambda_fixed=1000,groups=10,folds=5,clip=[1.05,4.95]))
    hashes = {}
    for query,count in COUNTS.items():
        ds = [hashlib.sha256() for _ in range(488)]
        operators = [hashlib.sha256(),hashlib.sha256()]
        source = reader(query,m)
        try:
            for block,lo in enumerate(range(0,count,8192)):
                hi = min(lo+8192,count)
                z,f = source.block(lo,hi)
                operators[0].update(z.tobytes());operators[1].update(f.tobytes())
                for i,column in enumerate(RAW):
                    ds[i].update(np.clip(z[:,column],0,6).astype('<f4').tobytes())
                for i,row in enumerate(added,391):
                    values=vectors(row,query,lo,hi)
                    if not np.isfinite(values).all():raise ValueError('Finite complete scalar required')
                    ds[i].update(values.tobytes())
                if block%20 == 0:guard.check()
            if [d.hexdigest() for d in operators] != m['virtual_c_order_sha256'][query]:
                raise ValueError('Every complete original matched391 operator bit required')
        finally:source.close()
        hashes[query] = [d.hexdigest() for d in ds]
        print('Complete scalar novelty stream:',query,flush=True)
    seen={}
    for i in range(391):seen.setdefault(tuple(hashes[q][i] for q in COUNTS),'original391:%03d'%i)
    retained,aliases = [],[]
    for i,row in enumerate(added,391):
        key=tuple(hashes[q][i] for q in COUNTS)
        if key in seen:
            aliases.append(dict(id=row['id'],alias_of=seen[key],clipped_pair_sha256=list(key)))
        else:
            seen[key]=row['id'];retained.append(row)
    parent.check(bindings);guard.check()
    u.write(REPORT/'input-manifest.json',dict(state='complete_all488_scalar_pairs_pending_independent_replay',
        source_sha256=bindings,plan_sha256=u.sha(REPORT/'plan.json'),clipped_sha256=hashes,
        retained=retained,aliases=aliases,raw_predictions=391+len(retained),design_columns=467+len(retained),
        qualifying_labels_read=False,models_fitted=0))
    print('Genuinely new pairs:',len(retained),'exact paired aliases:',len(aliases),flush=True)


def design(z,rows,query,lo,hi):
    extra=np.column_stack([vectors(row,query,lo,hi) for row in rows]) if rows else np.empty((hi-lo,0),dtype=np.float32)
    return np.column_stack([z[:,:-1],extra,z[:,-1:]])


def fit():
    proof=u.read(REPORT/'input-verification.json');im=u.read(REPORT/'input-manifest.json')
    if (proof['state']!='complete_independent_all488_scalar_pairs_and_novelty' or
        proof['input_manifest_sha256']!=u.sha(REPORT/'input-manifest.json')):
        raise ValueError('Full independent novelty required before fitting')
    if (REPORT/'fit-manifest.json').exists() or (RUN/'stats.npz').exists():raise ValueError('Existing fit preserved')
    guard=Guard();m,added,bindings=sources()
    if bindings!=im['source_sha256']:raise ValueError('Frozen source closure differs')
    for name in ['labels.npy','users.npy','groups.npy']:u.link_or_copy(parent.RUN/name,RUN/name)
    y=np.load(RUN/'labels.npy');groups=np.load(RUN/'groups.npy');d=im['design_columns']
    gram=np.zeros((10,d,d));rhs=np.zeros((10,d));source=reader('probex',m);stream=hashlib.sha256()
    try:
        for block,lo in enumerate(range(0,len(y),8192)):
            hi=min(lo+8192,len(y));z,_=source.block(lo,hi);a=design(z,im['retained'],'probex',lo,hi)
            if not np.isfinite(a).all():raise ValueError('Finite full design required')
            stream.update(a.tobytes())
            for group in range(10):
                keep=groups[lo:hi]==group;b=a[keep];gram[group]+=b.T@b;rhs[group]+=b.T@y[lo:hi][keep]
            if block%10==0:guard.check()
    finally:source.close()
    weights=np.stack([parent.n.ridge_weights(gram,rhs,np.arange(10)//2!=fold,1000) for fold in range(5)])
    pred=np.empty(len(y),dtype='<f4');source=reader('probex',m)
    try:
        for block,lo in enumerate(range(0,len(y),8192)):
            hi=min(lo+8192,len(y));z,_=source.block(lo,hi);a=design(z,im['retained'],'probex',lo,hi)
            for fold in range(5):
                held=groups[lo:hi]//2==fold;pred[lo:hi][held]=np.clip(a[held]@weights[fold],1.05,4.95).astype('<f4')
            if block%10==0:guard.check()
    finally:source.close()
    np.savez(RUN/'stats.npz',gram=gram,rhs=rhs);np.save(RUN/'weights.npy',weights);pred.tofile(RUN/'ridge.f32')
    parent.check(bindings);guard.check()
    u.write(REPORT/'fit-manifest.json',dict(state='complete_fixed_lambda_delta_ridge_pending_independent_replay',
        source_sha256=bindings,input_manifest_sha256=u.sha(REPORT/'input-manifest.json'),
        input_verification_sha256=u.sha(REPORT/'input-verification.json'),known_design_sha256=stream.hexdigest(),
        raw_predictions=im['raw_predictions'],design_columns=d,ridge_rmse=parent.score(pred,y),
        parent391_rmse=m['ridge_rmse'],fold_rmse=[parent.score(pred[groups//2==f],y[groups//2==f]) for f in range(5)],
        prepared_sha256={p.name:u.sha(p) for p in RUN.iterdir() if p.is_file()},qualifying_labels_read=False))
    print('Fixed matched391 / expanded ridge:',m['ridge_rmse'],parent.score(pred,y),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['admit','fit']);args=parser.parse_args()
    admit() if args.action=='admit' else fit()
