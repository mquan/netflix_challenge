#!/usr/bin/env python3
"""Complete local native split3 snapshot bank, retaining full original query rows."""
import os
for key in ['OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS']:os.environ[key]='2'
from pathlib import Path
import hashlib
import numpy as np
import upstream_repro as u

REPORT=u.ROOT/'experiments/upstream-split3-full-raw-bank'
JOBS=[('upstream-split3-pilot','s3-attn16','completion-verification.json'),
    ('upstream-split3-diversity','s3-dnn16t','s3-dnn16t-verification.json'),
    ('upstream-split3-diversity','s3-dnn32nm','s3-dnn32nm-verification.json'),
    ('upstream-split3-diversity','s3-nade12s','s3-nade12s-verification.json'),
    ('upstream-split3-next3','s3-attn64','s3-attn64-verification.json'),
    ('upstream-split3-next3','s3-rbm200','s3-rbm200-verification.json'),
    ('upstream-split3-knn3-chains','s3-dnn32nm__knn3','s3-dnn32nm__knn3-verification.json'),
    ('upstream-split3-knn3-chains','s3-nade12s__knn3','s3-nade12s__knn3-verification.json')]

def main():
    if REPORT.exists():raise ValueError('Existing complete full-row bank evidence preserved')
    REPORT.mkdir();sources={};candidates=[]
    population=u.ROOT/'experiments/upstream-split3-full-population-v2/verification.json';gate=u.read(population)
    if gate['state']!='complete_original_known_origin_and_all_split3_training_disjointness':raise ValueError('Complete original training/population proof required')
    sources.update(gate['source_sha256']);sources[str(population.relative_to(u.ROOT))]=u.sha(population)
    reference=u.ROOT/'artifacts/upstream-split3-next3/source/data'
    def bind(path,expected=None):
        key=str(path.relative_to(u.ROOT));digest=u.sha(path) if expected is None else expected
        if key in sources and sources[key]!=digest:raise ValueError('Conflicting original source bindings')
        sources[key]=digest
    native_roots=set()
    u.write(REPORT/'plan.json',dict(jobs=JOBS,rule='Every locally complete original native pair and captured original snapshot, full1760494-row probe3 and2817131-row qualifying. Fixed producer order, terminal first then epoch names; remove only exact full-pair duplicates. No outcome-based feature subset or fitting. Compact remote returns excluded until native full probe vectors available.'))
    for namespace,job,filename in JOBS:
        path=u.ROOT/'experiments'/namespace/filename;r=u.read(path)
        if (r['qualifying_labels_read'] is not False or not r['qualifying_targets_zero'] or not r['full_known_projection_recomputed']):raise ValueError('Complete own original producer proof required')
        if namespace=='upstream-split3-pilot':
            if r['state']!='complete_original_two_phase_split3_pilot' or not r['all24_query_snapshots_verified']:raise ValueError('Original full attention16 pilot required')
            pp=path.parent/'plan.json'
            if u.read(pp)['source_pin']!=u.PIN:raise ValueError('Original source pin differs')
            bind(pp)
        elif '__knn3' in job:
            if r['state']!='complete_original_two_phase_knn3_chain' or not r['both_full_native_neighbor_builds_verified']:raise ValueError('Complete original KNN3 chain required')
        elif r['state']!='complete_original_two_phase_split3_model' or not r['both_full_epoch_traces']:raise ValueError('Both complete original native epoch traces required')
        for key,h in r['bindings'].items():bind(u.ROOT/key,h)
        bind(path)
        files=[u.ROOT/key for key in r['bindings'] if key.endswith('.probe_3.npy') and
            (Path(key).name==job+'.probe_3.npy' or Path(key).name.startswith(job+'_ep'))]
        files.sort(key=lambda p:(p.name!=job+'.probe_3.npy',p.name))
        if not files:raise ValueError('Full native probe3 vector absent')
        for probe in files:
            name=probe.name.removesuffix('.probe_3.npy');qual=probe.with_name(name+'.qual.npy')
            if str(qual.relative_to(u.ROOT)) not in r['bindings']:raise ValueError('Complete native qualifying counterpart missing')
            native_roots.add(probe.parent.parent)
            candidates.append(dict(name=name,probe3=str(probe.relative_to(u.ROOT)),qual=str(qual.relative_to(u.ROOT)),producer=job,proof=str(path.relative_to(u.ROOT))))
    for root in native_roots:
        for stage,count in [('probe_3',1760494),('qual',2817131)]:
            for name in ['user_idxs','item_idxs','dates']+(['ratings'] if stage=='probe_3' else []):
                a=root/'data'/stage/(name+'.npy');b=reference/stage/(name+'.npy')
                if not np.array_equal(np.load(a,mmap_mode='r'),np.load(b,mmap_mode='r')):raise ValueError('Every native query identity/label row must share original order')
                bind(a)
    cache={}
    for key,digest in sources.items():
        p=u.ROOT/key;s=p.stat();identity=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
        if identity not in cache:cache[identity]=u.sha(p)
        if cache[identity]!=digest:raise ValueError('Original native source/output changed: '+key)
    retained=[];duplicates=[];seen={}
    for row in candidates:
        hashes=[]
        for query,count in [('probe3',1760494),('qual',2817131)]:
            p=u.ROOT/row[query];a=np.load(p,mmap_mode='r')
            if a.shape!=(count,) or a.dtype!=np.float32 or not np.isfinite(a).all():raise ValueError('Every full raw pair must be finite float32')
            hashes.append(hashlib.sha256(memoryview(a).cast('B')).hexdigest())
        identity=tuple(hashes)
        if identity in seen:duplicates.append(dict(name=row['name'],duplicate_of=seen[identity]))
        else:
            seen[identity]=row['name'];row['payload_sha256']=dict(zip(['probe3','qual'],hashes));retained.append(row)
    bind(Path(__file__))
    u.write(REPORT/'manifest.json',dict(state='complete_full_probe3_own_raw_bank_all_native_pairs_verified',
        source_pin=u.PIN,probe3_rows=1760494,qualifying_rows=2817131,columns=retained,duplicates=duplicates,
        raw_columns=len(retained),all_native_query_rows_and_labels_aligned=True,
        source_sha256=sources,qualifying_labels_read=False,public_prediction_values_read=False,
        models_fitted=0,scope='Full local own native prediction bank and exact-pair deduplication only. No score/feature selection. All original producer budgets complete.'))
    print(len(retained),'full own native columns;',len(duplicates),'exact terminal/snapshot aliases',flush=True)

if __name__=='__main__':main()
