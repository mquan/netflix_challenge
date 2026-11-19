#!/usr/bin/env python3
"""Screen our completed, unused predictors named in the public feature schema.

Reads public JSON metadata only. All numerical features come from our own
receipt-backed training outputs; qualifying labels are never opened.
"""
import argparse
import datetime as dt
import hashlib
import os
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '2')
import numpy as np
import upstream_repro as u
import upstream_production_bank as bank
import upstream_source_extension as extension
import upstream_temporal93 as temporal
import upstream_remote_batch as algebra
from upstream_missing_screen import producer_for
from upstream_full_reproduce import receipt_valid

REPORT = u.ROOT/'experiments/upstream-ready-public-schema'
RUN = u.ROOT/'artifacts/upstream-ready-public-schema'


def content(path, clip):
    values = np.load(path, mmap_mode='r', allow_pickle=False)
    h = hashlib.sha256()
    for lo in range(0, len(values), 65536):
        a = values[lo:lo+65536]
        h.update(np.ascontiguousarray(np.clip(a, 0, 6) if clip else a).tobytes())
    return h.hexdigest()


def main(metadata):
    if REPORT.exists() or RUN.exists():
        raise ValueError('Existing experiment preserved')
    schema = u.read(metadata)
    if schema['columns'] != 619 or len(schema['names']) != 839:
        raise ValueError('Unexpected public schema')
    raw = schema['names'][:536]
    names = [Path(p).stem for p in raw if '/public-new/columns/' in p]
    if len(names) != 151 or len(set(names)) != 151:
        raise ValueError('Unexpected new-split feature schema')
    base_rows, jobs = bank.recipe()
    present = {r['name'] for r in base_rows} | set(extension.NEW) | set(temporal.NEW)
    if len(present) != 93:
        raise ValueError('Frozen own93 recipe differs')
    sources = {}
    parent = u.read(temporal.REPORT/'manifest.json')
    for name, digest in parent['prepared_sha256'].items():
        path = temporal.RUN/name
        if u.sha(path) != digest:
            raise ValueError('Parent bank changed: '+name)
        sources[str(path.relative_to(u.ROOT))] = digest
    sources[str((temporal.REPORT/'manifest.json').relative_to(u.ROOT))] = u.sha(temporal.REPORT/'manifest.json')
    records, ready, verified = [], [], {}
    for name in names:
        owner, reason = producer_for(name, jobs)
        row = dict(name=name, producer=owner, resolution=reason, already_in_own93=name in present)
        rp = u.BASE/'fresh-receipts'/(str(owner)+'.json')
        if name not in present and owner and rp.exists():
            if owner not in verified:
                verified[owner] = receipt_valid(owner, jobs)
            if verified[owner]:
                rec = u.read(rp)
                paths = {s: 'preds_new/'+name+'.'+s+'.npy' for s in ['probex', 'qual']}
                if all(p in rec['outputs'] for p in paths.values()):
                    row['paired_files'] = {s: str((u.FRESH/p).relative_to(u.ROOT)) for s,p in paths.items()}
                    row['sha256'] = {s: rec['outputs'][p] for s,p in paths.items()}
                    row['receipt'] = str(rp.relative_to(u.ROOT))
                    sources[row['receipt']] = u.sha(rp)
                    for s,count in [('probex',1128164),('qual',2817131)]:
                        p = u.ROOT/row['paired_files'][s]
                        a = np.load(p,mmap_mode='r',allow_pickle=False)
                        if a.shape != (count,) or a.dtype != np.dtype('float32') or not np.isfinite(a).all():
                            raise ValueError('Malformed own prediction: '+name)
                        sources[row['paired_files'][s]] = row['sha256'][s]
                    ready.append(row)
        records.append(row)
    # Deduplicate the actual clipped known vectors, checking paired qual values
    # only for equal known vectors. No labels enter this decision.
    own = u.read(u.ROOT/'experiments/upstream-phase-bank/manifest.json')['columns']
    own += [dict(name=name,clip=True,files={s:str((u.FRESH/f'preds_new/{name}.{s}.npy').relative_to(u.ROOT)) for s in ['probex','qual']}) for name in extension.NEW+temporal.NEW]
    seen = {}
    for r in own:
        p = u.ROOT/r['files']['probex']
        seen.setdefault(content(p,r['clip']),[]).append(r)
    selected=[]
    for row in ready:
        digest = content(u.ROOT/row['paired_files']['probex'],True)
        duplicate=None
        for old in seen.get(digest,[]):
            if content(u.ROOT/row['paired_files']['qual'],True) == content(u.ROOT/old['files']['qual'],old['clip']):
                duplicate=old['name']; break
        row['duplicate_of']=duplicate
        if duplicate is None:
            selected.append(row)
            seen.setdefault(digest,[]).append(dict(name=row['name'],clip=True,files=row['paired_files']))
    REPORT.mkdir(parents=True); RUN.mkdir()
    plan=dict(created=dt.datetime.now(dt.timezone.utc).isoformat(),source_pin=u.PIN,
        public_schema_sha256=u.sha(metadata),public_prediction_values_read=False,
        raw_columns=536,legacy_raw_columns=122,split3_raw_columns=263,new_raw_columns=151,
        meta_columns=83,support_columns=220,own93_new_schema_overlap=len(set(names)&present),
        coverage=records,ready_unused=ready,fixed_additions=[r['name'] for r in selected],
        selection_rule='All complete unused published new-split raw predictors, paired-value duplicates removed before scoring. No score-based subset selection.',
        scope='Public schema coverage is not full-bank reproduction. Current registry covers the new split; older splits and meta learners remain separate work.',
        qualifying_labels_read=False,code_sha256=u.sha(Path(__file__)))
    u.write(REPORT/'plan.json',plan)
    sources[str((REPORT/'plan.json').relative_to(u.ROOT))]=u.sha(REPORT/'plan.json')
    if not selected:
        raise ValueError('No unused predictors available')
    old=np.load(temporal.RUN/'design.npy',mmap_mode='r')
    y=np.load(temporal.RUN/'labels.npy');groups=np.load(temporal.RUN/'groups.npy')
    extra=np.column_stack([np.clip(np.load(u.ROOT/r['paired_files']['probex'],allow_pickle=False),0,6) for r in selected])
    result=algebra.screen(old,extra,y,groups,dict(control=[],main=list(range(len(selected)))))
    scores=[]
    for arm,values in result.items():
        p=RUN/(arm+'-ridge.npy');np.save(p,values['prediction'])
        score=extension.x.metric(values['prediction'],y)
        if arm=='control' and abs(score-parent['baseline_rmse'])>1e-10:
            raise ValueError('Own93 control mismatch')
        scores.append(dict(arm=arm,rmse=score,prediction_sha256=u.sha(p),
            fold_rmse=[extension.x.metric(values['prediction'][groups//2==f],y[groups//2==f]) for f in range(5)]))
    u.write(REPORT/'ridge-screen.json',dict(state='complete',records=scores,source_sha256=sources,
        script_sha256=u.sha(Path(__file__)),fixed_additions=plan['fixed_additions'],
        qualifying_labels_read=False,public_prediction_values_read=False,
        scope='Known-ProbeX fixed all-ready ridge diagnostic, not a nonlinear promotion or qualifying score.'))
    print(dict(ready=len(ready),selected=plan['fixed_additions'],scores=scores),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--metadata',type=Path,required=True)
    main(p.parse_args().metadata)
