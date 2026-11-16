#!/usr/bin/env python3
"""Import explicitly audited Probe-fitted first-level blends, never Quiz fits.

This differs from raw-base imports: these are PRETRAINED BLENDERS fitted on
development ratings. Source random-row cross-fitting is not nested within our
user folds, and legacy new-split epoch-weight inputs were fitted on ProbeX.
Consequently subsequent development CV is not independent. Qualifying ratings
are never fitted by the selected source jobs or by this importer.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import time
import urllib.request
import numpy as np
import tomli
from public_probe_bank import BASE_URL,verified_file
from streaming_blend import digest


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--index',required=True);ap.add_argument('--source',required=True)
    ap.add_argument('--output',required=True);ap.add_argument('--jobs',type=int,default=6)
    ap.add_argument('--include-higher-probe',action='store_true')
    ap.add_argument('--reuse-root')
    a=ap.parse_args();root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
    source=Path(a.source)
    index={p:dict(path=p,size=int(n),md5=h) for p,n,h in re.findall(
        r'path = "([^"]+)"\s+size = (\d+)\s+md5 = "([^"]+)"',Path(a.index).read_text())}
    original_rows=np.fromfile('artifacts/public-new/dataset/original-probe-rows.u32',dtype='<u4')
    s3map=np.fromfile('artifacts/public-bank/split3-map.u32',dtype='<u4')[original_rows]
    newmap=np.fromfile('artifacts/public-new/dataset/processed-to-probex.u32',dtype='<u4')
    oldmap=np.fromfile('artifacts/public-bank/independent-source-map.u32',dtype='<u4')[original_rows]
    if len(s3map)!=1128164 or len(newmap)!=1128164:raise ValueError('unexpected ProbeX mapping')
    proofs={str(p):digest(p) for p in [Path(a.index),
        Path('artifacts/public-new/dataset/manifest.json'),
        Path('artifacts/public-new/dataset/original-probe-rows.u32'),
        Path('artifacts/public-new/dataset/processed-to-probex.u32'),
        Path('artifacts/public-bank/split3-map.u32')]}
    code=['src/blend.rs','src/lib.rs','src/bin/gbm.rs','src/bin/mlp.rs','src/bin/fwls.rs',
          'pipeline-3.toml','pipeline-new.toml','models-3.toml','models-new.toml',
          'voting-3.toml','voting-new.toml','pipeline-old.toml','models-old.toml','voting-old.toml']
    proofs.update({str(source/p):digest(source/p) for p in code})
    planned=[];omitted=[]
    for split,groups,pr,mapping,rows in [('3',['tx','rbm','nn','other'],'probe_3',s3map,1760494),
                                       ('new',['gold','silver'],'probex',newmap,1128164),
                                       ('old',['integrated'],'probe',oldmap,1408395)]:
        pipeline=tomli.loads((source/('pipeline-'+split+'.toml')).read_text())
        for name,job in pipeline['jobs'].items():
            if job.get('jobtype') not in ['gbmblend','mlpblend','fwlsblend']:continue
            if split=='old':
                # Other old jobs depend on undocumented historical tsvd_fwls
                # input columns. These two use only audited raw SVDs/constants.
                if not a.include_higher_probe or name not in ['fwls3','fwls4']:continue
            elif not a.include_higher_probe and job.get('groups')!=groups:continue
            for seed in job.get('seeds',[1]):
                stem=f'preds_{split}/{name}-s{seed}';dev=stem+'.'+pr+'.npy';qual=stem+'.qual.npy'
                if dev not in index or qual not in index:omitted.append(stem);continue
                planned.append(dict(split=split,job=name,seed=seed,job_config=job,source_rows=rows,
                                    development=index[dev],qualifying=index[qual]))
    policy=dict(source_commit='49c9b3676e6bab90fdf77e9a2da984edbfe4fe5c',
        source_model_kind='Pretrained Probe-fitted blender, not raw base predictor',
        fit_proof='src/blend.rs cvk_blend/cvk_blend_std pass only x_pr/y_pr into B::fit; y_ql is used only for row count and post-fit diagnostic. src/bin/fwls.rs forms Gram/RHS only from y_pr. src/lib.rs epoch_blend fits split.pr.',
        selected_jobs=('Source3/new Probe-only gbm/mlp/fwls jobs, including higher Probe-fit layers; old only fwls3/4 on audited raw integrated SVDs and constants. Other old jobs excluded for undocumented tsvd_fwls dependencies.'
            if a.include_higher_probe else 'Only gbm/mlp/fwls jobs on raw model groups (3:tx/rbm/nn/other, new:gold/silver), no final ridge ensembles or higher-level blend dependencies'),
        qualifying_outcomes_used_for_weight_fitting=False,
        source_qualifying_labels_used_for_postfit_diagnostics=True,
        independent_development_cv=False,
        development_limitation='Source folds split rows, not users; new-split legacy epoch-weight inputs fit all ProbeX. Later user-fold development scores cannot be claimed independent or strictly nested.',
        source_hyperparameters_adapted_to_public_quiz=True)
    plan=dict(policy=policy,proof_files=proofs,columns=planned,omitted=omitted,qualifying_labels_read=False)
    (root/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(stage='plan',columns=len(planned),omitted=omitted)),flush=True)
    def fetch(entry):
        path=root/'source'/entry['path'];path.parent.mkdir(parents=True,exist_ok=True)
        if a.reuse_root:
            cached=Path(a.reuse_root)/'source'/entry['path']
            if verified_file(cached,entry):os.link(cached,path);return path
        for attempt in range(4):
            try:
                with urllib.request.urlopen(BASE_URL+entry['path'],timeout=90) as response:payload=response.read()
                if len(payload)!=entry['size'] or hashlib.md5(payload).hexdigest()!=entry['md5']:
                    raise ValueError('download differs from index')
                path.write_bytes(payload);return path
            except Exception:
                if attempt==3:raise
                time.sleep(attempt+1)
    def prepare(item):
        dev=fetch(item['development']);qual=fetch(item['qualifying'])
        values=np.load(dev,allow_pickle=False);query=np.load(qual,allow_pickle=False)
        if values.shape!=(item['source_rows'],) or query.shape!=(2817131,) or not np.isfinite(values).all() or not np.isfinite(query).all():
            raise ValueError('invalid pretrained predictions')
        mapping={'3':s3map,'new':newmap,'old':oldmap}[item['split']]
        dest=root/f"{item['split']}__{item['job']}-s{item['seed']}.f32"
        values[mapping].astype('<f4').tofile(dest)
        return dict(path=str(dest.resolve()),sha256=digest(dest),source=item['development']['path'],
            raw_development_sha256=digest(dev),qualifying_path=str(qual.resolve()),qualifying_sha256=digest(qual),
            qualifying_source=item['qualifying']['path'],job=item['job'],split=item['split'],seed=item['seed'])
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
        records=[]
        for item in pool.map(prepare,planned):
            records.append(item);print(json.dumps(dict(stage='verified',columns=len(records),source=item['source'])),flush=True)
    (root/'manifest.json').write_text(json.dumps(dict(state='complete',policy=policy,plan_sha256=digest(root/'plan.json'),columns=records),indent=2)+'\n')


if __name__=='__main__':main()
