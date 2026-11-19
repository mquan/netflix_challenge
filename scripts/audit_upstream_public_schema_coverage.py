#!/usr/bin/env python3
"""Map public feature names to source-resolved jobs without loading predictions."""
import argparse
from pathlib import Path
import subprocess
import upstream_repro as u

REPORT=u.ROOT/'experiments/upstream-all-pipeline-coverage'


def main(metadata):
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=u.SOURCE,text=True).strip()!=u.PIN:
        raise ValueError('Wrong source commit')
    if subprocess.check_output(['git','diff','--name-only','HEAD'],cwd=u.SOURCE,text=True).strip():
        raise ValueError('Tracked upstream source changed')
    registry_path=REPORT/'registry.json';registry=u.read(registry_path);schema=u.read(metadata)
    original=u.read(u.BASE/'registry.json')['jobs']
    if registry['new']['jobs']!=original:raise ValueError('New registry differs from original pinned parser')
    raw=schema['names'][:536];sections={}
    for split,marker,pred in [('old','/preds_old/','probe'),('3','/preds_3/','probe_3'),('new','/public-new/columns/','probex')]:
        names=[Path(p).stem for p in raw if marker in p];rows=[]
        for name in names:
            output=f'preds_{split}/{name}.{pred}.npy'
            owners=[j for j,v in registry[split]['jobs'].items() if output in v['outputs']]
            if len(owners)>1:raise ValueError('Ambiguous producer')
            row=dict(name=name,declared_output=output,exact_declared_producers=owners)
            if split=='old' and not owners:
                # Similar source recipes are candidates, not original execution proof.
                root=name.split('_ep')[0]
                row['split1_candidate_producer']='s1-'+root if 's1-'+root in registry['1']['jobs'] else None
                row['candidate_is_exact_reproduction_proof']=False
            rows.append(row)
        sections[split]=dict(raw_features=len(names),source_jobs=len(registry[split]['jobs']),
            exact_declared_outputs=sum(bool(r['exact_declared_producers']) for r in rows),records=rows)
    if [sections[s]['raw_features'] for s in ['old','3','new']]!=[122,263,151]:
        raise ValueError('Public raw feature groups differ')
    source_hashes={}
    files=subprocess.check_output(['git','ls-files'],cwd=u.SOURCE,text=True).splitlines()
    for name in files:
        p=u.SOURCE/name
        if p.is_file():source_hashes[name]=u.sha(p)
    proof=dict(source_pin=u.PIN,registry_sha256=u.sha(registry_path),
        source_tracked_files_sha256=source_hashes,
        registry_helper_sha256=u.sha(u.ROOT/'scripts/upstream_all_registry.rs'),
        registry_binary_sha256=u.sha(u.BASE/'target/release/repro_all_registry'),
        auditor_sha256=u.sha(Path(__file__)),public_metadata_sha256=u.sha(metadata),
        raw_features=536,meta_features=83,support_features=220,sections=sections,
        public_prediction_values_read=False,qualifying_labels_read=False,
        source_registry_is_training_completion_proof=False,
        next_lane='Reproduce split3 native jobs in an isolated tree, with original capacities/epochs/commands and qualifying targets zero. Project source probe_3 outputs onto known ProbeX only after unique row-identity and training-disjointness verification. Fit meta learners locally.',
        data_preparation_rule='Reuse verified own ingested data; do not execute ingest jobs that reference recovered qualifying ratings. Source recipe availability does not authorize label import.',
        scope='Metadata and pinned-source coverage audit; no new training, prediction, or score claim.')
    u.write(REPORT/'coverage.json',proof)
    print({s:{k:v for k,v in section.items() if k!='records'} for s,section in sections.items()})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--metadata',required=True,type=Path);main(p.parse_args().metadata)
