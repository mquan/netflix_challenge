"""Record the latest development-selected vector after independent streaming verification."""
import argparse
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import time

import own_neighbor_experiment as ne
import own_overnight_blend as blend

DEST = Path('experiments/OWN-overnight-20261126.json')
BASE = Path('experiments/OWN17X-cf96-context-tree-20261124.json')
VERIFY = ne.RANK/'evaluate.py'


def read(p):
    return json.loads(p.read_text())


def update():
    state = read(blend.OUT/'queue.json')
    stage = blend.OUT/state['parent']
    if DEST.exists() and read(DEST)['stage'] == stage.name:
        return
    frozen = read(stage/'freeze.json')
    audit_dir = stage
    while not (audit_dir/'audit.json').exists():
        prior = read(audit_dir/'freeze.json')
        if prior['selected'] != 'parent':
            return
        audit_dir = blend.OUT/prior['parent']
    assert ne.sha(stage/'selected.f32') == frozen['qualifying_hashes']['selected']
    spec = importlib.util.spec_from_file_location('rank_audit', VERIFY)
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    scores = verifier.verify_audit(audit_dir, frozen['qualifying_hashes'])['selected']
    target = .85725
    base = read(ne.BASE/'followup'/'audit.json')['separate_context_and_tree']
    summary = dict(as_of=datetime.now().astimezone().isoformat(),stage=stage.name,experiment_root=str(blend.OUT),
        target=target,scope='Only locally trained Netflix ratings models and their descendants.',
        baseline_receipt=str(BASE),baseline_receipt_sha256=ne.sha(BASE),baseline=base,
        selection_rule='Lower pooled ProbeX RMSE and improvement in at least four of five fixed user folds.',
        known_comparison_is_nested=False,qualifying_labels_used_for_fitting_or_selection=False,
        recovered_labels_are_official_evaluation=False,stages=state['stages'],
        selected_probe=read(audit_dir/'freeze.json')['candidate'],
        qualifying_vector=str(stage/'selected.f32'),sha256=frozen['qualifying_hashes']['selected'],
        freeze_sha256=ne.sha(stage/'freeze.json'),qualifying=scores,
        improvement_over_baseline={g:base[g]['rmse']-s['rmse'] for g,s in scores.items()},
        distance_above_target={g:s['rmse']-target for g,s in scores.items()},
        independent_verifier=str(VERIFY),verifier_sha256=ne.sha(VERIFY),
        predictions_provenance=read(audit_dir/'freeze.json')['qual_sources'])
    tmp = DEST.with_suffix('.tmp')
    tmp.write_text(json.dumps(summary,indent=2)+'\n')
    tmp.replace(DEST)
    print(stage.name,scores,flush=True)


def inventory():
    import own_overnight_queue as queue
    state=read(blend.OUT/'queue.json')
    result=read(DEST)
    assert result['stage']==state['parent']
    assert ne.sha(result['qualifying_vector'])==result['sha256']
    roots=dict(queue.FAMILIES)
    roots.update({k:Path(v) for k,v in read(blend.OUT/'extra-families.json').items()})
    stages={}
    for trial in state['stages']:
        for family in trial['added']:stages[family]=trial['name']
    panel={}
    for name,root in roots.items():
        record=dict(directory=str(root))
        frozen=root/'freeze.json';selection=root/'selection.json'
        if frozen.exists():
            report=read(frozen);record['freeze_sha256']=ne.sha(frozen)
            record['development_choice']=report['selected']
            if name in stages:record['joint_evaluation_stage']=stages[name]
            if 'parent' in report:record['development_parent']=report['parent']
            if report['qualifying_hashes']:
                record['frozen_vector_sha256']=report['qualifying_hashes']['selected']
                assert ne.sha(root/'selected.f32')==record['frozen_vector_sha256']
            record['status']='completed'
        elif selection.exists():
            record['status']='development_completed'
            record['development_choice']=read(selection)['selected']
        else:
            record['status']='no_completed_development_selection'
        panel[name]=record
    result['candidate_inventory']=panel
    refs=[
        Path('artifacts/rank1000-knnf-20261125/inputs.json'),
        Path('../netflix_upstream_repro/experiments/upstream-cf96-complete-bank/manifest.json'),
        Path('../netflix_upstream_repro/experiments/upstream-factored-rbm-gpu400-deterministic-metal/paired-verification.json'),
        Path('artifacts/own-joint-cohort-source-20261126/manifest.json'),
        Path('artifacts/own-producer-bank-20261126/sources.json'),
        Path('artifacts/own-trajectory-bank-20261126/sources.json'),
        Path('experiments/QUAL-B2-fixed128-recipe-20261116.json'),
    ]
    result['additional_training_lineage_receipts']={str(path):ne.sha(path) for path in refs if path.is_file()}
    result['experiment_code_sha256']={str(p):ne.sha(p) for p in sorted(Path('scripts').glob('own*')) if p.suffix in ('.py','.cpp','.hpp')}
    chronology=ne.ROOT/'own-suffix-attention-20261126/chronology.json'
    if chronology.exists():
        result['competition_chronology']={'path':str(chronology),'sha256':ne.sha(chronology),**read(chronology)}
    result['inventory_as_of']=datetime.now().astimezone().isoformat()
    tmp=DEST.with_suffix('.inventory.tmp');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(DEST)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch',action='store_true')
    parser.add_argument('--inventory',action='store_true')
    args = parser.parse_args()
    while True:
        update()
        if args.inventory:inventory()
        if not args.watch or datetime.now().timestamp()>=datetime(2026,11,26,4,59,30).timestamp():
            break
        time.sleep(25)
