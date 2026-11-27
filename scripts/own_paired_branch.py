"""Require a unique qualifying forecast for each distinct development forecast, and refit raw rating models on that basis."""
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import own_aligned_forecasts as aligned
import own_overnight_blend as blend
import own_neighbor_experiment as ne
import own_schema_branch as schema

OUT=ne.ROOT/'own-paired-branch-20261126'
BASE='schema-01'
# Availability is decided by completed independent training runs. Paired roles have the same column semantics.
PACKAGES=[
    ('own-transition-20261126','package','stage-24'),
    ('own-suffix-factors-20261126','signal',1),
    ('own-choice-factors-20261126','signal',3),
    ('own-sparse-cohort-20261126','signal',1),
    ('own-suffix-attention-20261126','raw',None),
    ('own-cf-temporal-attention-20261126','attention',None),
    ('own-temporal-continuation-20261126','attention',None),
    ('own-hybrid-attention-20261126','raw',None),
    ('own-slate-attention-20261126','attention',None),
    ('own-single-execution-distribution-20261126','package','stage-24'),
    ('own-paired-title-neighbors-20261126','package','paired-01'),
    ('own-paired-ease-20261126','package','paired-01'),
    ('own-learned-series-20261126','package','paired-01'),
]


def audit_schema():
    source=aligned.OUT/'schema.json';data=json.loads(source.read_text())['forecasts']
    duplicates=[]
    for arm in (0,1):
        table=defaultdict(list)
        for name,v in data.items():table[v['sha256'][arm]].append(name)
        for sig,names in table.items():
            if len(names)>1:
                assert all(ne.sha(data[n]['paths'][arm])==sig for n in names)
                duplicates.append(dict(arm=('probex','qual')[arm],sha256=sig,columns=names,
                    other_role_hashes=[data[n]['sha256'][1-arm] for n in names]))
    canonical={k:v for k,v in data.items() if 'producer' in v}
    assert all(len({v['sha256'][i] for v in canonical.values()})==len(canonical) for i in (0,1))
    schema.write(OUT/'pairing.json',dict(source=str(source),source_sha256=ne.sha(source),forecast_pairs=len(data),
        canonical_single_execution_pairs=len(canonical),ambiguous_pairs=duplicates,qualifying_outcomes_read=False))


def main(ablation=False):
    shared=OUT
    folder=ne.ROOT/'own-paired-ablation-20261126' if ablation else OUT
    base='paired-03' if ablation else BASE;prefix='ablation' if ablation else 'paired'
    folder.mkdir(exist_ok=True)
    if not ablation:audit_schema()
    statefile=folder/'queue.json'
    initial=json.loads((blend.OUT/base/'freeze.json').read_text())
    if ablation and not (folder/'protocol.json').exists():
        assert not (ne.ROOT/'own-single-execution-distribution-20261126'/'selection.json').exists()
        schema.write(folder/'protocol.json',dict(registered=datetime.now().astimezone().isoformat(),base=base,
            excluded='own-single-execution-distribution-20261126',reason='Measure the incremental effect of learning across 202 raw forecast columns. Both branches use the same user folds and the same treatment of direct rating models.',
            final_selection='Report both branches. The post-calibrated inclusive pairing branch remains the primary result.'))
    state=json.loads(statefile.read_text()) if statefile.exists() else dict(parent=base,keys=initial['columns'],stages=[],bindings={},excluded_after_base=['own-aligned-forecasts-20261126','own-distribution-stack-20261126','own-verified-distribution-20261126'])
    while datetime.now()<datetime(2026,11,26,4,56):
        new=[]
        extra=shared/'extra-packages.json'
        packages=PACKAGES+(json.loads(extra.read_text()) if extra.exists() else [])
        stage=prefix+'-%02d'%(len(state['stages'])+1)
        existing=blend.OUT/stage/'freeze.json'
        recovered_keys=json.loads(existing.read_text())['columns'] if existing.is_file() else None
        for family,kind,param in packages:
            if family in state['bindings'] or (ablation and family=='own-single-execution-distribution-20261126'):continue
            folder=ne.ROOT/family;freeze=folder/'freeze.json'
            if not freeze.exists():continue
            rep=json.loads(freeze.read_text())
            if kind=='package':
                if not rep['qualifying_hashes']:continue
                assert Path(rep['parent']).name==param;key=family
                paths=[folder/'oof.npy',folder/'qual-correction.npy']
            else:
                paths=[folder/'probex.npy',folder/'qual.npy']
                if not all(p.is_file() for p in paths):continue
                key=f'{kind}:{family}:{param}' if kind=='signal' else f'{kind}:{family}'
            if key in state['keys'] or (recovered_keys is not None and key not in recovered_keys):continue
            assert not rep.get('qualifying_labels_used_for_fitting',True)
            state['bindings'][family]=dict(mode=kind,freeze=str(freeze),freeze_sha256=ne.sha(freeze),
                paths=list(map(str,paths)),hashes=[ne.sha(p) for p in paths]);new.append(key)
        if new:
            stage=prefix+'-%02d'%(len(state['stages'])+1);keys=state['keys']+new;dest=blend.OUT/stage
            if not (dest/'freeze.json').exists():
                subprocess.run([sys.executable,'-u','scripts/own_overnight_blend.py',stage,*keys,'--parent',state['parent']],check=True)
            fr=json.loads((dest/'freeze.json').read_text());assert fr['columns']==keys and fr['parent']==state['parent']
            state['stages'].append(dict(stage=stage,added=new,selected=fr['selected'],improved_folds=fr['improved_folds']))
            state.update(parent=stage,keys=keys)
        schema.write(statefile,state);time.sleep(14)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--without-distribution',action='store_true');main(p.parse_args().without_distribution)
