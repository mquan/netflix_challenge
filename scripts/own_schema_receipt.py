"""Verify and publish the development-selected branch whose forecast inputs retain their rating semantics."""
import argparse
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import time
import hashlib
import numpy as np
import own_anchor_context as anchor
import own_neighbor_experiment as ne
import own_overnight_blend as blend
import own_overnight_receipt as full
import own_schema_branch as branch

DEST=Path('experiments/OWN-schema-overnight-20261126.json')


def read(path):return json.loads(path.read_text())


def streamed_sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def verify_recipe(root,frozen):
    y,folds,dev_base,qual_base=anchor.history_control()
    checks={}
    for known,base in ((True,dev_base),(False,qual_base)):
        src=frozen['known_sources' if known else 'qual_sources']
        weights=np.array(frozen['fold_weights'] if known else [frozen['weights']])
        expected=(np.load(root/'known-selected.npy',mmap_mode='r') if known
                  else np.memmap(root/'selected.f32','<f4','r'))
        assets=[]
        for key in frozen['columns']:
            ref=src[key];path=Path(ref['path']);assert streamed_sha(path)==ref['sha256']
            v=np.load(path,mmap_mode='r') if path.suffix=='.npy' else np.memmap(path,'<f4','r')
            if key.startswith('signal:'):v=v[:,::int(key.rsplit(':',1)[1])]
            if key=='bayes' and known:
                import own_retained_history as history
                v=v[history.map_rows()]
            assets.append((v.reshape(len(base),ref['width']),key=='retained' or key=='bayes' or key.startswith(('attention','raw:'))))
        worst=0.;rss=0.;rounded_diff=0
        for j in range(0,len(base),32768):
            sl=slice(j,min(len(base),j+32768));b=base[sl];acc=b.copy();offset=0
            for v,absolute in assets:
                a=v[sl].astype(float);width=a.shape[1]
                if absolute:a-=b[:,None]
                if known:
                    for fold in range(5):
                        mask=folds[sl]==fold;acc[mask]+=a[mask]@weights[fold,offset:offset+width]
                else:acc+=a@weights[0,offset:offset+width]
                offset+=width
            assert offset==weights.shape[1]
            acc=ne.clipped(acc)
            if not known:acc=acc.astype('<f4')
            diff=acc-expected[sl];worst=max(worst,float(np.abs(diff).max()))
            if not known:rounded_diff+=int(np.count_nonzero(diff))
            if known:rss+=float(np.square(acc-y[sl]).sum())
        assert worst<(5e-12 if known else 5e-7),worst
        info=dict(rows=len(base),maximum_absolute_difference=worst,verified_source_files=len(assets))
        if known:
            info['reconstructed_rmse']=(rss/len(base))**.5
            assert abs(info['reconstructed_rmse']-frozen['candidate']['pooled'])<1e-12
        else:info['different_float32_values']=rounded_diff
        checks['probex' if known else 'qualifying']=info
    return checks


def add_inventory(report,root,stage):
    report['recipe_reconstruction']=verify_recipe(root,stage)
    inventory=read(full.DEST)
    if 'candidate_inventory' in inventory:
        inventory_rows=inventory['candidate_inventory'].values()
        from collections import Counter
        report['experiment_inventory']=dict(path=str(full.DEST),sha256=streamed_sha(full.DEST),
            recorded_at=inventory['inventory_as_of'],status_counts=dict(Counter(v['status'] for v in inventory_rows)))
    report['active_branch_source_sha256']={str(p):streamed_sha(p) for p in sorted(Path('scripts').glob('own*'))
                                             if p.suffix in ('.cpp','.py','.hpp')}
    report['selected_package_receipts']={}
    import own_overnight_queue as queue
    roots=dict(queue.FAMILIES)
    roots.update({k:Path(v) for k,v in read(blend.OUT/'extra-families.json').items()})
    for k in stage['columns']:
        d=roots.get(k)
        if d and (d/'freeze.json').exists():report['selected_package_receipts'][k]=streamed_sha(d/'freeze.json')
    raw={}
    for name in ('choice-factors','suffix-factors','suffix-impute','exact-attention','cf-temporal-attention',
                 'temporal-continuation','slate-attention','hybrid-attention'):
        folder=ne.ROOT/('own-'+name+'-20261126')
        for fname in ('probex.npy','qual.npy','trainx.pt','fulltrain.pt'):
            p=folder/fname
            if p.is_file():raw[str(p)]=streamed_sha(p)
    report['supplemental_raw_model_artifacts']=raw
    from collections import Counter
    import own_paired_branch as paired_branch
    extra=paired_branch.OUT/'extra-packages.json'
    items=paired_branch.PACKAGES+(read(extra) if extra.exists() else [])
    panel=dict(inventory.get('candidate_inventory',{}))
    for family,kind,param in items:
        if family in panel:continue
        folder=ne.ROOT/family
        record={'directory':str(folder),'status':'no_completed_development_selection'}
        frozen=folder/'freeze.json';selected=folder/'selection.json'
        if frozen.exists() or selected.exists():
            f=read(frozen if frozen.exists() else selected)
            record.update(status='completed' if frozen.exists() else 'development_completed',development_choice=f['selected'])
            if frozen.exists():record['freeze_sha256']=streamed_sha(frozen)
            if f['qualifying_hashes']:
                record['frozen_vector_sha256']=f['qualifying_hashes']['selected']
                assert streamed_sha(folder/'selected.f32')==record['frozen_vector_sha256']
        panel[family]=record
    report['combined_candidate_inventory']=panel
    report['combined_status_counts']=dict(Counter(v['status'] for v in panel.values()))



def update(force=False,inventory=False,paired=False):
    dest=Path('experiments/OWN-paired-overnight-20261126.json') if paired else DEST
    if paired:
        import own_paired_branch as pair
        state=read(pair.OUT/'queue.json')
        state['excluded']={k:'One or more input forecasts have a different duplicate in each role.' for k in state['excluded_after_base']}
    else:state=read(branch.OUT/'queue.json')
    root=blend.OUT/state['parent']
    if dest.exists() and not force:
        old=read(dest)
        if old['stage']==root.name and old['excluded_directions']==state['excluded']:return
    frozen=read(root/'freeze.json');digest=frozen['qualifying_hashes']['selected']
    assert ne.sha(root/'selected.f32')==digest
    audited=root
    while not (audited/'audit.json').exists():
        r=read(audited/'freeze.json')
        if r['selected']!='parent':return
        audited=blend.OUT/r['parent']
    spec=importlib.util.spec_from_file_location('own_stage_verifier',full.VERIFY)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    scores=mod.verify_audit(audited,{'selected':digest})['selected']
    base=read(ne.BASE/'followup/audit.json')['separate_context_and_tree']
    stage=read(audited/'freeze.json')
    checks=[ne.ROOT/v for v in ('own-split-diagnostics-20261126/report.json','own-aligned-forecasts-20261126/schema.json',
           'own-peer-replay-20261126/manifest.json','own-peer-cf-20261126/manifest.json','own-extra-joint-20261126/manifest.json',
           'own-suffix-attention-20261126/chronology.json','own-peer-temporal-20261126/manifest.json',
           'own-peer-innovation-20261126/protocol.json','own-day-states-20261126/freeze.json','own-rbm-joint-20261126/freeze.json')]
    report=dict(as_of=datetime.now().astimezone().isoformat(),stage=root.name,selected_at=audited.name,target=.85725,
        scope='Predictions from models trained locally on the Netflix ratings and their own-model descendants.',
        branch_reason='The earlier distribution stack processed 14 history statistics as absolute rating forecasts. This branch resumes at the preceding checkpoint and excludes its trained descendants.',
        parent_before_schema_correction=branch.ANCHOR,selection_rule='Lower pooled ProbeX RMSE and improvement in at least four of five fixed user folds.',
        qualifying_labels_used_for_fitting_or_selection=False,known_comparison_is_nested=False,recovered_labels_are_official_evaluation=False,
        selected_probe=stage['candidate'],qualifying=scores,baseline=base,baseline_receipt=str(full.BASE),
        improvement_over_baseline={k:base[k]['rmse']-v['rmse'] for k,v in scores.items()},
        distance_above_target={k:v['rmse']-.85725 for k,v in scores.items()},qualifying_vector=str(root/'selected.f32'),
        sha256=digest,freeze_sha256=ne.sha(root/'freeze.json'),excluded_directions=state['excluded'],stages=state['stages'],
        prediction_sources=stage['qual_sources'],independent_verifier=str(full.VERIFY),verifier_sha256=ne.sha(full.VERIFY),
        data_and_schema_checks={str(p):ne.sha(p) for p in checks if p.exists()})
    if paired:
        report['branch_reason']='Thirteen distinct ProbeX forecast pairs collapse to identical qualifying bytes. This branch restarts from before those pairs entered the forecast features and admits independently generated signals or single-execution forecast pairs.'
        report['parent_before_schema_correction']=pair.BASE
        report['paired_input_audit']={'path':str(pair.OUT/'pairing.json'),'sha256':streamed_sha(pair.OUT/'pairing.json')}
        report['pairing_branch_source_bindings']=state['bindings']
        report['exploratory_design']='Protocol revised after inspecting earlier recovered scores. All weights and the admission of individual predictions use ProbeX only.'
    if inventory:add_inventory(report,root,stage)
    branch.write(dest,report)
    print(root.name,scores,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--watch',action='store_true');ap.add_argument('--inventory',action='store_true');ap.add_argument('--paired',action='store_true');args=ap.parse_args()
    while True:
        update(not args.watch,args.inventory,args.paired)
        if not args.watch or datetime.now()>=datetime(2026,11,26,4,59,30):break
        time.sleep(25)
