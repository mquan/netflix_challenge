#!/usr/bin/env python3
"""Fixed split3 terminal/ep08 additions to the own103 known-ProbeX ridge bank."""
from pathlib import Path
import math
import numpy as np
import upstream_cf48_103 as temporal
import upstream_remote_batch as algebra
import upstream_repro as u

RUN=u.ROOT/'artifacts/upstream-split3-105-screen'
REPORT=u.ROOT/'experiments/upstream-split3-105-screen'
NEW=['s3-attn16','s3-attn16_ep08']

def main():
    if RUN.exists() or REPORT.exists():raise ValueError('Existing split3 screen preserved')
    receiptpath=u.ROOT/'experiments/upstream-split3-pilot/paired-receipt.json'
    proofpath=receiptpath.parent/'completion-verification.json'
    receipt,proof=u.read(receiptpath),u.read(proofpath)
    if proof['state']!='complete_original_two_phase_split3_pilot' or proof['paired_receipt_sha256']!=u.sha(receiptpath):
        raise ValueError('Original two-phase split3 completion/projection proof required')
    sources=dict(proof['bindings'])
    for name,digest in sources.items():
        if u.sha(u.ROOT/name)!=digest:raise ValueError('Split3 completion input changed: '+name)
    temporal.configure();parent=temporal.n.verify_prepared()
    parentproof=temporal.REPORT/'cache-column-verification.json'
    if u.read(parentproof)['manifest_sha256']!=u.sha(temporal.REPORT/'manifest.json'):
        raise ValueError('Own103 cache verification changed')
    rows=[]
    for row in receipt['projected_pairs']:
        if row['name'] not in NEW:continue
        paths={s:row[s] for s in ['probex','qual']}
        for s,name in paths.items():
            if u.sha(u.ROOT/name)!=row[s+'_sha256']:raise ValueError('Split3 paired vector changed')
            sources[name]=row[s+'_sha256']
        rows.append(dict(name=row['name'],paired_files=paths,duplicate_of=None))
    if [r['name'] for r in rows]!=NEW:raise ValueError('Fixed public snapshot order changed')
    for p in [receiptpath,proofpath,temporal.REPORT/'manifest.json',parentproof,Path(__file__)]:
        sources[str(p.relative_to(u.ROOT))]=u.sha(p)
    for name,digest in parent['prepared_sha256'].items():sources[str((temporal.RUN/name).relative_to(u.ROOT))]=digest
    y=np.load(temporal.RUN/'labels.npy');groups=np.load(temporal.RUN/'groups.npy')
    old=np.load(temporal.RUN/'design.npy',mmap_mode='r')
    extra=np.column_stack([np.clip(np.load(u.ROOT/r['paired_files']['probex']),0,6) for r in rows])
    if old.shape!=(len(y),179) or extra.shape!=(len(y),2):raise ValueError('Known103/projection row layout mismatch')
    REPORT.mkdir();RUN.mkdir()
    u.write(REPORT/'plan.json',dict(ready_unused=rows,fixed_additions=NEW,
        rule='Append both source-published split3 attention columns together to own103. Fixed control/all-two arms, unchanged five user folds and ridge penalty. No qualifying outcomes or feature-subset search.',
        qualifying_labels_read=False,public_prediction_values_read=False))
    sources[str((REPORT/'plan.json').relative_to(u.ROOT))]=u.sha(REPORT/'plan.json')
    scored=algebra.screen(old,extra,y,groups,dict(control=[],main=[0,1]));records=[]
    for arm,result in scored.items():
        p=RUN/(arm+'-ridge.npy');np.save(p,result['prediction'])
        score=math.sqrt(math.fsum((float(a)-float(b))**2 for a,b in zip(result['prediction'],y))/len(y))
        if arm=='control' and abs(score-parent['baseline_rmse'])>1e-12:raise ValueError('Own103 ridge control differs')
        records.append(dict(arm=arm,rmse=score,prediction_sha256=u.sha(p),
            fold_rmse=[temporal.n.x.metric(result['prediction'][groups//2==f],y[groups//2==f]) for f in range(5)]))
    np.savez(RUN/'stats.npz',gram=scored['main']['gram'],rhs=scored['main']['rhs'])
    u.write(REPORT/'ridge-screen.json',dict(state='complete',records=records,fixed_additions=NEW,source_sha256=sources,
        script_sha256=u.sha(Path(__file__)),qualifying_labels_read=False,public_prediction_values_read=False,
        scope='Own105 known-ProbeX ridge diagnostic using independently verified split3 projection. No nonlinear or qualifying promotion.'))
    print(records,flush=True)

if __name__=='__main__':main()
