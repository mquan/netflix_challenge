"""Compare fitting and prediction coordinates for the same users and movies without outcome labels."""
import json
from pathlib import Path
import re
import numpy as np
import own_full_ridge as bank
import own_neighbor_experiment as ne
import own_trajectory_bank as trajectory

OUT=ne.ROOT/'own-split-diagnostics-20261126'


def first_rows(field):
    vals=[ne.history(phase,field) for phase in ('probex','qual')]
    n=max(int(v.max()) for v in vals)+1;tables=[]
    for v in vals:
        ids,ix=np.unique(v,return_index=True);rows=np.full(n,-1,'i');rows[ids]=ix;tables.append(rows)
    pick=(tables[0]>=0)&(tables[1]>=0)
    return [row[pick][::5] for row in tables]


def main():
    OUT.mkdir(exist_ok=True)
    common={k:first_rows(f) for k,f in [('u','user_idxs'),('i','item_idxs')]}
    report={};allcols=json.loads((trajectory.OUT/'sources.json').read_text())['support']
    families={}
    for key,info in allcols.items():
        tag=re.search(r'/([^/]+)\.([ui])[0-9]+\.probex.npy$',key)
        if not tag:continue
        family,mode=tag.groups();a,b=[np.load(path,mmap_mode='r')[ix] for path,ix in zip(info['paths'],common[mode])]
        r=float(np.corrcoef(a,b)[0,1]);report[key]=r;families.setdefault(family+'.'+mode,[[],[]])
        families[family+'.'+mode][0].append(a);families[family+'.'+mode][1].append(b)
    derived={}
    for family,(a,b) in families.items():
        a,b=np.column_stack(a),np.column_stack(b)
        a=(a-a.mean(0))/a.std(0).clip(.00001);b=(b-b.mean(0))/b.std(0).clip(.00001)
        cross=b.T@a/len(a);u,_,v=np.linalg.svd(cross,full_matrices=False)
        r=float(np.mean(a*(b@(u@v))))
        derived[family]=dict(width=a.shape[1],median_same_coordinate_correlation=float(np.median(np.diag(cross))),aligned_mean_correlation=r)
    cols=bank.sources();types={}
    for key in cols:
        name=key.rsplit('/',1)[-1]
        category='virtual rating-history statistic' if name.startswith('vf') else 'named training output'
        types[category]=types.get(category,0)+1
    result=dict(shared_entity_correlations=report,matching_rows=derived,bank_output_types=types,qualifying_ratings_read=False)
    (OUT/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'matching_rows':derived,'bank_output_types':types},indent=2))


if __name__=='__main__':main()
