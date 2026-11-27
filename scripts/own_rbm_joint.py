"""Marginalize the missing slate jointly over 64 importance-weighted hidden draws from three original RBMs."""
import json
from pathlib import Path
import subprocess
import numpy as np
from catboost import CatBoostRegressor
from scipy.stats import qmc
import own_anchor_context as anchor
import own_day_states as ds
import own_full_context as full
import own_joint_cohort_source as joint
import own_neighbor_experiment as ne
import own_overnight_blend as blend
import own_retained_history as ret
OUT=ne.ROOT/'own-rbm-joint-20261126'
PARENT=blend.OUT/'schema-04'
MODELS=('RBM-C0-control-10e','RBM-C7-visible-30e','RBM-C7-visible-60e')


def main():
    OUT.mkdir(exist_ok=True);dataset=Path('data/processed/netflix.bin')
    orig=json.loads((joint.OUT/'manifest.json').read_text());assert ds.sha(dataset)==orig['training_data_sha256']
    y,f,_,_=anchor.history_control();ix=ret.map_rows();n=1408395;visible=np.ones(n,'f');visible[ix]=0;visible.tofile(OUT/'visible.f32')
    qmc.Sobol(100,scramble=True,seed=9971).random_base2(6).astype('f').tofile(OUT/'points.f32')
    subprocess.run(['c++','-O3','-std=c++17','-Wno-deprecated-declarations','scripts/own_rbm_joint.cpp','-framework','Accelerate','-o',str(OUT/'joint')],check=True)
    cmd=[str(OUT/'joint'),str(dataset),str(joint.OUT/'zero-queries.bin'),str(OUT/'visible.f32'),str(OUT/'points.f32')]
    lineage={};prior={}
    for name in ('RBM-C0-control-10e','RBM-C7-visible-10e','RBM-C7-visible-30e','RBM-C7-visible-45e','RBM-C7-visible-60e'):
        rp=Path('experiments',name+'.json');r=json.loads(rp.read_text());args=r['command']
        assert args[1]=='rbm' and args[args.index('--data')+1]==str(dataset)
        for flag in ('--init-model','--init'):
            if flag in args:assert args[args.index(flag)+1] in prior
        md,p=[r[k] for k in ('model_artifact','predictions_artifact')]
        for a in (md,p):assert ds.sha(a['path'])==a['sha256']
        prior[md['path']]=md['sha256']
        lineage[name]=dict(receipt=str(rp),receipt_sha256=ds.sha(rp),model=md)
        if name in MODELS:cmd.extend((md['path'],p['path'],str(OUT/(name+'.f32'))))
    subprocess.run(cmd,check=True)
    paths=[OUT/(k+'.f32') for k in MODELS];v=[np.memmap(p,'<f4','r').reshape(4225526,7) for p in paths]
    assert all(np.isfinite(a).all() for a in v)
    fr=json.loads((PARENT/'freeze.json').read_text());assert ne.sha(PARENT/'selected.f32')==fr['qualifying_hashes']['selected']
    base=np.load(PARENT/'known-selected.npy');qb=np.fromfile(PARENT/'selected.f32','<f4')
    def design(rows,b,phase):
        blocks=[]
        for table in v:
            a=table[rows];blocks.extend((a[:,0]-b,a[:,1:5]-a[:,:1],a[:,3]-a[:,2],a[:,5:7]))
        return np.column_stack((b,full.cf96(phase)-b,*blocks)).astype('f')
    def model():return CatBoostRegressor(iterations=600,depth=6,learning_rate=.045,l2_leaf_reg=75,thread_count=2,random_seed=4575,verbose=False,allow_writing_files=False)
    x=design(ix,base,'probex');choice,stats,held=anchor.crossfit(x,y,f,base,model)
    report=dict(parent=str(PARENT),selected=choice,known=stats,original_rbm_lineage=lineage,signals={str(p):ds.sha(p) for p in paths},
        qualifying_hashes={},qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(report,indent=2)+'\n')
    if choice!='control':
        delta=model().fit(x,y-base).predict(design(slice(n,None),qb,'qual'),thread_count=2)
        np.save(OUT/'oof.npy',held);np.save(OUT/'qual-correction.npy',delta)
        ne.clipped(qb+float(choice)*delta).astype('<f4').tofile(OUT/'selected.f32');report['qualifying_hashes']['selected']=ne.sha(OUT/'selected.f32')
    (OUT/'freeze.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
