"""Transfer a user's learned historical day effects to their terminal movies, conditioning on observed ratings."""
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from catboost import CatBoostRegressor
import own_anchor_context as anchor
import own_anchor_covariance as cov
import own_full_context as full
import own_joint_cohort_source as joint
import own_neighbor_experiment as ne
import own_overnight_blend as blend
import own_retained_history as ret
OUT=ne.ROOT/'own-day-states-20261126'
PARENT=blend.OUT/'schema-04'


def sha(path):
    digest=hashlib.sha256()
    with open(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(8<<20),b''):digest.update(chunk)
    return digest.hexdigest()


def main():
    OUT.mkdir(exist_ok=True);ds=Path('data/processed/netflix.bin')
    p=Path('experiments/J4-joint128-day128.json');r=json.loads(p.read_text())
    parent=json.loads((joint.OUT/'manifest.json').read_text())['models']['J3-joint128-reg005']
    cmd=r['command'];assert cmd[1]=='joint' and cmd[cmd.index('--data')+1]==str(ds) and cmd[cmd.index('--init-model')+1]==parent['model_artifact']['path']
    assert sha(parent['model_artifact']['path'])==parent['model_artifact']['sha256']
    m,pred=[r[k] for k in ('model_artifact','predictions_artifact')];assert sha(m['path'])==m['sha256'] and sha(pred['path'])==pred['sha256']
    u,i,d,old=cov.original();seen,_=anchor.training_indices('trainx',u,i,d);ix=ret.map_rows()
    np.column_stack((old,seen)).astype('f').tofile(OUT/'anchors.f32')
    subprocess.run(['c++','-O3','-std=c++17','-Wno-deprecated-declarations','scripts/own_day_states.cpp','-framework','Accelerate','-o',str(OUT/'day-states')],check=True)
    raw=OUT/'states.f32';subprocess.run([str(OUT/'day-states'),str(ds),str(joint.OUT/'zero-queries.bin'),m['path'],
                  str(OUT/'anchors.f32'),str(raw),pred['path']],check=True)
    v=np.memmap(raw,'<f4','r').reshape(4225526,20);assert np.isfinite(v).all()
    y,f,_,_=anchor.history_control();fr=json.loads((PARENT/'freeze.json').read_text());assert ne.sha(PARENT/'selected.f32')==fr['qualifying_hashes']['selected']
    base=np.load(PARENT/'known-selected.npy');qb=np.fromfile(PARENT/'selected.f32','<f4')
    def design(v,b,phase):
        ref=np.memmap(parent['vector'],'<f4','r')[ix if phase=='probex' else slice(len(old),None)]
        return np.column_stack((v,v[:,0]-b,v[:,0]-ref,b,full.cf96(phase)-b)).astype('f')
    def model():return CatBoostRegressor(iterations=650,depth=6,learning_rate=.043,l2_leaf_reg=75,thread_count=2,
                              random_seed=5264,verbose=False,allow_writing_files=False)
    x=design(v[ix],base,'probex');choice,stats,held=anchor.crossfit(x,y,f,base,model)
    rep=dict(parent=str(PARENT),selected=choice,known=stats,training_receipt_sha256=sha(p),initialization=parent,
        state_features_sha256=sha(raw),qualifying_hashes={},qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(rep,indent=2)+'\n')
    if choice!='control':
        delta=model().fit(x,y-base).predict(design(v[len(old):],qb,'qual'),thread_count=2)
        np.save(OUT/'oof.npy',held);np.save(OUT/'qual-correction.npy',delta)
        ne.clipped(qb+float(choice)*delta).astype('<f4').tofile(OUT/'selected.f32');rep['qualifying_hashes']['selected']=ne.sha(OUT/'selected.f32')
    (OUT/'freeze.json').write_text(json.dumps(rep,indent=2)+'\n')


if __name__=='__main__':main()
