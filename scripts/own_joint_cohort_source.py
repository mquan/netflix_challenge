"""Replay five original-rating factor models on every Probe and qualifying address with the same training history."""
import json
from pathlib import Path
import subprocess
import numpy as np
from gbdt_blend import PROBE
import own_anchor_context as anchor
import own_neighbor_experiment as ne
import own_retained_history as ret

OUT=ne.ROOT/'own-joint-cohort-source-20261126'
MODELS=('J2-joint128-reg02','J3-joint128-reg005','J5-joint256-reg01','K1-joint256-reg02-d97-mt','PQ2-factor200-frequency')


def main():
    OUT.mkdir(exist_ok=True)
    dataset=Path('data/processed/netflix.bin');data_sha=ne.sha(dataset)
    assert data_sha=='08b417f7601339c95727cd02af4f839b6043c326a5fb0350245a1f6f2054c673'
    ret.map_rows()
    mapping=ret.QUAL/'dataset/dense-to-raw-user.npy'
    manifest=json.loads((mapping.parent/'manifest.json').read_text())
    assert ne.sha(mapping)==manifest['outputs'][mapping.name]
    raw=np.load(mapping);order=np.argsort(raw)
    rid=ne.history('trainx','uidx_to_uid')[ne.history('qual','user_idxs')]
    q=np.zeros(len(rid),dtype=PROBE)
    q['user']=order[np.searchsorted(raw[order],rid)]
    np.testing.assert_array_equal(raw[q['user']],rid)
    q['item']=ne.history('qual','item_idxs')+1;q['day']=ne.history('qual','dates')+314
    path=OUT/'zero-queries.bin';q.tofile(path)
    subprocess.run(['c++','-O3','-std=c++17','-Wno-deprecated-declarations','scripts/own_joint_cohort_source.cpp',
                    '-framework','Accelerate','-o',str(OUT/'score')],check=True)
    argv=[str(OUT/'score'),str(dataset),str(path)]
    sources={}
    for name in MODELS:
        receipt=Path('experiments',name+'.json');r=json.loads(receipt.read_text());cmd=r['command']
        assert cmd[1]=='joint' and cmd[cmd.index('--data')+1]==str(dataset) and '--init-model' not in cmd
        if str(dataset) in r.get('inputs',{}):assert r['inputs'][str(dataset)]['sha256']==data_sha
        sources[name]={'training_receipt':str(receipt),'receipt_sha256':ne.sha(receipt)}
        for k in ('model_artifact','predictions_artifact'):
            info=r[k];p=Path(info['path'])
            if not p.exists():p=Path('../netflix_personas')/p
            assert ne.sha(p)==info['sha256']
            sources[name][k]={'path':str(p),'sha256':info['sha256']}
        argv.extend((sources[name]['model_artifact']['path'],str(OUT/(name+'.f32'))))
    subprocess.run(argv,check=True)
    for name in MODELS:
        p=OUT/(name+'.f32');v=np.memmap(p,dtype='<f4',mode='r')
        orig=np.fromfile(sources[name]['predictions_artifact']['path'],dtype='<f4')
        assert len(v)==1408395+2817131 and np.isfinite(v).all()
        err=float(np.max(np.abs(orig-v[:len(orig)])))
        assert err<.0001,(name,err)
        sources[name].update(vector=str(p),vector_sha256=ne.sha(p),largest_probe_replay_difference=err)
    (OUT/'manifest.json').write_text(json.dumps(dict(training_data_sha256=data_sha,models=sources,
      cohorts=['original Probe','qualifying'],query_outcomes_read=False),indent=2)+'\n')


if __name__=='__main__':main()
