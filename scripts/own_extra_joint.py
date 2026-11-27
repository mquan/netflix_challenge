"""Replay three original-training factor checkpoints, including two verified J2 continuations."""
import json
from pathlib import Path
import subprocess
import numpy as np
import own_joint_cohort_source as joint
import own_neighbor_experiment as ne
OUT=ne.ROOT/'own-extra-joint-20261126'


def main():
    OUT.mkdir(exist_ok=True);original=json.loads((joint.OUT/'manifest.json').read_text())
    baseline=original['models']['J2-joint128-reg02']['model_artifact']
    assert ne.sha(baseline['path'])==baseline['sha256']
    data=Path('data/processed/netflix.bin');assert ne.sha(data)==original['training_data_sha256']
    inputs={}
    args=[str(joint.OUT/'score'),str(data),str(joint.OUT/'zero-queries.bin')]
    for name in ('J1-joint64','J6-frequency-factors128','J7-continuation-control128'):
        p=Path('experiments',name+'.json');r=json.loads(p.read_text());cmd=r['command']
        assert cmd[1]=='joint' and cmd[cmd.index('--data')+1]==str(data)
        if '--init-model' in cmd:assert cmd[cmd.index('--init-model')+1]==baseline['path']
        assert not set(cmd)&{'--base','--inputs','--predictions-in'}
        model,pred=[r[k] for k in ('model_artifact','predictions_artifact')]
        assert all(ne.sha(v['path'])==v['sha256'] for v in (model,pred))
        path=OUT/(name+'.f32');args.extend((model['path'],str(path)))
        inputs[name]=dict(receipt=str(p),receipt_sha256=ne.sha(p),model=model,predictions=pred,vector=str(path))
    subprocess.run(args,check=True)
    for m in inputs.values():
        v=np.memmap(m['vector'],'<f4','r');orig=np.fromfile(m['predictions']['path'],'<f4')
        assert v.shape==(4225526,) and np.isfinite(v).all()
        err=float(np.max(abs(v[:len(orig)]-orig)));assert err<.0002,err
        m.update(vector_sha256=ne.sha(m['vector']),original_replay_max_difference=err)
    (OUT/'manifest.json').write_text(json.dumps(dict(inputs=inputs,original_dataset_sha256=original['training_data_sha256'],
        initialization=baseline,qualifying_labels_read=False),indent=2)+'\n')


if __name__=='__main__':main()
