"""Score the original-training-split local Gaussian process on native qualifying identifiers."""
import json
from pathlib import Path
import subprocess

import numpy as np
from gbdt_blend import HEADER, load
import own_anchor_context as anchor
import own_neighbor_experiment as ne
import own_retained_history as ret
import own_overnight_blend as blend

OUT=ne.ROOT/'own-original-gp-20261126'
GPS=('GP-C4-temporal-outputs','GP-j2-outputs','GP-implicit-outputs')
ROOT_DATA=Path('data/processed/netflix.bin')


def target_dataset():
    path=OUT/'queries.bin'
    d=load(ROOT_DATA)
    prod=load(ret.QUAL/'dataset/train-probe-qualifying.bin')
    header=np.fromfile(ROOT_DATA,dtype=HEADER,count=1)
    header[0]['probe_count']=len(prod['probe'])
    assert np.all(prod['probe']['rating']==0)
    with path.open('wb') as stream:
        for x in (header,d['train_offsets'],d['train_items'],d['train_days'],d['train_ratings'],prod['probe'],
                  d['implicit_offsets'],d['implicit_items'],d['implicit_days']):
            x.tofile(stream)
    return path


def main():
    OUT.mkdir(exist_ok=True)
    y,f,_,_=anchor.history_control()
    parent=blend.OUT/'first'
    p=np.load(parent/'known-selected.npy')
    q=np.fromfile(parent/'selected.f32',dtype='<f4').astype(float)
    ix=ret.map_rows()
    known=[]
    recipes=[]
    for key in GPS:
        report=json.loads(Path('experiments',key+'.json').read_text())
        run=json.loads(Path(report['source_manifest']).read_text())
        cmd=run['command']
        assert cmd[0]=='build/moonshot_gp' and cmd[cmd.index('--data')+1]==str(ROOT_DATA)
        model=Path(cmd[cmd.index('--model')+1])
        assert model.name in ('joint128-reg005.bin','joint128-reg02.bin')
        joint='J3-joint128-reg005' if 'reg005' in model.name else 'J2-joint128-reg02'
        m=json.loads(Path('experiments',joint+'.json').read_text())
        assert m['command'][1]=='joint' and '--init-model' not in m['command']
        assert ne.sha(model)==m['model_artifact']['sha256']==run['model_artifact']['sha256']
        assert ne.sha(ROOT_DATA)==run['inputs'][str(ROOT_DATA)]['sha256']
        paths=list(report['outputs'])[:2]
        for path in paths:
            assert ne.sha(path)==report['outputs'][path]['sha256']
            known.append(np.fromfile(path,dtype='<f4')[ix])
        recipes.append((key,cmd,paths))
    known=np.column_stack(known)
    np.save(OUT/'probex.npy',known)
    pred,stats=ret.evaluate(known,y,f,p)
    control=blend.metrics(p,y,f)
    ok=ret.qualifies(stats,control)
    report=dict(selected='candidate' if ok else 'control',control=control,candidate=stats,qualifying_hashes={},
                known_comparison_is_nested=False,qualifying_labels_used_for_fitting=False,
                both_prediction_partitions_use_original_training=True)
    print(json.dumps(report),flush=True)
    (OUT/'selection.json').write_text(json.dumps(report,indent=2)+'\n')
    if ok:
        dataset=target_dataset()
        # Use the same dataset-equivalence check as the upstream eval-joint command.
        src=Path('src/moonshot_gp.cpp').read_text()
        old='load_joint(value_after(args, "--model"), d, t, joint_fingerprint(d))'
        assert src.count(old)==1
        patched=OUT/'original-gp.cpp'
        patched.write_text(src.replace(old,old.replace('joint_fingerprint(d)','bound_fingerprint(args, d)')))
        binary=OUT/'original-gp'
        subprocess.run(['c++','-O3','-std=c++17','-Wno-deprecated-declarations','-Isrc',str(patched),'-framework','Accelerate','-o',str(binary)],check=True)
        qual=[]
        for name,cmd,paths in recipes:
            argv=cmd.copy()
            argv[0]=str(binary)
            argv[argv.index('--data')+1]=str(dataset)
            argv[argv.index('--output')+1]=str(OUT/name)
            argv+=['--fingerprint-data',str(ROOT_DATA)]
            with (OUT/(name+'.log')).open('w') as log:subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,check=True)
            for j in range(len(paths)):
                z=np.fromfile(OUT/(name+f'.{j}.f32'),dtype='<f4')
                assert z.shape==q.shape and np.isfinite(z).all()
                qual.append(z)
        qual=np.column_stack(qual)
        np.save(OUT/'qual.npy',qual)
        delta=(qual-q[:,None])@np.array(stats['weights'])
        np.save(OUT/'oof.npy',pred-p)
        np.save(OUT/'qual-correction.npy',delta)
        path=OUT/'selected.f32'
        ne.clipped(q+delta).astype('<f4').tofile(path)
        report['qualifying_hashes']['selected']=ne.sha(path)
    (OUT/'freeze.json').write_text(json.dumps(report,indent=2)+'\n')
    if ok:ne.audit(OUT,ne.LABELS)


if __name__=='__main__':main()
