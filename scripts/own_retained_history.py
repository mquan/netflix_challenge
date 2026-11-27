"""Transfer five independently trained local history predictors and screen Bayesian timeSVD++."""
import json
from pathlib import Path
import subprocess

import numpy as np
import own_neighbor_experiment as ne
import own_anchor_context as anchor
from own_empirical_neighborhood import fit

OUT=ne.ROOT/'own-retained-history-20261126'
MODELS=('A2h-snapshot-4-8','MT1-fixed10','MT3-fixed7','MT8-fixed7','MH3-fixed4')
QUAL=ne.ROOT/'qualifying'


def map_rows():
    u,i,d,_=anchor.old_probe()
    key=u.astype('i8')*32768+i
    order=np.argsort(key)
    qu,qi,qd=[ne.history('probex',k) for k in ('user_idxs','item_idxs','dates')]
    qkey=qu.astype('i8')*32768+qi
    ix=order[np.searchsorted(key[order],qkey)]
    np.testing.assert_array_equal(key[ix],qkey)
    np.testing.assert_array_equal(d[ix],qd)
    info=json.loads((QUAL/'dataset/manifest.json').read_text())
    path=QUAL/'dataset/qualifying-order.npy'
    assert ne.sha(path)==info['outputs'][path.name]
    original=np.load(path,mmap_mode='r')
    uid=ne.history('trainx','uidx_to_uid')
    native=ne.history('qual','user_idxs')
    np.testing.assert_array_equal(original['raw_user'],uid[native])
    np.testing.assert_array_equal(original['item']-1,ne.history('qual','item_idxs'))
    np.testing.assert_array_equal(original['day'].astype('i')-314,ne.history('qual','dates'))
    return ix


def read_pairs():
    ix=map_rows()
    known,qual=[],[]
    for model in MODELS:
        r=json.loads((QUAL/model/'report.json').read_text())
        assert r['state']=='complete' and r['training_requested']
        assert ne.sha(QUAL/'dataset/manifest.json')==r['dataset_manifest_sha256']
        kp=Path(r['column'])
        assert kp.parent.resolve()==(ne.ROOT/'retained-neural').resolve() and ne.sha(kp)==r['column_sha256']
        qp=QUAL/model/'model.f32'
        assert ne.sha(qp)==r['predictions_sha256']
        if model==MODELS[0]:
            a=json.loads(Path(str(kp)+'.json').read_text())
            assert a['epochs']==[4,5,6,7,8] and a['config']['objective']=='masked'
            assert a['data_sha256']=='08b417f7601339c95727cd02af4f839b6043c326a5fb0350245a1f6f2054c673'
            assert r['snapshot_epochs']==a['epochs']
        else:
            assert r['config']['objective']=='masked' and 'masked_history.py' in r['sources']
        known.append(np.fromfile(kp,dtype='<f4')[ix])
        qual.append(np.fromfile(qp,dtype='<f4'))
    rec=json.loads(Path('experiments/QUAL-B2-fixed128-recipe-20261116.json').read_text())
    bp=Path(rec['column'])
    assert ne.sha(bp)==rec['verified_files'][str(bp.relative_to(Path.cwd()))]
    b=np.fromfile(bp,dtype='<f4')[ix]
    return np.column_stack(known),np.column_stack(qual),b


def evaluate(x,y,f,p):
    z=x.astype(float)-p[:,None]
    held=p.copy()
    ws=[]
    for j in range(5):
        tr=f!=j
        w=fit(z[tr],(y-p)[tr])
        held[~tr]=ne.clipped(p[~tr]+z[~tr]@w)
        ws.append(w.tolist())
    w=fit(z,y-p)
    return held,dict(pooled=ne.rmse(held,y),folds=[ne.rmse(held[f==j],y[f==j]) for j in range(5)],
                     weights=w.tolist(),fold_weights=ws)


def qualifies(candidate,base):
    return candidate['pooled']<base['pooled'] and sum(a<b for a,b in zip(candidate['folds'],base['folds']))>=4


def main():
    OUT.mkdir(exist_ok=True)
    y,f,p,q=anchor.history_control()
    known,qual,b=read_pairs()
    np.save(OUT/'known.npy',known)
    np.save(OUT/'qual.npy',qual)
    base=dict(pooled=ne.rmse(p,y),folds=[ne.rmse(p[f==i],y[f==i]) for i in range(5)])
    held,a=evaluate(known,y,f,p)
    held_b,ab=evaluate(np.column_stack((known,b)),y,f,p)
    take=qualifies(a,base)
    add_b=qualifies(ab,a if take else base)
    report=dict(control=base,history=a,bayes_and_history=ab,selected='history' if take else 'control',
                bayes_eligible=add_b,qualifying_hashes={},known_comparison_is_nested=False,
                qualifying_labels_used_for_fitting=False,models=MODELS)
    (OUT/'selection.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    if take:
        np.save(OUT/'oof.npy',held-p)
        delta=(qual.astype(float)-q[:,None])@np.array(a['weights'])
        np.save(OUT/'qual-correction.npy',delta)
        path=OUT/'selected.f32'
        ne.clipped(q+delta).astype('<f4').tofile(path)
        report['qualifying_hashes']['selected']=ne.sha(path)
    (OUT/'freeze.json').write_text(json.dumps(report,indent=2)+'\n')
    if take: ne.audit(OUT,ne.LABELS)
    if add_b:
        out=OUT/'bayes'
        out.mkdir()
        rec=json.loads((QUAL/'B2-fixed128/report.json').read_text())
        binary=Path(rec['command'][0])
        assert ne.sha(binary)==rec['binary_sha256']
        assert ne.sha(QUAL/'dataset/train-probe-qualifying.bin')==rec['data_sha256']
        argv=rec['command'].copy()
        argv[argv.index('--model')+1]=str(out/'model.bin')
        argv[argv.index('--predictions')+1]=str(out/'model.f32')
        argv+=['--init-model',str(QUAL/'B2-fixed128/model.bin')]
        with (out/'run.log').open('w') as log:
            subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,check=True)
        qb=np.fromfile(out/'model.f32',dtype='<f4')
        assert qb.shape==q.shape and np.isfinite(qb).all()
        qq=np.column_stack((qual,qb))
        delta=(qq-q[:,None])@np.array(ab['weights'])
        np.save(out/'oof.npy',held_b-p)
        np.save(out/'qual-correction.npy',delta)
        path=out/'selected.f32'
        ne.clipped(q+delta).astype('<f4').tofile(path)
        frozen=report|dict(selected='bayes_and_history',qualifying_hashes={'selected':ne.sha(path)},command=argv)
        (out/'freeze.json').write_text(json.dumps(frozen,indent=2)+'\n')
        ne.audit(out,ne.LABELS)


if __name__=='__main__':main()
