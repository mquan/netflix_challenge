"""Cross-fit a residual model on the user's rating history and the frozen neighborhood model."""
import ctypes
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess

import numpy as np
from catboost import CatBoostRegressor
import own_neighbor_experiment as ne

OUT = ne.ROOT / 'own-history-context-20261126'
FOLDS = 5
ITERATIONS = 1200


def neighbor_control():
    y, folds, base, q = ne.baseline()
    report = json.loads((ne.OUT/'freeze.json').read_text())
    col = int(report['selected'])
    x = np.load(ne.OUT/'known.npy')[:,col].astype(float)
    oof = base.copy()
    for f in range(FOLDS):
        tr = folds != f
        w = np.clip(x[tr] @ (y[tr]-base[tr]) / (x[tr] @ x[tr]),0,2)
        oof[~tr] = ne.clipped(base[~tr] + w*x[~tr])
    np.testing.assert_allclose(ne.rmse(oof,y),report['known'][str(col)]['pooled'],atol=1e-12,rtol=0)
    path = ne.OUT/'selected.f32'
    assert ne.sha(path) == report['qualifying_hashes']['selected']
    return y, folds, oof, np.fromfile(path,dtype='<f4').astype(float)


def features(train, query, base):
    v = np.array(ne.verified_rank(train,True))
    v /= np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-12)
    similarities = v @ v.T
    users = ne.history(query,'user_idxs')
    items = ne.history(query,'item_idxs')
    dates = ne.history(query,'dates')
    ucounts = ne.history(train,'user_cnts')
    counts = ne.history(train,'item_cnts')
    starts = np.r_[0, ucounts.cumsum()].astype('i')
    ratings = ne.history(train,'ratings')
    residuals = ratings - ne.verified_rank(train)
    hitems, hdates = ne.history(train,'item_idxs'), ne.history(train,'dates')
    ctx = np.empty((len(users),54),'f')
    lib = ctypes.CDLL(str((OUT/'features.dylib').resolve()))
    func = lib.user_context
    pt, it = ctypes.c_void_p, ctypes.c_int
    func.argtypes = [it]+[pt]*9+[it,pt]
    func.restype = None
    def chunk(bounds):
        offset, end = bounds
        sl = slice(offset, end)
        args = (users[sl],items[sl],dates[sl],starts,hitems,hdates,ratings,residuals,similarities)
        func(end-offset,*[a.ctypes.data for a in args],len(counts),ctx[sl].ctypes.data)
    blocks = [(i,min(len(users),i+8192)) for i in range(0,len(users),8192)]
    with ThreadPoolExecutor(4) as executor:
        list(executor.map(chunk,blocks))
    parent = ne.verified_rank(query)
    signal = np.load(ne.OUT/('known.npy' if query=='probex' else 'qual.npy'))
    x = np.column_stack((base, parent-base, dates, np.log1p(counts[items]),
                         ne.history(train,'item_years')[items], signal, ctx)).astype('f')
    assert np.isfinite(x).all()
    np.save(OUT/(query+'.npy'), x)
    return x


def model():
    return CatBoostRegressor(iterations=ITERATIONS,depth=6,learning_rate=.04,
        l2_leaf_reg=25,random_seed=20261126,thread_count=4,allow_writing_files=False,
        verbose=False)


def main():
    OUT.mkdir(exist_ok=True)
    subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC','scripts/own_neighbor_features.cpp','-o',str(OUT/'features.dylib')],check=True)
    y, folds, p, q = neighbor_control()
    x = features('trainx','probex',p)
    residual = y-p
    held = np.empty(len(y))
    for f in range(FOLDS):
        tr = folds!=f
        m = model().fit(x[tr],residual[tr])
        held[~tr] = m.predict(x[~tr],thread_count=4)
        print('fold',f,ne.rmse(ne.clipped(p[~tr]+held[~tr]),y[~tr]),flush=True)
        np.save(OUT/f'fold{f}.npy',held[~tr])
    np.save(OUT/'oof.npy',held)
    # Fixed shrinkages evaluated with the same user-fold gate as the control.
    def metrics(pred):
        return dict(pooled=ne.rmse(pred,y),folds=[ne.rmse(pred[folds==f],y[folds==f]) for f in range(FOLDS)])
    stats = {'control':metrics(p)}
    for w in (.25,.5,.75,1.):
        stats[str(w)] = metrics(ne.clipped(p+w*held))
    eligible = [k for k in stats if k!='control' and stats[k]['pooled']<stats['control']['pooled']
                and sum(a<b for a,b in zip(stats[k]['folds'],stats['control']['folds']))>=4]
    selected=min(eligible,key=lambda k:stats[k]['pooled']) if eligible else 'control'
    report = dict(selected=selected,known=stats,known_comparison_is_nested=False,qualifying_hashes={},
                  qualifying_labels_used_for_fitting=False,baseline_sha256=ne.sha(ne.OUT/'selected.f32'))
    (OUT/'selection.json').write_text(json.dumps(report,indent=2)+'\n')
    if selected!='control':
        qx = features('fulltrain','qual',q)
        m = model().fit(x,residual)
        delta=m.predict(qx,thread_count=4)
        np.save(OUT/'qual-correction.npy',delta)
        pred=OUT/'selected.f32'
        ne.clipped(q+float(selected)*delta).astype('<f4').tofile(pred)
        report['qualifying_hashes']['selected']=ne.sha(pred)
    (OUT/'freeze.json').write_text(json.dumps(report,indent=2)+'\n')
    if report['qualifying_hashes']:
        ne.audit(OUT,ne.LABELS)


if __name__=='__main__':
    main()
