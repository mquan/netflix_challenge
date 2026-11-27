"""Screen independent chronological base-model outputs on the schema-corrected development parent."""
import argparse
import ctypes
import json
from pathlib import Path
from time import sleep
import numpy as np
import torch
import own_anchor_context as anchor
import own_full_context as full
import own_neighbor_experiment as ne
import own_overnight_blend as blend
import own_retained_history as ret
import own_suffix_cf as cf
import own_suffix_factors as factors
import own_suffix_impute as impute
PARENT=blend.OUT/'schema-03'
LIB={'cf':cf,'impute':impute,'factors':factors}


def design(kind,v,train,phase,base):
    check=full.cf96(phase)
    if kind=='factors':signals=v
    else:
        np.testing.assert_allclose(v[:,6],check,rtol=0,atol=.0004)
        signals=np.column_stack((v[:,5]-check,v[:,7:]-check[:,None],v[:,5]-base))
    u,m=[ne.history(phase,k) for k in ('user_idxs','item_idxs')]
    uc,ic=[ne.history(train,k) for k in ('user_cnts','item_cnts')]
    g=np.column_stack((np.tanh((np.log1p(uc[u])-5)/1.3),np.tanh((np.log1p(ic[m])-8)/1.8),
                         np.tanh((base-3.5)/.8),np.tanh((check-base)*4)))
    tiny=signals.clip(-.65,.65)
    return np.column_stack((signals,tiny,signals*(abs(tiny)-.1),
                (.55*tiny[:,:,None]*g[:,None,:]).reshape(len(base),-1))).astype('f')


def main(kind):
    module=LIB[kind];src=module.OUT;out=ne.ROOT/('own-schema-suffix-'+kind+'-20261126');out.mkdir(exist_ok=True)
    while not (src/'selection.json').is_file():sleep(15)
    y,f,_,_=anchor.history_control();fr=json.loads((PARENT/'freeze.json').read_text());assert ne.sha(PARENT/'selected.f32')==fr['qualifying_hashes']['selected']
    base=np.load(PARENT/'known-selected.npy');qb=np.fromfile(PARENT/'selected.f32','<f4')
    path=src/'probex.npy';v=np.load(path);x=design(kind,v,'trainx','probex',base)
    def fit(a,y):
        g=a.T@a;g.flat[::len(g)+1]+=360;return np.linalg.solve(g,a.T@y)
    held=np.empty(len(y))
    for k in range(5):
        tr=f!=k;held[~tr]=x[~tr]@fit(x[tr],(y-base)[tr])
    w=fit(x,y-base)
    ctrl=blend.metrics(base,y,f);scores=blend.metrics(ne.clipped(base+held),y,f)
    ok=ret.qualifies(scores,ctrl);rep=dict(parent=str(PARENT),selected='candidate' if ok else 'control',known=scores,control=ctrl,
        known_raw_model_only={'path':str(path),'sha256':ne.sha(path)},source_implementation_sha256=ne.sha(module.__file__),
        qualifying_hashes={},qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (out/'selection.json').write_text(json.dumps(rep,indent=2)+'\n')
    if ok:
        while not (src/'freeze.json').exists():sleep(15)
        qp=src/'qual.npy'
        if qp.is_file():qv=np.load(qp)
        else:
            assert kind!='factors'
            module.OUT=out/'raw';module.OUT.mkdir(exist_ok=True);torch.set_num_threads(2)
            lib=ctypes.CDLL(str((cf.OUT/'suffix.dylib').resolve())) if kind!='cf' else ctypes.CDLL(str((src/'suffix.dylib').resolve()))
            lib.suffix_cf.argtypes=[ctypes.c_int]*4+[ctypes.c_void_p]*18;lib.suffix_cf.restype=None
            qv=module.run('fulltrain','qual',lib);qp=module.OUT/'qual.npy'
        rep['qualifying_raw_model_only']={'path':str(qp),'sha256':ne.sha(qp)}
        delta=design(kind,qv,'fulltrain','qual',qb)@w
        np.save(out/'oof.npy',held);np.save(out/'qual-correction.npy',delta)
        ne.clipped(qb+delta).astype('<f4').tofile(out/'selected.f32');rep['qualifying_hashes']['selected']=ne.sha(out/'selected.f32')
    (out/'freeze.json').write_text(json.dumps(rep,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('kind',choices=LIB);main(parser.parse_args().kind)
