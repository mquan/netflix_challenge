"""Bayesian factor residuals around exact target-removed ordinal CFNADE forecasts."""
import ctypes
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
from gbdt_blend import load
import own_anchor_context as anchor
import own_cf_context as cf
import own_full_context as full
import own_neighbor_experiment as ne
import own_overnight_blend as blend
import own_retained_history as ret
import own_slate_bayes as slate
from own_empirical_neighborhood import fit

OUT=ne.ROOT/'own-residual-bayes-20261126'


def offsets(train,phase,dataset):
    mode='known' if phase=='probex' else 'qual'
    root=OUT/mode
    root.mkdir(exist_ok=True)
    d=load(dataset)
    uc,ic=[ne.history(train,k) for k in ('user_cnts','item_cnts')]
    i,day,r=[ne.history(train,k) for k in ('item_idxs','dates','ratings')]
    off=np.r_[0,uc.cumsum()].astype('i')
    rawfile=ret.QUAL/'dataset/dense-to-raw-user.npy'
    receipt=json.loads((rawfile.parent/'manifest.json').read_text())
    assert ne.sha(rawfile)==receipt['outputs'][rawfile.name]
    raw=np.load(rawfile);canonical=ne.history(train,'uidx_to_uid')
    mapping=np.searchsorted(canonical,raw).astype('i')
    np.testing.assert_array_equal(canonical[mapping],raw)
    lib=ctypes.CDLL(str((OUT/'offset.dylib').resolve()))
    it,pt,ft=ctypes.c_int,ctypes.c_void_p,ctypes.c_float
    lib.verify_rows.argtypes=[it,it]+[pt]*9;lib.verify_rows.restype=it
    lib.cf_totals.argtypes=[it,it]+[pt]*7;lib.cf_totals.restype=None
    lib.predict_rows.argtypes=[it,ft,ft]+[pt]*14;lib.predict_rows.restype=None
    def ptr(*xs):return [x.ctypes.data if x is not None else None for x in xs]
    assert lib.verify_rows(len(uc),len(ic),*ptr(mapping,d['train_offsets'],d['train_items'],d['train_days'],
                        d['train_ratings'],off,i,day,r))==0
    s=cf.checkpoint(train);a=np.ascontiguousarray(s['a'].cumsum(0,dtype=float),'f')
    tot=np.empty((len(uc),96),'f');md=np.empty(len(uc),'f')
    lib.cf_totals(len(uc),len(ic),*ptr(off,i,day,r,a,tot,md))
    avg=day.mean(dtype=float);md[uc==0]=avg
    sd=(sum(np.sum((day[j:j+1000000].astype(float)-avg)**2) for j in range(0,len(day),1000000))/len(day))**.5
    tot=np.ascontiguousarray(tot@s['enc_proj'].T)
    ep=np.ascontiguousarray((a@s['enc_proj'].T).transpose(1,0,2))
    op=np.ascontiguousarray((s['p'].cumsum(0,dtype=float).astype('f')@s['dec_proj']).transpose(1,0,2))
    b,ub,ud=[np.ascontiguousarray(s[k].cumsum(0,dtype=float).T,'f') for k in ('b','user_bias','user_drift')]
    common=ptr(uc,md,tot,ep,op,s['enc_bias'],b,ub,ud)
    def pred(us,mi,dd,rr,out):
        lib.predict_rows(len(us),sd,avg,*ptr(us,mi,dd,rr),*common,out.ctypes.data)
    q=d['probe'];v=np.empty(len(q),'f')
    for j in range(0,len(q),16384):
        sl=slice(j,j+16384)
        pred(mapping[q['user'][sl]],np.ascontiguousarray(q['item'][sl]),np.ascontiguousarray(q['day'][sl]),None,v[sl])
    rows=ret.map_rows() if mode=='known' else slice(None)
    err=float(np.max(abs(v[rows]-full.cf96(phase))))
    assert err<.0003,(mode,err)
    v.tofile(root/'offset-query.f32')
    nn=len(d['train_ratings']);target=np.memmap(root/'offset-train.f32',mode='w+',shape=(nn,),dtype='<f4')
    def chunk(start):
        end=min(start+192,len(uc))
        bounds=d['train_offsets'][start:end+1].astype('i8')
        sl=slice(bounds[0],bounds[-1])
        us=np.repeat(mapping[start:end],np.diff(bounds))
        pred(us,d['train_items'][sl],d['train_days'][sl],d['train_ratings'][sl],target[sl])
    started=time.monotonic()
    with ThreadPoolExecutor(2) as pool:
        for j,_ in enumerate(pool.map(chunk,range(0,len(uc),192))):
            if j%500==0:print(mode,j,round(time.monotonic()-started),flush=True)
    target.flush()
    assert np.isfinite(target).all() and target.min()>=.99999 and target.max()<=5.00001
    info=dict(train_rows=nn,native_query_max_difference=err,training_coordinates_and_stars_verified=True,
              baseline_multiplier=.85,cf_train=train,cf_query=phase)
    (root/'offsets.json').write_text(json.dumps(info,indent=2)+'\n')


def build():
    subprocess.run(['c++','-O3','-std=c++17','-Wno-deprecated-declarations','-shared','-fPIC',
           'scripts/own_residual_bayes.cpp','scripts/own_cf_adapter.cpp','-framework','Accelerate','-o',str(OUT/'offset.dylib')],check=True)
    src=OUT/'src';src.mkdir(exist_ok=True)
    sig=json.loads((slate.OUT/'setup.json').read_text())
    assert sig['adapted_header_sha256']==ne.sha(slate.OUT/'src/bayes.hpp')
    for p in (slate.OUT/'src').iterdir():shutil.copyfile(p,src/p.name)
    p=src/'bayes.hpp';code=p.read_text()
    patches={
        'std::vector<double> probe_pred;':'std::vector<double> probe_pred; std::vector<float> train_offset, query_offset;',
        'if (fac < 0) probe_pred[n] = m.w0+qr+bu.q+bi.q;':
            'if (fac < 0) probe_pred[n] = .85*(query_offset[n]-3.6)+m.w0+qr+bu.q+bi.q;',
        'double(d.train_ratings[p])-(m.w0+qr+bu.q+bi.q)':
            'double(d.train_ratings[p])-(.85*(train_offset[p]-3.6)+m.w0+qr+bu.q+bi.q)',
        'Sampler s(d, t, feats, m, threads);': '''Sampler s(d, t, feats, m, threads);
      std::ifstream off_train(value_after(args,"--offset-train"),std::ios::binary);
      std::ifstream off_query(value_after(args,"--offset-query"),std::ios::binary);
      read_vector(off_train,s.train_offset,d.header.train_count);
      read_vector(off_query,s.query_offset,d.probe.size());
      if(off_train.peek()!=EOF || off_query.peek()!=EOF) fail("offset vector mismatch");''',
        'joint_fingerprint(d) ^':'joint_fingerprint(d) ^ 0xe22e416cb8299517ULL ^',
    }
    for a,b in patches.items():
        assert code.count(a)==1
        code=code.replace(a,b)
    p.write_text(code)
    subprocess.run(['c++','-O3','-DNDEBUG','-std=c++17','-Wno-deprecated-declarations',str(src/'netflix.cpp'),
                    '-framework','Accelerate','-o',str(OUT/'netflix')],check=True)


def train(train,phase,dataset):
    mode='known' if phase=='probex' else 'qual'
    root=OUT/mode
    if not (root/'offsets.json').exists():offsets(train,phase,dataset)
    args=[str(OUT/'netflix'),'bayes','--data',str(dataset),'--cohort-bag',str(slate.OUT/'cohort.bin'),
          '--offset-train',str(root/'offset-train.f32'),'--offset-query',str(root/'offset-query.f32'),
          '--model',str(root/'model.bin'),'--predictions',str(root/'model.f32'),'--samples','64',
          '--snapshot-every','16','--checkpoint-every','32','--burn-in','12','--factors','32','--threads','4','--seed','5883']
    (root/'command.json').write_text(json.dumps(args)+'\n')
    with (root/'run.log').open('w') as stream:subprocess.run(args,check=True,stdout=stream,stderr=subprocess.STDOUT)
    rows=ret.map_rows() if mode=='known' else slice(None)
    return np.column_stack([np.fromfile(root/f'model.s{s:04d}.f32','<f4')[rows] for s in (32,48,64)])


def design(v,base,phase):
    return np.column_stack((v-base[:,None],v[:,-1]-full.cf96(phase)))


def main():
    OUT.mkdir(exist_ok=True);build()
    pdir=blend.OUT/json.loads((blend.OUT/'queue.json').read_text())['parent']
    base=np.load(pdir/'known-selected.npy')
    y,f,_,_=anchor.history_control()
    x=design(train('trainx','probex',slate.KNOWN),base,'probex')
    held=np.empty(len(y))
    for j in range(5):
        tr=f!=j
        held[~tr]=x[~tr]@fit(x[tr],(y-base)[tr])
    w=fit(x,y-base);ctrl=blend.metrics(base,y,f);known=blend.metrics(ne.clipped(base+held),y,f)
    take=ret.qualifies(known,ctrl)
    rep=dict(parent=str(pdir),control=ctrl,known=known,weights=w.tolist(),selected='candidate' if take else 'control',
             qualifying_hashes={},qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(rep,indent=2)+'\n')
    if take:
        assert ne.sha(pdir/'selected.f32')==json.loads((pdir/'freeze.json').read_text())['qualifying_hashes']['selected']
        qb=np.fromfile(pdir/'selected.f32','<f4').astype(float)
        delta=design(train('fulltrain','qual',slate.QUAL),qb,'qual')@w
        np.save(OUT/'oof.npy',held);np.save(OUT/'qual-correction.npy',delta)
        ne.clipped(qb+delta).astype('<f4').tofile(OUT/'selected.f32')
        rep['qualifying_hashes']['selected']=ne.sha(OUT/'selected.f32')
    (OUT/'freeze.json').write_text(json.dumps(rep,indent=2)+'\n')


if __name__=='__main__':main()
