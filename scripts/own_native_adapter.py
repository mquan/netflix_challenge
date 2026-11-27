"""Adapt both sides of locally trained CF96 through its native ordinal decoder using target-removed chronological users."""
import ctypes
import json
import subprocess
import numpy as np
import own_anchor_context as anchor
import own_cf_context as cf
import own_neighbor_experiment as ne
import own_overnight_blend as blend
import own_retained_history as ret
import own_suffix_cf as suffix
from own_empirical_neighborhood import fit
OUT=ne.ROOT/'own-native-adapter-20261126'
PARENT=blend.OUT/'schema-08'


def run(train,phase,lib):
    data=suffix.Data(train,phase,None);s=cf.checkpoint(train)
    a=data.enc.reshape(-1,96);vec=np.linalg.eigh(a.T@a/len(a))[1][:,-16:]
    inp=np.ascontiguousarray(s['enc_proj']@vec,'f')
    # The same fixed output subspace is used for reading hidden states and differentiating through the decoder.
    _,_,v=np.linalg.svd(s['dec_proj'],full_matrices=False);read=np.ascontiguousarray(v[:16],'f')
    out=np.full((4,len(data.native)),np.nan,'f')
    args=(data.off,data.i,data.d,data.r,data.boundary,data.grid,data.fi,data.fd,data.obs,data.lookup,data.enc,data.eproj,data.eb,
          data.op,data.b,data.ub,data.ud,inp,read,out)
    lib.native_adapter(data.nu,len(data.geo),len(data.native),data.sd,*[v.ctypes.data for v in args])
    assert np.isfinite(out).all()
    u=ne.history(phase,'user_idxs');valid=ne.history(train,'user_cnts')[u]>0
    err=float(np.max(abs(out[3,valid]-data.native[valid])));assert err<.0004,err
    np.save(OUT/(phase+'.npy'),out[:3].T)
    return out[:3].T,err


def main():
    OUT.mkdir(exist_ok=True)
    subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC','-Wno-deprecated-declarations','scripts/own_native_adapter.cpp',
                    '-framework','Accelerate','-o',str(OUT/'adapter.dylib')],check=True)
    lib=ctypes.CDLL(str((OUT/'adapter.dylib').resolve()));lib.native_adapter.argtypes=[ctypes.c_int]*3+[ctypes.c_float]+[ctypes.c_void_p]*20
    lib.native_adapter.restype=None
    y,f,_,_=anchor.history_control();r=json.loads((PARENT/'freeze.json').read_text())
    assert ne.sha(PARENT/'selected.f32')==r['qualifying_hashes']['selected'];b=np.load(PARENT/'known-selected.npy');qb=np.fromfile(PARENT/'selected.f32','<f4')
    v,err=run('trainx','probex',lib);held=np.empty(len(y))
    for k in range(5):
        tr=f!=k;held[~tr]=v[~tr]@fit(v[tr],(y-b)[tr])
    w=fit(v,y-b);ctrl=blend.metrics(b,y,f);known=blend.metrics(ne.clipped(b+held),y,f);ok=ret.qualifies(known,ctrl)
    rep=dict(parent=str(PARENT),selected='candidate' if ok else 'control',known=known,control=ctrl,weights=w.tolist(),native_replay={'probex':err},
             qualifying_hashes={},qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(rep,indent=2)+'\n')
    if ok:
        qv,err=run('fulltrain','qual',lib);rep['native_replay']['qual']=err
        delta=qv@w;np.save(OUT/'oof.npy',held);np.save(OUT/'qual-correction.npy',delta)
        ne.clipped(qb+delta).astype('<f4').tofile(OUT/'selected.f32');rep['qualifying_hashes']['selected']=ne.sha(OUT/'selected.f32')
    (OUT/'freeze.json').write_text(json.dumps(rep,indent=2)+'\n')


if __name__=='__main__':main()
