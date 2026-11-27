"""Train a factorization machine with latent user drift, latent movie time bins and date-matched exposures."""
import ctypes
import json
import shutil
import subprocess
import numpy as np
from gbdt_blend import load
import own_anchor_context as anchor
import own_full_context as full
import own_neighbor_experiment as ne
import own_overnight_blend as blend
import own_retained_history as ret
import own_slate_bayes as slate
from own_empirical_neighborhood import fit

OUT=ne.ROOT/'own-mixed-bayes-20261126'
SAMPLES=(24,32,40)


def prepare():
    src=OUT/'src';src.mkdir(exist_ok=True)
    sig=json.loads((slate.OUT/'setup.json').read_text())
    assert ne.sha(slate.OUT/'src/bayes.hpp')==sig['adapted_header_sha256']
    for p in (slate.OUT/'src').iterdir():shutil.copyfile(p,src/p.name)
    subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC','scripts/own_mixed_bayes.cpp','-o',str(OUT/'select.dylib')],check=True)
    lib=ctypes.CDLL(str((OUT/'select.dylib').resolve()));lib.select_exposures.argtypes=[ctypes.c_int]*2+[ctypes.c_void_p]*9
    lib.select_exposures.restype=ctypes.c_int
    d=load(slate.KNOWN);nu=d['users'];nm=d['items']
    bag=slate.OUT/'cohort.bin'
    assert ne.sha(bag)==sig['cohort_sha256']
    with bag.open('rb') as f:
        qo=np.fromfile(f,'<u8',nu+1);qi=np.fromfile(f,'<u2',int(qo[-1]));qd=np.fromfile(f,'<u2',int(qo[-1]));assert not f.read()
    lengths=np.diff(d['implicit_offsets']).astype(float);qcounts=np.diff(qo)
    upper=int(np.minimum(lengths,qcounts+22+1.3*np.sqrt(lengths)).sum())+nu
    off=np.zeros(nu+1,'<u8');i=np.empty(upper,'<u2');day=np.empty(upper,'<u2')
    args=(d['implicit_offsets'],d['implicit_items'],d['implicit_days'],qo,qi,qd,off,i,day)
    assert lib.select_exposures(nu,nm,*[a.ctypes.data for a in args])==0
    path=OUT/'exposures.bin';count=int(off[-1])
    with path.open('wb') as f:
        for a in (off,i[:count],day[:count]):a.tofile(f)
    sig.update(exposure_sha256=ne.sha(path),exposure_count=count)
    p=src/'bayes.hpp';code=p.read_text()
    for old,new in [(f'0x{sig["cohort_sha256"][:16]}ULL',f'0x{sig["exposure_sha256"][:16]}ULL'),
                    (f'd.header.implicit_count = {sig["cohort_addresses"]};',f'd.header.implicit_count = {count};')]:
        assert code.count(old)==1
        code=code.replace(old,new)
    p.write_text(code)
    subprocess.run(['c++','-O3','-DNDEBUG','-std=c++17','-Wno-deprecated-declarations',str(src/'netflix.cpp'),
                    '-framework','Accelerate','-o',str(OUT/'netflix')],check=True)
    sig.update(mixed_header_sha256=ne.sha(p),mixed_binary_sha256=ne.sha(OUT/'netflix'))
    (OUT/'setup.json').write_text(json.dumps(sig,indent=2)+'\n')


def train(mode,ds,rows):
    root=OUT/mode;root.mkdir(exist_ok=True)
    args=[str(OUT/'netflix'),'bayes','--data',str(ds),'--cohort-bag',str(OUT/'exposures.bin'),
        '--model',str(root/'model.bin'),'--predictions',str(root/'model.f32'),'--samples','40',
        '--snapshot-every','8','--checkpoint-every','24','--burn-in','10','--factors','40','--threads','4',
        '--seed','11853','--item-bin','2','--drift','2']
    (root/'command.json').write_text(json.dumps(args)+'\n')
    with (root/'run.log').open('w') as log:subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True)
    return np.column_stack([np.fromfile(root/f'model.s{s:04d}.f32','<f4')[rows] for s in SAMPLES])


def main():
    OUT.mkdir(exist_ok=True);prepare()
    parent=blend.OUT/json.loads((blend.OUT/'queue.json').read_text())['parent']
    base=np.load(parent/'known-selected.npy');y,f,_,_=anchor.history_control();ix=ret.map_rows()
    v=train('known',slate.KNOWN,ix)
    ref=json.loads((ret.QUAL/'B2-fixed128/report.json').read_text());assert ne.sha(ne.ROOT/'retained-neural/B2-full-r64.s0128.f32')==ref['column_sha256']
    b=np.fromfile(ref['column'],'<f4')[ix]
    def design(v,p,b,phase):return np.column_stack((v-p[:,None],v[:,-1]-b,v[:,-1]-full.cf96(phase)))
    x=design(v,base,b,'probex');held=np.empty(len(y))
    for j in range(5):
        tr=f!=j;held[~tr]=x[~tr]@fit(x[tr],(y-base)[tr])
    w=fit(x,y-base)
    orig=blend.metrics(base,y,f);stats=blend.metrics(ne.clipped(base+held),y,f);ok=ret.qualifies(stats,orig)
    rep=dict(known=stats,control=orig,parent=str(parent),weights=w.tolist(),selected='candidate' if ok else 'control',
             qualifying_hashes={},qualifying_labels_used_for_fitting=False,known_comparison_is_nested=False)
    (OUT/'selection.json').write_text(json.dumps(rep,indent=2)+'\n')
    if ok:
        assert ne.sha(parent/'selected.f32')==json.loads((parent/'freeze.json').read_text())['qualifying_hashes']['selected']
        qb=np.fromfile(parent/'selected.f32','<f4').astype(float);b=np.fromfile(ret.OUT/'bayes/model.f32','<f4')
        delta=design(train('qual',slate.QUAL,slice(None)),qb,b,'qual')@w
        np.save(OUT/'oof.npy',held);np.save(OUT/'qual-correction.npy',delta)
        ne.clipped(qb+delta).astype('<f4').tofile(OUT/'selected.f32')
        rep['qualifying_hashes']['selected']=ne.sha(OUT/'selected.f32')
    (OUT/'freeze.json').write_text(json.dumps(rep,indent=2)+'\n')


if __name__=='__main__':main()
