"""Train a Bayesian factorization machine with implicit bags restricted to competition addresses."""
import json
import shutil
import subprocess
from pathlib import Path
import numpy as np
from gbdt_blend import load, PROBE
import own_joint_cohort_source as joint
import own_neighbor_experiment as ne
import own_anchor_context as anchor
import own_retained_history as ret
import own_overnight_blend as blend
from own_empirical_neighborhood import fit

OUT = ne.ROOT/'own-slate-bayes-20261126'
ARCHIVE = ret.QUAL/'bayes-source-0c1f29d'
KNOWN = Path('data/processed/netflix.bin')
QUAL = ret.QUAL/'dataset/train-probe-qualifying.bin'
SWEEPS = (32,48,64)


def prepare():
    manifest = json.loads((ARCHIVE/'source-manifest.json').read_text())
    receipt = json.loads((ret.QUAL/'B2-fixed128/report.json').read_text())
    assert ne.sha(ARCHIVE/'source-manifest.json') == receipt['source_manifest_sha256']
    src = OUT/'src'
    src.mkdir(exist_ok=True)
    for name, digest in manifest['original_sources'].items():
        if name == 'src/bayes.hpp':
            digest = manifest['adapted_bayes_sha256']
        assert ne.sha(ARCHIVE/name) == digest
        shutil.copyfile(ARCHIVE/name, OUT/name)
    data = load(KNOWN)
    q = np.fromfile(joint.OUT/'zero-queries.bin', dtype=PROBE)
    assert len(q) == 2817131 and not np.any(q['rating'])
    qual = load(QUAL)
    for k in ('user','item','day'):
        np.testing.assert_array_equal(q[k], qual['probe'][k])
    cols = [np.concatenate((data['probe'][k], q[k])) for k in ('user','item','day')]
    user, movie, day = cols
    order = np.lexsort((movie,day,user))
    offsets = np.r_[0,np.bincount(user,minlength=data['users']).cumsum()].astype('<u8')
    n = int(offsets[-1])
    assert n == 4225526 and movie.min() >= 1 and movie.max() <= data['items']
    bag = OUT/'cohort.bin'
    with bag.open('wb') as file:
        offsets.tofile(file)
        movie[order].astype('<u2').tofile(file)
        day[order].astype('<u2').tofile(file)
    file = src/'bayes.hpp'
    code = file.read_text()
    before = 'const Dataset d = load_dataset(value_after(args, "--data"));'
    mark = 'const uint64_t fingerprint = joint_fingerprint(d);'
    assert code.count(before) == code.count(mark) == 1
    fingerprint = ne.sha(bag)
    patch = f"""
  const uint64_t fingerprint = joint_fingerprint(d) ^ 0x{fingerprint[:16]}ULL;
  std::ifstream bag(value_after(args, "--cohort-bag"), std::ios::binary);
  if (!bag) fail("missing competition cohort");
  d.header.implicit_count = {n};
  read_vector(bag, d.implicit_offsets, d.header.users+1);
  read_vector(bag, d.implicit_items, d.header.implicit_count);
  read_vector(bag, d.implicit_days, d.header.implicit_count);
  if (d.implicit_offsets.back()!=d.header.implicit_count || bag.peek()!=EOF) fail("bad competition cohort");
  for (auto &r : d.probe) r.rating = 0;
"""
    file.write_text(code.replace(before,before.removeprefix('const ')).replace(mark,patch))
    subprocess.run(['c++','-O3','-DNDEBUG','-std=c++17','-Wno-deprecated-declarations',str(src/'netflix.cpp'),
                    '-framework','Accelerate','-o',str(OUT/'netflix')],check=True)
    return dict(cohort_sha256=fingerprint, original_sources_commit=manifest['commit'],
                adapted_header_sha256=ne.sha(file),binary_sha256=ne.sha(OUT/'netflix'),cohort_addresses=n)


def train(mode, data):
    root = OUT/mode
    root.mkdir(exist_ok=True)
    argv = [str(OUT/'netflix'),'bayes','--data',str(data),'--cohort-bag',str(OUT/'cohort.bin'),
            '--model',str(root/'model.bin'),'--predictions',str(root/'model.f32'),
            '--samples','64','--factors','32','--burn-in','12','--snapshot-every','16',
            '--checkpoint-every','32','--threads','4','--seed','4173']
    (root/'command.json').write_text(json.dumps(argv)+'\n')
    with (root/'run.log').open('w') as stream:
        subprocess.run(argv,stdout=stream,stderr=subprocess.STDOUT,check=True)


def design(mode,base,rows,b2):
    vectors = [np.fromfile(OUT/mode/f'model.s{s:04d}.f32','<f4')[rows] for s in SWEEPS]
    return np.column_stack([v-base for v in vectors]+[vectors[-1]-b2])


def summary(y,f,p):
    return dict(pooled=ne.rmse(y,p),folds=[ne.rmse(y[f==i],p[f==i]) for i in range(5)])


def main():
    OUT.mkdir(exist_ok=True)
    sig=prepare()
    pdir=blend.OUT/json.loads((blend.OUT/'queue.json').read_text())['parent']
    sig.update(parent=str(pdir),qualifying_hashes={},qualifying_labels_used_for_fitting=False)
    (OUT/'setup.json').write_text(json.dumps(sig,indent=2)+'\n')
    train('known',KNOWN)
    y,f,_,_=anchor.history_control()
    base=np.load(pdir/'known-selected.npy')
    rows=ret.map_rows()
    ref=json.loads((ret.QUAL/'B2-fixed128/report.json').read_text())
    path=Path(ref['column'])
    assert ne.sha(path)==ref['column_sha256']
    x=design('known',base,rows,np.fromfile(path,'<f4')[rows])
    held=base.copy()
    for k in range(5):
        tr=f!=k
        held[~tr]=ne.clipped(base[~tr]+x[~tr]@fit(x[tr],(y-base)[tr]))
    w=fit(x,y-base)
    sig.update(control=summary(y,f,base),candidate=summary(y,f,held),weights=w.tolist())
    accept=ret.qualifies(sig['candidate'],sig['control'])
    sig['selected']='candidate' if accept else 'parent'
    (OUT/'selection.json').write_text(json.dumps(sig,indent=2)+'\n')
    if accept:
        train('qual',QUAL)
        parent=json.loads((pdir/'freeze.json').read_text())
        assert ne.sha(pdir/'selected.f32')==parent['qualifying_hashes']['selected']
        qb=np.fromfile(pdir/'selected.f32','<f4').astype('d')
        z=design('qual',qb,slice(None),np.fromfile(ret.OUT/'bayes/model.f32','<f4'))
        delta=z@w
        np.save(OUT/'oof.npy',held-base)
        np.save(OUT/'qual-correction.npy',delta)
        ne.clipped(qb+delta).astype('<f4').tofile(OUT/'selected.f32')
        sig['qualifying_hashes']['selected']=ne.sha(OUT/'selected.f32')
    (OUT/'freeze.json').write_text(json.dumps(sig,indent=2)+'\n')


if __name__=='__main__':
    main()
