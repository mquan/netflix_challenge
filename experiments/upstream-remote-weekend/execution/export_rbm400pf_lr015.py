"""Export the owner-requested s3-rbm400pf rerun with LR=0.015 (tree r11, run
outside the helper so plan.json stays bound for the live jobs). Same binary and
recipe as plan job s3-rbm400pf except LR (0.02 -> 0.015) and RAYON threads.
Columns are renamed s3-rbm400pf-lr015*. Exports every epoch pair whose probe_3
and qual are both finite, plus the terminal pair if finite. Reads no qual labels."""
import json, sys, tarfile, hashlib
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
import upstream_remote_weekend as h

TREE = ROOT / 'artifacts/upstream-remote-weekend-r11/source'
NAME = 's3-rbm400pf'; JOB = 's3-rbm400pf-lr015'; EPOCHS = 15
DEST = ROOT / 'artifacts/exports' / JOB; ARC = ROOT / 'artifacts/archives' / JOB
assert not DEST.exists() and not ARC.exists(), 'existing export preserved'

def finite(path, n):
    a = np.load(path, mmap_mode='r', allow_pickle=False)
    return a.shape == (n,) and a.dtype == np.float32 and bool(np.isfinite(a).all())

mapping = h.projection(h.arrays(TREE / 'data/probe_3', labels=True),
                       h.arrays(TREE / 'data/probex', labels=True))
if not np.array_equal(mapping, np.load(TREE.parent / 'probex-row-map.npy')):
    raise SystemExit('independent row projection differs')
cols = [(NAME, JOB)] + [(f'{NAME}_ep{e:02}', f'{JOB}_ep{e:02}') for e in range(1, EPOCHS + 1)]
DEST.mkdir(parents=True); records = []; skipped = []
for native, col in cols:
    p, q = TREE / f'preds_3/{native}.probe_3.npy', TREE / f'preds_3/{native}.qual.npy'
    if not (p.exists() and q.exists() and finite(p, 1760494) and finite(q, 2817131)):
        skipped.append(native); continue
    known = h.vector(p, 1760494)[mapping]; qual = h.vector(q, 2817131)
    target = DEST / (col + '.npz'); np.savez_compressed(target, probex=known, qual=qual)
    with np.load(target, allow_pickle=False) as s:
        for k, v in [('probex', known), ('qual', qual)]:
            if not np.array_equal(s[k].view(np.uint32), v.view(np.uint32)): raise SystemExit('lossless differs')
    records.append(dict(column=col, native=native, file=target.name, sha256=h.sha(target),
                        native_probe_sha256=h.sha(p), native_qual_sha256=h.sha(q)))
if not records: raise SystemExit('no finite epoch pairs')
out = TREE / f'preds_3/{NAME}.out'; log = TREE.parent / f'{JOB}.log'
complete = not skipped
proof = dict(state='complete_lr015_rerun' if complete else 'partial_lr015_rerun_finite_pairs_only',
             observed=h.stamp(), job=JOB, tree=str(TREE.parent.relative_to(ROOT)),
             recipe_change=dict(LR='0.015 (plan: 0.02)', RAYON_NUM_THREADS='8 (plan: 4)'),
             reason='plan recipe diverged to NaN in fulltrain->qual phase at epoch 6 (r8)',
             binary_sha256=h.sha(TREE / 'target/release/fit'), records=records, skipped_nonfinite=skipped,
             trace_sha256=h.sha(out), plan_sha256=h.sha(h.PLAN), independent_known_projection=True,
             qualifying_labels_read=False, scope='Own numerical outputs. No blend claim.')
(DEST / 'verification.json').write_text(json.dumps(proof, indent=1) + '\n')
(DEST / out.name).write_bytes(out.read_bytes())
if log.exists(): (DEST / log.name).write_bytes(log.read_bytes())

ARC.mkdir(parents=True)
part = ARC / f'{JOB}-part01.tar.gz'; files = sorted(p for p in DEST.iterdir() if p.is_file())
with tarfile.open(part, 'w:gz', compresslevel=3) as t:
    for p in files: t.add(p, arcname=f'{JOB}/{p.name}', recursive=False)
members = {}
with tarfile.open(part, 'r:gz') as t:
    for m in t: members[m.name.split('/', 1)[1]] = hashlib.sha256(t.extractfile(m).read()).hexdigest()
if members != {p.name: h.sha(p) for p in files}: raise SystemExit('archive members differ')
(ARC / 'index.json').write_text(json.dumps(dict(state=proof['state'], job=JOB,
    source_verification_sha256=h.sha(DEST / 'verification.json'), plan_sha256=h.sha(h.PLAN),
    parts=[dict(file=part.name, bytes=part.stat().st_size, sha256=h.sha(part), members_sha256=members)],
    raw_dataset_uploaded=False, qualifying_labels_read=False, observed=h.stamp()), indent=1) + '\n')
print(proof['state'], len(records), 'columns', 'skipped', skipped, part.stat().st_size, 'bytes')
