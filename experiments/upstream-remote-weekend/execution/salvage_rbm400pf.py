"""Owner-approved partial export of s3-rbm400pf (r8): the fulltrain->qual phase
diverged to NaN at epoch 6, so the helper's full export refuses it. This exports
only the finite epoch pairs ep01..ep05 (probe_3 from train_3 phase, qual from
fulltrain phase, same epoch index) in the helper's own .npz format, then packs a
verified tar part + index for upload_release.py. Reads no qual labels."""
import json, sys, tarfile, hashlib
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
import upstream_remote_weekend as h

TREE = ROOT / 'artifacts/upstream-remote-weekend-r8/source'
NAME = 's3-rbm400pf'; JOB = NAME + '-salvage'
DEST = ROOT / 'artifacts/exports' / JOB; ARC = ROOT / 'artifacts/archives' / JOB
EPOCHS = range(1, 6)
assert not DEST.exists() and not ARC.exists(), 'existing salvage preserved'

mapping = h.projection(h.arrays(TREE / 'data/probe_3', labels=True),
                       h.arrays(TREE / 'data/probex', labels=True))
if not np.array_equal(mapping, np.load(TREE.parent / 'probex-row-map.npy')):
    raise SystemExit('independent row projection differs')
DEST.mkdir(parents=True)
records = []
for e in EPOCHS:
    col = f'{NAME}_ep{e:02}'
    probe = h.vector(TREE / f'preds_3/{col}.probe_3.npy', 1760494)   # raises on NaN
    qual = h.vector(TREE / f'preds_3/{col}.qual.npy', 2817131)
    known = probe[mapping]; target = DEST / (col + '.npz')
    np.savez_compressed(target, probex=known, qual=qual)
    with np.load(target, allow_pickle=False) as s:
        for k, v in [('probex', known), ('qual', qual)]:
            if not np.array_equal(s[k].view(np.uint32), v.view(np.uint32)): raise SystemExit('lossless differs')
    records.append(dict(column=col, file=target.name, sha256=h.sha(target),
                        native_probe_sha256=h.sha(TREE / f'preds_3/{col}.probe_3.npy'),
                        native_qual_sha256=h.sha(TREE / f'preds_3/{col}.qual.npy')))
out = TREE / f'preds_3/{NAME}.out'
nan6 = np.load(TREE / f'preds_3/{NAME}_ep06.qual.npy', mmap_mode='r')
proof = dict(state='owner_approved_partial_salvage_finite_epoch_pairs', observed=h.stamp(), job=NAME,
             tree=str(TREE.parent.relative_to(ROOT)), epochs_exported=list(EPOCHS),
             reason='fulltrain->qual phase diverged: ep06+ qual all NaN; helper export requires finite terminal pair',
             ep06_qual_nan_count=int(np.isnan(nan6).sum()), records=records,
             trace_sha256=h.sha(out), plan_sha256=h.sha(h.PLAN), independent_known_projection=True,
             qualifying_labels_read=False, scope='Own numerical outputs, partial (epochs 1-5 of 15). No blend claim.')
(DEST / 'verification.json').write_text(json.dumps(proof, indent=1) + '\n')
(DEST / out.name).write_bytes(out.read_bytes())

ARC.mkdir(parents=True)
part = ARC / f'{JOB}-part01.tar.gz'; files = sorted(p for p in DEST.iterdir() if p.is_file())
with tarfile.open(part, 'w:gz', compresslevel=3) as t:
    for p in files: t.add(p, arcname=f'{JOB}/{p.name}', recursive=False)
members = {}
with tarfile.open(part, 'r:gz') as t:
    for m in t:
        d = hashlib.sha256(t.extractfile(m).read()).hexdigest(); members[m.name.split('/', 1)[1]] = d
if members != {p.name: h.sha(p) for p in files}: raise SystemExit('archive members differ')
(ARC / 'index.json').write_text(json.dumps(dict(state=proof['state'], job=JOB,
    source_verification_sha256=h.sha(DEST / 'verification.json'), plan_sha256=h.sha(h.PLAN),
    parts=[dict(file=part.name, bytes=part.stat().st_size, sha256=h.sha(part), members_sha256=members)],
    raw_dataset_uploaded=False, qualifying_labels_read=False, observed=h.stamp()), indent=1) + '\n')
print('salvaged', [r['column'] for r in records], part.stat().st_size, 'bytes')
