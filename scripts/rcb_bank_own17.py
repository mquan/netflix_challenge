#!/usr/bin/env python3
"""Build banks own17 (coordinator 17 columns) and own17x (own17 + bank v2) for rcb_blend.py.
Reads predictions and probex labels only; never opens qualifying labels."""
import hashlib, json, shutil, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; A = ROOT / 'artifacts'; E = ROOT / 'experiments/remote-context-blend'
D = A / 'own17-download'

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()

m = json.loads((D / 'own17-bank-columns.json').read_text())
K = np.load(D / 'own17-bank-known.npy', mmap_mode='r'); Q = np.load(D / 'own17-bank-qual.npy', mmap_mode='r')
v2 = json.loads((E / 'bank-v2.json').read_text())
K2 = np.load(A / 'bank-v2/known.npy', mmap_mode='r'); Q2 = np.load(A / 'bank-v2/qual.npy', mmap_mode='r')
o17 = [dict(column='own17:' + c['name'], kind=c['provenance_kind'], family=c['family'],
            known_sha256=c['known_sha256'], qual_sha256=c['qual_sha256']) for c in m['columns']]
for name, cols, parts in [('own17', o17, [(K, Q)]), ('own17x', o17 + v2['columns'], [(K, Q), (K2, Q2)])]:
    out = A / f'bank-{name}'; out.mkdir(exist_ok=True)
    km = np.lib.format.open_memmap(out / 'known.npy', 'w+', np.float32, (K.shape[0], len(cols)))
    qm = np.lib.format.open_memmap(out / 'qual.npy', 'w+', np.float32, (Q.shape[0], len(cols)))
    j = 0
    for k, q in parts:
        km[:, j:j + k.shape[1]] = k; qm[:, j:j + q.shape[1]] = q; j += k.shape[1]
    km.flush(); qm.flush(); del km, qm
    for f in ('known_y.npy', 'known_users.npy', 'known_ctx.npz', 'qual_ctx.npz'): shutil.copyfile(A / 'bank-v2' / f, out / f)
    rec = dict(version=name, columns=cols, n_known=int(K.shape[0]), n_qual=int(Q.shape[0]),
               own17_assets=dict(columns_json=sha(D / 'own17-bank-columns.json'), known=sha(D / 'own17-bank-known.npy'),
                                 qual=sha(D / 'own17-bank-qual.npy')),
               known_sha256=sha(out / 'known.npy'), qual_sha256=sha(out / 'qual.npy'),
               contexts='copied from bank-v2 (same rows)', cv_note='non-nested: own17 columns are user-fold OOF stack outputs',
               qualifying_labels_read=False)
    (E / f'bank-{name}.json').write_text(json.dumps(rec, indent=1) + '\n'); print(name, len(cols))
