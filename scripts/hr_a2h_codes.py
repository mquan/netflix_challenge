#!/usr/bin/env python3
"""Export label-free A2h representation features for every probe row (representation stacking).

For each query the transformer gives a user code at its [MASK] token, which the
session network shifts by same-day context into the vector the output head
decodes. Features (float32, probe order):

- user_pc0..7: the session-adjusted vector projected onto the top 8 principal
  directions of the movies' liking vectors sum_s (s-3) * output[i, s];
- item_pc0..7: the target movie's liking vector in the same basis;
- shift_norm, shift_cos: size of the session shift and its cosine with the code.

The PCA basis uses movie parameters only. Only user/item/day identifiers are
read from queries. A reconstruction check recomputes the expected rating from
the captured vectors and compares it with the saved prediction column.
"""
import argparse
import dataclasses
import hashlib
import json
import pathlib
import sys

import numpy as np
import torch


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True, help='scripts directory of the trainer that wrote the checkpoint')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--reference', required=True, help='saved prediction column of this checkpoint on --data')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--prefix', default='a2h')
    p.add_argument('--components', type=int, default=8)
    p.add_argument('--device', default='mps')
    p.add_argument('--threads', type=int, default=2)
    a = p.parse_args()
    out = pathlib.Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if list(out.glob(a.prefix + '_*.f32')):
        raise FileExistsError(out)
    source = pathlib.Path(a.source).resolve()
    sys.path.insert(0, str(source))
    import masked_sequence as ms
    from gbdt_blend import load
    from masked_history import gather
    torch.set_num_threads(a.threads)
    saved = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    c = dataclasses.replace(ms.Config(**saved['config']), device=a.device)
    if c.context != 'session' or c.head != 'distribution':
        raise ValueError('expects a session-context, distribution-head checkpoint')
    model = ms.SequenceModel(saved['items'], c, saved['weeks'])
    model.load_state_dict(saved['state'])
    model.to(a.device).eval()
    d = load(a.data)
    if ms.weeks(d) != saved['weeks'] or d['items'] != saved['items']:
        raise ValueError('dataset calendar or item count differs from the checkpoint')

    stars = torch.arange(1, 6, dtype=torch.float32)
    with torch.no_grad():
        liking = torch.einsum('s,isr->ir', stars - 3, model.output.detach().cpu()).numpy().astype(np.float64)
    centre = liking.mean(0)
    _, _, vt = np.linalg.svd(liking - centre, full_matrices=False)
    basis = vt[:a.components].T  # rank x k, movie parameters only

    q = d['probe']
    n = len(q)
    vector = np.empty((n, c.rank), dtype=np.float32)
    code = np.empty((n, c.rank), dtype=np.float32)
    recon = np.empty(n, dtype=np.float32)
    users = q['user'].astype(np.int64)
    order = np.argsort(users, kind='stable')
    offsets = np.searchsorted(users[order], np.arange(d['users'] + 1))
    pc = dataclasses.replace(c, batch_tokens=2 * c.batch_tokens, batch_users=2 * c.batch_users)
    lengths = ms.capped_lengths(d, pc, np.diff(offsets))
    host = dataclasses.replace(pc, device='cpu')

    def make(group):
        rows, index, _ = gather(offsets, group)
        sel = order[index]
        items = q['item'][sel].astype(np.int64) - 1
        return ms.prepare_batch(d, group, host, queries=(rows, items, q['day'][sel]))

    with torch.no_grad():
        for group, b in ms.prefetched(make, ms.length_batches(lengths, np.unique(users), pc), pc.device):
            sel = order[gather(offsets, group)[1]]
            raw_code = model.encode(b)
            v = model.target_code(raw_code, b)
            vector[sel] = v.cpu().numpy()
            code[sel] = raw_code[b['target_rows']].cpu().numpy()
            recon[sel] = model.expected(model.decode(raw_code, b['features'], b['target_rows'],
                                                     b['target_items'], b)).cpu().numpy()
    reference = np.fromfile(a.reference, dtype='<f4')
    max_error = float(np.abs(recon - reference).max())
    if max_error > 1e-3:
        raise ValueError(f'reconstruction differs from reference by {max_error}')

    items = q['item'].astype(np.int64) - 1
    shift = vector - code
    features = {}
    up = vector.astype(np.float64) @ basis
    ip = (liking[items] - centre) @ basis
    for k in range(a.components):
        features[f'user_pc{k}'] = up[:, k]
        features[f'item_pc{k}'] = ip[:, k]
    sn = np.linalg.norm(shift, axis=1)
    features['shift_norm'] = sn
    features['shift_cos'] = (shift * code).sum(1) / np.maximum(sn * np.linalg.norm(code, axis=1), 1e-8)
    hashes = {}
    for name, values in features.items():
        path = out / f'{a.prefix}_{name}.f32'
        np.asarray(values, dtype='<f4').tofile(path)
        hashes[str(path)] = digest(path)
    report = dict(checkpoint=a.checkpoint, checkpoint_sha256=digest(a.checkpoint),
                  source_sha256=digest(source / 'masked_sequence.py'),
                  base_source_sha256=digest(source / 'masked_history.py'),
                  data=a.data, data_sha256=digest(a.data), reference=a.reference,
                  reference_sha256=digest(a.reference), reconstruction_max_abs_error=max_error,
                  script_sha256=digest(__file__), components=a.components,
                  explained_variance=[float(x) for x in
                                      (np.linalg.svd(liking - centre, compute_uv=False) ** 2)[:a.components]],
                  outputs=hashes)
    (out / f'{a.prefix}_features.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'outputs'}, indent=2))


if __name__ == '__main__':
    main()
