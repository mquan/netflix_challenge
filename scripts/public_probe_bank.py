#!/usr/bin/env python3
"""Import only original-split base predictions from gtakacs' public store.

Never fetches qualifying labels/predictions, alternative splits, or trained
blender outputs. Source model checkpoint/hyperparameter selection used Probe,
so these columns support development comparisons, not an untouched test claim.
"""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.request

import numpy as np


BASE_URL = 'https://huggingface.co/buckets/gtakacs/netflix_prize/resolve/'
PREFIXES = ('als', 'asym-', 'attn-', 'bk1-', 'bkbias', 'bknbr', 'catmf-',
            'cfnade_', 'dnn-', 'fbias', 'ge14', 'knn-', 'mf-', 'mfrbm-',
            'rbm-', 'rbm_', 'rbmb-', 'rbmr-', 'rbmx-', 'tsvd')


def eligible(path):
    name = path.removeprefix('preds_old/')
    return (path.startswith('preds_old/') and '/' not in name
            and name.endswith('.probe.npy') and name.startswith(PREFIXES)
            and not any(s in name for s in ('fwls', 'blended', 'epblended')))


def eligible_three(path):
    name = path.removeprefix('preds_3/')
    return (path.startswith('preds_3/s3-') and '/' not in name
            and name.endswith('.probe_3.npy')
            and not any(s in name for s in ('fwls', 'blended', '__epochs')))


def eligible_three_support(path):
    return ((path.startswith('preds_3/vf/vf') or path.startswith('preds_3/lat/s3-'))
            and path.endswith('.probe_3.npy') and '..' not in path)


def remap(values, mapping, source_rows=None):
    if values.ndim != 1 or len(values) != (len(mapping) if source_rows is None else source_rows):
        raise ValueError('source column shape does not match mapping')
    if not np.isfinite(values).all():
        raise ValueError('nonfinite source values')
    return values[mapping].astype('<f4')


def verified_file(path, entry):
    return (path.exists() and path.stat().st_size == entry['size']
            and hashlib.md5(path.read_bytes()).hexdigest() == entry['md5'])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--index', required=True, help='public index TOML, not labels')
    ap.add_argument('--mapping', required=True, help='processed row -> source row u32')
    ap.add_argument('--output', required=True)
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--split', choices=['old', 'three'], default='old')
    ap.add_argument('--source-rows', type=int)
    ap.add_argument('--kind', choices=['predictions', 'support'], default='predictions')
    a = ap.parse_args()
    root = Path(a.output); root.mkdir(parents=True, exist_ok=True)
    entries = [dict(path=p, size=int(n), md5=h) for p, n, h in re.findall(
        r'path = "([^"]+)"\s+size = (\d+)\s+md5 = "([^"]+)"', Path(a.index).read_text())
        if (eligible(p) if a.split == 'old' else
            eligible_three(p) if a.kind == 'predictions' else eligible_three_support(p))]
    if not entries: raise ValueError('no eligible original-split columns')
    mapping = np.fromfile(a.mapping, dtype='<u4')
    source_rows = a.source_rows or len(mapping)
    if len(np.unique(mapping)) != len(mapping) or np.any(mapping >= source_rows):
        raise ValueError('mapping is not injective or is out of range')
    if a.split == 'old' and not np.array_equal(np.sort(mapping), np.arange(len(mapping))):
        raise ValueError('original-split mapping must be bijective')
    plan = dict(source='https://github.com/gtakacs/netflix_prize',
                index_sha256=hashlib.sha256(Path(a.index).read_bytes()).hexdigest(),
                mapping_sha256=hashlib.sha256(Path(a.mapping).read_bytes()).hexdigest(),
                rows=len(mapping), source_rows=source_rows, split=a.split, kind=a.kind, entries=entries,
                exclusion='qualifying labels and predictions, label-containing splits, learned blender outputs',
                metric='Probe development only; source checkpoint selection used Probe')
    plan_path = root/'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError('existing import plan differs')
    plan_path.write_text(json.dumps(plan, indent=2)+'\n')
    print(json.dumps(dict(columns=len(entries), bytes=sum(e['size'] for e in entries))), flush=True)

    def fetch(e):
        name = Path(e['path']).name
        raw = root/name
        if not verified_file(raw, e):
            for attempt in range(4):
                try:
                    with urllib.request.urlopen(BASE_URL+e['path'], timeout=90) as response:
                        payload = response.read()
                    if len(payload) != e['size'] or hashlib.md5(payload).hexdigest() != e['md5']:
                        raise ValueError('download size/MD5 mismatch: '+name)
                    temporary = raw.with_suffix('.part'); temporary.write_bytes(payload)
                    temporary.replace(raw)
                    break
                except Exception:
                    if attempt == 3: raise
                    time.sleep(1+attempt)
        values = remap(np.load(raw, allow_pickle=False), mapping, source_rows)
        dest = root/name.replace('.probe_3.npy', '.f32').replace('.probe.npy', '.f32')
        temporary = dest.with_suffix('.part'); values.tofile(temporary); temporary.replace(dest)
        return dict(source=e['path'], path=str(dest.resolve()),
                    sha256=hashlib.sha256(dest.read_bytes()).hexdigest(), md5=e['md5'])

    with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
        records = []
        for record in pool.map(fetch, entries):
            records.append(record)
            print('imported', len(records), record['source'], flush=True)
    (root/'manifest.json').write_text(json.dumps(dict(plan=plan, columns=records), indent=2)+'\n')


if __name__ == '__main__': main()
