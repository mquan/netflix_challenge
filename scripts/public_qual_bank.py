#!/usr/bin/env python3
"""Fetch matching full-training qualifying PREDICTIONS, never target labels.

Selects only counterparts of already audited base/support files. No wildcard
blender outputs and no data/qual_ratings files can enter this plan.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import time
import urllib.request
import numpy as np
from public_probe_bank import (BASE_URL, eligible, eligible_three, eligible_three_support,
                               eligible_new, eligible_new_support, verified_file)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--index',required=True)
    ap.add_argument('--manifests',nargs='+',required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--jobs',type=int,default=6)
    ap.add_argument('--reuse-manifests',nargs='*',default=[])
    a=ap.parse_args()
    index={p:dict(path=p,size=int(n),md5=h) for p,n,h in re.findall(
        r'path = "([^"]+)"\s+size = (\d+)\s+md5 = "([^"]+)"',Path(a.index).read_text())}
    root=Path(a.output); root.mkdir(parents=True,exist_ok=True)
    cache={}
    for manifest in a.reuse_manifests:
        for col in json.loads(Path(manifest).read_text())['columns']:
            cache[col['md5']]=Path(col['path'])
    selected=[]; omitted=[]
    for manifest in a.manifests:
        for col in json.loads(Path(manifest).read_text())['columns']:
            source=col['source']
            if not (eligible(source) or eligible_three(source) or eligible_three_support(source)
                    or eligible_new(source) or eligible_new_support(source)):
                raise ValueError('unaudited source column '+source)
            target=source.replace('.probe_3.npy','.qual.npy').replace('.probe.npy','.qual.npy').replace('.probex.npy','.qual.npy')
            if target not in index:
                omitted.append(source)
                continue
            selected.append(index[target])
    if len({e['path'] for e in selected})!=len(selected): raise ValueError('duplicate qualifying file')
    plan=dict(rows=2817131,entries=selected,labels_requested=False,
              omitted_unavailable_counterparts=omitted,
              source_manifests={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in a.manifests},
              source_order='original qualifying.txt parse order',
              purpose='Unlabeled final predictions, no training or score selection')
    pp=root/'plan.json'
    if pp.exists() and json.loads(pp.read_text())!=plan: raise ValueError('plan differs')
    pp.write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(files=len(selected),bytes=sum(e['size'] for e in selected),
                          omitted_unavailable=omitted)),flush=True)

    def fetch(e):
        dest=root/e['path']; dest.parent.mkdir(parents=True,exist_ok=True)
        cached=cache.get(e['md5'])
        if not dest.exists() and cached is not None and verified_file(cached,e):
            os.link(cached,dest)
        if not verified_file(dest,e):
            for attempt in range(4):
                try:
                    with urllib.request.urlopen(BASE_URL+e['path'],timeout=90) as response: b=response.read()
                    if len(b)!=e['size'] or hashlib.md5(b).hexdigest()!=e['md5']: raise ValueError('checksum mismatch')
                    tmp=dest.with_suffix('.part'); tmp.write_bytes(b); tmp.replace(dest); break
                except Exception:
                    if attempt==3: raise
                    time.sleep(attempt+1)
        values=np.load(dest,allow_pickle=False)
        if values.shape!=(plan['rows'],) or not np.isfinite(values).all(): raise ValueError('invalid predictions '+str(dest))
        return dict(source=e['path'],path=str(dest.resolve()),md5=e['md5'],
                    sha256=hashlib.sha256(dest.read_bytes()).hexdigest())

    with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
        columns=[]
        for c in pool.map(fetch,selected):
            columns.append(c)
            print('verified',len(columns),c['source'],flush=True)
    (root/'manifest.json').write_text(json.dumps(dict(plan=plan,columns=columns),indent=2)+'\n')


if __name__=='__main__': main()
