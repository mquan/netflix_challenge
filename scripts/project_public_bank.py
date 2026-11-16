#!/usr/bin/env python3
"""Project strict original-Probe base columns to a separately audited subset."""
import argparse
import json
from pathlib import Path
import numpy as np
from streaming_blend import digest


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',required=True)
    ap.add_argument('--rows',required=True)
    ap.add_argument('--output',required=True)
    a=ap.parse_args()
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    if (root/'manifest.json').exists(): ap.error('output exists')
    rows=np.fromfile(a.rows,dtype='<u4')
    if not len(rows) or len(np.unique(rows))!=len(rows): raise ValueError('invalid projection')
    parent=json.loads(Path(a.manifest).read_text())
    records=[]
    for col in parent['columns']:
        if digest(col['path'])!=col['sha256']: raise ValueError('input changed')
        source=np.fromfile(col['path'],dtype='<f4')
        if np.any(rows>=len(source)): raise ValueError('projection out of bounds')
        dest=root/col['source'].replace('.probe_3.npy','.f32').replace('.probe.npy','.f32')
        dest.parent.mkdir(parents=True,exist_ok=True)
        source[rows].tofile(dest)
        record=dict(col,path=str(dest.resolve()),sha256=digest(dest))
        records.append(record)
    report=dict(columns=records,parent_sha256=digest(a.manifest),rows_sha256=digest(a.rows),
                rows=len(rows),metric='Alternative ProbeX only; not original full-Probe CV',
                upstream='Original Probe entirely excluded from these source base fits')
    (root/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Projected',len(records),'base columns to',len(rows),'rows',flush=True)


if __name__=='__main__': main()
