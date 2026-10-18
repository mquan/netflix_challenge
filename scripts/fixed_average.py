#!/usr/bin/env python3
"""Save and score a fixed equal average of two probe prediction columns.

The equal weight is specified before this script reads labels. The reported
score is still development evidence if the inputs were selected using probe.
"""
import argparse
import hashlib
import json

import numpy as np

from gbdt_blend import blend_fold, load


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', required=True)
    ap.add_argument('--a', required=True)
    ap.add_argument('--b', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--report', required=True)
    a = ap.parse_args()
    d = load(a.data)
    y = d['probe']['rating'].astype(np.float64)
    first = np.fromfile(a.a, dtype='<f4').astype(np.float64)
    second = np.fromfile(a.b, dtype='<f4').astype(np.float64)
    if first.size != y.size or second.size != y.size:
        ap.error('prediction length mismatch')
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        ap.error('nonfinite prediction')
    prediction = np.clip((first+second)/2, 1, 5)
    prediction.astype('<f4').tofile(a.output)
    fold = blend_fold(d['probe']['user'])
    score = lambda take: float(np.sqrt(np.mean((y[take]-prediction[take])**2)))
    report = dict(method='fixed equal average', weight_a=.5, weight_b=.5,
                  data=dict(path=a.data, sha256=digest(a.data)),
                  inputs=[dict(path=path, sha256=digest(path)) for path in (a.a, a.b)],
                  output=dict(path=a.output, sha256=digest(a.output)),
                  rmse=score(slice(None)),
                  fold_rmse=[score(fold == f) for f in range(5)],
                  caveat='Probe-guided development score; not independent test')
    with open(a.report, 'w') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
