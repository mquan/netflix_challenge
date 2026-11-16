#!/usr/bin/env python3
"""Export frozen predictions in original Netflix submission order, locally."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
from audit_qualifying import frozen_predictions


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--dataset-manifest', required=True)
    ap.add_argument('--output', required=True)
    a = ap.parse_args()
    pred, model = frozen_predictions(a.model_dir)
    manifest = json.loads(Path(a.dataset_manifest).read_text())
    raw = Path(manifest['qualifying'])
    if sha(raw) != manifest['qualifying_sha256']:
        raise ValueError('qualifying order source changed')
    if len(pred) != manifest['qualifying_count']:
        raise ValueError('qualifying row count differs')
    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    n = headers = 0
    with raw.open() as source, output.open('xb') as binary:
        with gzip.GzipFile(filename='', mode='wb', fileobj=binary, mtime=0) as dest:
            for line in source:
                line = line.strip()
                if line.endswith(':'):
                    dest.write((line+'\n').encode('ascii'))
                    headers += 1
                else:
                    if n >= len(pred):
                        raise ValueError('too many qualifying rows')
                    dest.write((format(pred[n], '.17g')+'\n').encode('ascii'))
                    n += 1
    if n != len(pred):
        raise ValueError('too few qualifying rows')
    # Verify exact row order, headers, and floating-point round trips.
    n = 0
    with raw.open() as source, gzip.open(output, 'rt') as exported:
        for line in source:
            value = exported.readline().strip()
            if line.strip().endswith(':'):
                if value != line.strip():
                    raise ValueError('movie header differs')
            else:
                if float(value) != pred[n]:
                    raise ValueError('prediction round trip differs')
                n += 1
        if exported.read():
            raise ValueError('unexpected extra export rows')
    report = dict(state='complete', output=str(output), rows=n, movie_headers=headers,
                  sha256=sha(output), prediction_sha256=model['frozen_prediction_sha256'],
                  raw_qualifying_sha256=manifest['qualifying_sha256'],
                  verified_exact_round_trip=True, uploaded=False)
    Path(str(output)+'.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
