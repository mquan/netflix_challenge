#!/usr/bin/env python3
"""Isolated inference for frozen residual trees; never import PyTorch."""
import argparse
import json
from pathlib import Path
import sys
import lightgbm as lgb
import numpy as np
from streaming_blend import digest
from qual_matrix import verified_matrix


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--models-dir',required=True);ap.add_argument('--qual',required=True)
    ap.add_argument('--output',required=True);ap.add_argument('--threads',type=int,default=2)
    a=ap.parse_args();root=Path(a.models_dir);report=json.loads((root/'report.json').read_text())
    if report['state']!='models_ready' or 'torch' in sys.modules:raise ValueError('invalid isolated model state')
    q=json.loads(Path(a.qual+'.json').read_text())
    if q['bank_sha256']!=report['bank_sha256'] or q['qualifying_labels_read'] is not False:
        raise ValueError('qualifying schema differs')
    raw=verified_matrix(a.qual,q);result=np.zeros(len(raw));seeds=report['config']['seeds']
    for seed in seeds:
        path=root/f'tree-{seed}.txt'
        if digest(path)!=report['outputs'][path.name]:raise ValueError('tree model changed')
        model=lgb.Booster(model_file=str(path))
        if model.num_feature()!=raw.shape[1]:raise ValueError('feature width differs')
        for lo in range(0,len(raw),65536):
            result[lo:lo+65536]+=model.predict(raw[lo:lo+65536],num_threads=a.threads)/len(seeds)
    np.save(a.output,result,allow_pickle=False)
    Path(a.output+'.json').write_text(json.dumps(dict(state='complete',torch_loaded=False,
        models_report_sha256=digest(root/'report.json'),qual_manifest_sha256=digest(a.qual+'.json'),
        output_sha256=digest(a.output),qualifying_labels_read=False),indent=2)+'\n')


if __name__=='__main__':main()
