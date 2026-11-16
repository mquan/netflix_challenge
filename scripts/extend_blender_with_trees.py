#!/usr/bin/env python3
"""Add independently trained development trees to immutable NN blend models."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from streaming_blend import digest,predict_ridge


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--models-dir',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--threads',type=int,default=2)
    a=ap.parse_args();source=Path(a.models_dir)
    m=json.loads((source/'report.json').read_text());source_sha=digest(source/'report.json')
    if (m['state']!='models_ready' or not m['config']['nn_only'] or
        m['qualifying_inputs_provided'] is not False or m['qualifying_labels_read'] is not False):
        raise ValueError('expected development-only frozen NN models')
    bank=m['config']['bank'];b=json.loads(Path(bank+'.inputs.json').read_text())
    if digest(bank+'.inputs.json')!=m['bank_sha256'] or b['config'].get('sample_weights'):
        raise ValueError('different or weighted development bank')
    if digest(b['config']['meta'])!=b['meta_sha256']:raise ValueError('development metadata changed')
    root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
    for name,expected in m['outputs'].items():
        if digest(source/name)!=expected:raise ValueError('NN model changed')
        os.link(source/name,root/name)
    raw=np.load(bank+'.x.npy',mmap_mode='r');context=np.load(b['config']['meta'])['context']
    w=np.load(source/'ridge-weights.npz');inputs={}
    for name,key in [('development-base.npy','full'),('development-inner-base.npy','inner')]:
        expected=predict_ridge(raw[:,:b['columns']],context,w[key])
        if not np.allclose(expected,np.load(source/name),rtol=0,atol=1e-10):
            raise ValueError('cached ridge baseline differs from frozen weights')
        os.link(source/name,root/name);inputs[name]=digest(root/name)
    del raw,context,w
    ignored=[]
    if m['config'].get('drop_unrated'):
        ignored=[m['feature_width']-len(b['extra_meta_names'])+b['extra_meta_names'].index('log_unrated_n')]
    plan=dict(bank=bank,bank_sha256=m['bank_sha256'],output=str(root),
              seeds=m['config']['seeds'],threads=a.threads,inputs=inputs,ignored_columns=ignored)
    (root/'tree-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    report=copy.deepcopy(m);report['config'].update(output=str(root),nn_only=False)
    report.update(state='tree_worker',source_nn_report_sha256=source_sha,
        source_nn_models=str(source),blend_rule='Fixed equal average of seed-averaged tree and NN2 residuals',
        tree_suppressed_columns=ignored,tree_suppressed_features=['log_unrated_n'] if ignored else [])
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    subprocess.run([sys.executable,str(Path(__file__).with_name('train_tree_models.py')),
                    '--plan',str(root/'tree-plan.json')],check=True)
    tr=json.loads((root/'tree-report.json').read_text())
    if (tr['state']!='complete' or tr['torch_loaded'] or tr['bank_sha256']!=m['bank_sha256'] or
        tr['qualifying_labels_read'] is not False or len(tr['tree_rounds'])!=len(m['config']['seeds'])):
        raise ValueError('tree training provenance differs')
    if digest(source/'report.json')!=source_sha:raise ValueError('NN source report changed')
    for name,expected in m['outputs'].items():
        if digest(root/name)!=expected:raise ValueError('NN checkpoint changed during tree fitting')
    for name,expected in tr['outputs'].items():
        if digest(root/name)!=expected:raise ValueError('tree checkpoint changed')
    report.update(state='models_ready',tree_rounds=tr['tree_rounds'],
                  outputs=dict(m['outputs'],**tr['outputs']),tree_report_sha256=digest(root/'tree-report.json'))
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(state='models_ready',tree_rounds=tr['tree_rounds'])),flush=True)


if __name__=='__main__':main()
