#!/usr/bin/env python3
"""Refit the retained B2 Bayesian sampler for its fixed 128 sweeps on known data."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import time
import numpy as np
from production_neural import digest,validate_dataset


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--recipe',required=True);ap.add_argument('--dataset-manifest',required=True)
    ap.add_argument('--output',required=True)
    a=ap.parse_args();recipe=json.loads(Path(a.recipe).read_text())
    for path,expected in recipe['verified_files'].items():
        if digest(path)!=expected:raise ValueError('recipe input changed: '+path)
    historical=json.loads(Path(recipe['historical_report']).read_text())
    if digest(recipe['column'])!=historical['predictions_artifact']['sha256']:
        raise ValueError('development column does not match historical fixed-sweep prediction')
    expected=['--factors','64','--samples','128','--burn-in','16','--snapshot-every','16',
              '--checkpoint-every','8','--threads','4','--seed','2026']
    command=historical['command'];start_flags=command.index('--factors')
    if command[start_flags:]!=expected:raise ValueError('unexpected B2 recipe')
    data,manifest=validate_dataset(a.dataset_manifest)
    from gbdt_blend import load
    d=load(data)
    if len(d['probe'])!=manifest['qualifying_count'] or not np.all(d['probe']['rating']==0):
        raise ValueError('query labels must be unknown zero sentinels')
    if len(d['train_ratings'])!=manifest['train_count'] or not np.all((d['train_ratings']>=1)&(d['train_ratings']<=5)):
        raise ValueError('invalid known training ratings')
    del d
    root=Path(a.output);root.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    checkpoint=root/'model.bin';predictions=root/'model.f32'
    cmd=[recipe['binary'],'bayes','--data',str(data),'--model',str(checkpoint),
         '--predictions',str(predictions)]+expected
    report=dict(state='training',kind='fixed_sweep_bayesian_refit',column=recipe['column'],
        column_sha256=digest(recipe['column']),recipe_sha256=digest(a.recipe),
        source_manifest_sha256=digest(recipe['source_manifest']),binary_sha256=digest(recipe['binary']),
        original_binary_sha256=historical['binary_sha256'],
        binary_reconstruction_note='Rebuilt from historical binary-name commit 0c1f29d, with logging-only NaN for zero query labels. Original binary unavailable; reconstructed source and compiler binary retained.',
        command=cmd,fixed_samples=128,fixed_training_steps=128,training_step_unit='Gibbs sweep',
        training_requested=True,qualifying_labels_read=False,metric=None,records=[],
        dataset_manifest_sha256=digest(a.dataset_manifest),data_sha256=manifest['outputs'][data.name],
        checkpoint_path=str(checkpoint.resolve()),
        production_nondeterminism=recipe['production_nondeterminism'])
    def save():
        p=root/'report.tmp';p.write_text(json.dumps(report,indent=2)+'\n');p.replace(root/'report.json')
    save()
    child=None
    try:
        with (root/'training.log').open('x') as log:
            child=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            for line in child.stdout:
                log.write(line);log.flush();print(line,end='',flush=True)
                if line.startswith('sweep='):
                    fields=dict(re.findall(r'(\w+)=(\S+)',line))
                    report['records'].append(dict(sweep=int(fields['sweep']),train_rmse=float(fields['train_rmse']),
                        kept=int(fields['kept']),seconds=float(fields['seconds'])))
                    save()
            code=child.wait()
        if code:raise RuntimeError('native sampler exited '+str(code))
        if len(report['records'])!=128 or report['records'][-1]['sweep']!=128 or report['records'][-1]['kept']!=112:
            raise ValueError('fixed sample budget incomplete')
        values=np.fromfile(predictions,dtype='<f4')
        if len(values)!=manifest['qualifying_count'] or not np.isfinite(values).all():raise ValueError('invalid predictions')
        if digest(data)!=report['data_sha256']:raise ValueError('training data changed')
        report.update(state='complete',prediction_count=len(values),predictions_sha256=digest(predictions),
            checkpoint_sha256=digest(checkpoint),prediction_min=float(values.min()),prediction_max=float(values.max()),
            elapsed_seconds=time.monotonic()-started)
        save()
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.terminate();child.wait(timeout=15)
        report.update(state='failed',error=repr(error));save();raise


if __name__=='__main__':main()
