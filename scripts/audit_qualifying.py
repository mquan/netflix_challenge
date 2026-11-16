#!/usr/bin/env python3
"""One post-freeze audit against community-provided qualifying labels.

This program cannot fit, tune, or alter predictions. It refuses unfinished
models or a prediction hash different from the frozen production report.
The result is a local recovered-label measurement, not live Netflix scoring.
"""
import argparse
import datetime
import gzip
import hashlib
import json
from pathlib import Path
import urllib.request
import numpy as np

COMMIT='49c9b3676e6bab90fdf77e9a2da984edbfe4fe5c'
LABEL_PATH='data/qual_ratings/qual_ratings.csv.gz'
RAW_URL=f'https://raw.githubusercontent.com/gtakacs/netflix_prize/{COMMIT}/{LABEL_PATH}'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_predictions(directory):
    root=Path(directory);report=json.loads((root/'report.json').read_text())
    if report.get('state')!='complete' or report.get('qualifying_labels_read') is not False:
        raise ValueError('a complete, label-isolated production report is required')
    path=root/'qualifying-final.f32'
    expected=report.get('frozen_prediction_sha256')
    if not expected or sha(path)!=expected: raise ValueError('frozen prediction hash mismatch')
    pred=np.fromfile(path,dtype='<f4').astype(np.float64)
    if len(pred)!=report['qualifying_rows'] or not np.isfinite(pred).all(): raise ValueError('bad predictions')
    if np.any((pred<1)|(pred>5)): raise ValueError('final predictions must already be clipped')
    return pred,report


def scores(pred,ratings,is_test):
    if pred.shape!=ratings.shape or pred.shape!=is_test.shape: raise ValueError('label shape mismatch')
    if not np.all((ratings>=1)&(ratings<=5)) or not np.all(np.isin(is_test,[0,1])):
        raise ValueError('invalid label domain')
    output={}
    for value,name in [(0,'quiz'),(1,'test')]:
        mask=is_test==value
        if not mask.any(): raise ValueError('empty evaluation split')
        output[name]=dict(rows=int(mask.sum()),rmse=float(np.sqrt(np.mean((pred[mask]-ratings[mask])**2))))
    output['target_rmse']=.85725
    output['test_baseline_rmse']=.9525
    output['test_improvement_percent']=100*(1-output['test']['rmse']/.9525)
    output['target_reached']=output['test']['rmse']<=.85725
    return output


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model-dir',required=True)
    ap.add_argument('--output-dir',required=True)
    a=ap.parse_args()
    # Verify the frozen model BEFORE any request or read of target labels.
    pred,model=frozen_predictions(a.model_dir)
    if len(pred)!=2817131: raise ValueError('not the full original qualifying set')
    root=Path(a.output_dir);root.mkdir(parents=True,exist_ok=False)
    started=datetime.datetime.now(datetime.timezone.utc).isoformat()
    plan=dict(state='predictions_verified_before_label_access',started_utc=started,
              prediction_sha256=model['frozen_prediction_sha256'],model_report_sha256=sha(Path(a.model_dir)/'report.json'),
              label_source=RAW_URL,source_commit=COMMIT,
              purpose='Post-freeze recovered-label audit only; no model fitting')
    (root/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    tree=json.load(urllib.request.urlopen(f'https://api.github.com/repos/gtakacs/netflix_prize/git/trees/{COMMIT}?recursive=1'))
    entries=[x for x in tree['tree'] if x['path']==LABEL_PATH]
    if len(entries)!=1: raise ValueError('label source metadata missing')
    expected=entries[0]
    payload=urllib.request.urlopen(RAW_URL,timeout=90).read()
    blob=hashlib.sha1(f'blob {len(payload)}\0'.encode()+payload).hexdigest()
    if blob!=expected['sha'] or len(payload)!=expected['size']: raise ValueError('source label blob checksum mismatch')
    label_file=root/'qual_ratings.csv.gz';label_file.write_bytes(payload);label_file.chmod(0o444)
    with gzip.open(label_file,'rt') as stream:
        header=stream.readline().strip().split(',')
        if header!=['rating','is_test']: raise ValueError('unexpected label schema')
        labels=np.loadtxt(stream,delimiter=',',dtype=np.int16)
    if labels.shape!=(len(pred),2): raise ValueError('source label row count mismatch')
    result=scores(pred,labels[:,0],labels[:,1])
    counts=(result['quiz']['rows'],result['test']['rows'])
    if sorted(counts)!=[1408342,1408789]:
        raise ValueError('unexpected source split counts')
    result['split_names_follow_source_is_test']=True
    result['published_split_count_agreement']=counts==(1408342,1408789)
    result['split_provenance_warning']=(None if result['published_split_count_agreement'] else
        'Source is_test=0 has 1408789 rows and is_test=1 has 1408342; these counts '
        'reverse the published Quiz/Test sizes. Names follow source; no silent inversion.')
    result['both_source_groups_below_target']=all(result[k]['rmse']<=.85725 for k in ('quiz','test'))
    result.update(state='complete',source=plan,label_sha256=sha(label_file),label_git_blob_sha1=blob,
                  completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  provenance='Community-provided recovered qualifying labels from gtakacs/netflix_prize; not live or official Netflix scoring',
                  model_fitting_performed=False)
    if sha(Path(a.model_dir)/'qualifying-final.f32')!=model['frozen_prediction_sha256']:
        raise ValueError('predictions changed during audit')
    (root/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__': main()
