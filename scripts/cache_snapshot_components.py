#!/usr/bin/env python3
"""Export predeclared checkpoint components as they become available."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ['producer','reference','dataset-manifest','output','name']:
        ap.add_argument('--'+name,required=True)
    ap.add_argument('--wait-producer')
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--wait-existing-epochs',type=int,nargs='*',default=[])
    a=ap.parse_args();reference=json.loads(Path(a.reference).read_text())
    root=Path(a.output)
    if a.resume:
        previous=json.loads((root/'report.json').read_text())
        for key in ['producer','reference','dataset_manifest','output']:
            if previous['config'][key]!=vars(a)[key]:raise ValueError('different resumed cache plan')
        if previous['reference_sha256']!=sha(a.reference):raise ValueError('reference changed')
        (root/f'report.previous-{time.time_ns()}.json').write_text(json.dumps(previous,indent=2)+'\n')
    root.mkdir(parents=True,exist_ok=a.resume)
    report=dict(state='waiting',config=vars(a),reference_sha256=sha(a.reference),completed_epochs=[])

    def status(state,**extra):
        report.update(state=state,**extra);p=root/'report.tmp'
        p.write_text(json.dumps(report,indent=2)+'\n');p.replace(root/'report.json')
        print(json.dumps(dict(state=state,**extra)),flush=True)

    deadline=time.monotonic()+12*3600
    def wait_for(predicate):
        while not predicate():
            if time.monotonic()>deadline:raise TimeoutError('snapshot export wait expired')
            time.sleep(15)

    status('waiting')
    if a.wait_producer:
        path=Path(a.wait_producer)/'report.json'
        wait_for(lambda:path.exists() and json.loads(path.read_text()).get('state')=='complete')
    for epoch in reference['epochs']:
        status('waiting_checkpoint',epoch=epoch)
        path=Path(a.producer)/'snapshots/report.json'
        wait_for(lambda:path.exists() and str(epoch) in json.loads(path.read_text()).get('snapshots',{}))
        cache=Path(a.producer)/'inference-cache'/f'e{epoch}';cr=cache/'report.json'
        if not cr.exists():
            if cache.exists() or epoch in a.wait_existing_epochs:
                # A component launched by the root may still be running.
                wait_for(cr.exists)
            else:
                status('exporting',epoch=epoch)
                subprocess.run([sys.executable,'scripts/experiment.py','--name',a.name+'-e'+str(epoch),
                    '--strategy','Predeclared checkpoint inference cache; no fitting or rating labels; bounded GPU allocator',
                    '--','/usr/bin/env','PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.4',
                    'PYTORCH_MPS_LOW_WATERMARK_RATIO=0.1',sys.executable,
                    'scripts/predict_snapshot_ensemble.py','--producer',a.producer,'--reference',a.reference,
                    '--dataset-manifest',a.dataset_manifest,'--output',str(cache),
                    '--registry','experiments/QUAL-A2h-registry.json','--device','mps','--threads','1',
                    '--export-component',str(epoch),'--inference-token-multiplier','0.5'],check=True)
        cached=json.loads(cr.read_text())
        if (cached['state']!='complete' or cached['kind']!='snapshot_component' or
            cached['snapshot_epochs']!=[epoch] or cached['predictions_sha256']!=sha(cache/'model.f32')):
            raise ValueError('invalid component cache')
        report['completed_epochs'].append(epoch)
    status('complete',qualifying_labels_read=False)


if __name__=='__main__':main()
