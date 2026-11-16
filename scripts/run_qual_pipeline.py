#!/usr/bin/env python3
"""Run predeclared qualifying assembly/inference/audit when a base fit finishes."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for flag in ['name','producer','models-dir','base-bank','base-qual','bank','output','audit-output','dataset-manifest']:
        ap.add_argument('--'+flag,required=True)
    ap.add_argument('--reference-column');ap.add_argument('--row-map')
    a=ap.parse_args();root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    report_path=root/'pipeline-report.json'
    if report_path.exists():raise ValueError('pipeline report exists')
    report=dict(state='waiting',config=vars(a),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),stages=[])

    def status(state):
        report['state']=state;temp=root/'pipeline-report.tmp'
        temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(report_path)
        print(json.dumps(dict(state=state)),flush=True)

    def run(stage,strategy,command):
        name=a.name+'-'+stage;status(stage)
        subprocess.run([sys.executable,'scripts/experiment.py','--name',name,'--strategy',strategy,'--']+
                       [sys.executable]+command,check=True)
        report['stages'].append(name)

    status('waiting')
    deadline=time.monotonic()+12*3600
    while time.monotonic()<deadline:
        path=Path(a.producer)/'report.json';mp=Path(a.models_dir)/'report.json'
        producer=json.loads(path.read_text()) if path.exists() else {}
        models=json.loads(mp.read_text()) if mp.exists() else {}
        if producer.get('state')=='complete' and models.get('state')=='models_ready':break
        if producer.get('state') in ['failed','interrupted_for_resource_rescheduling']:
            raise ValueError('base producer failed or was interrupted')
        time.sleep(15)
    else:raise TimeoutError('producer/model wait expired')
    qual=str(root/'qual');final=str(root/'final')
    command=['scripts/append_qual_column.py','--base-bank',a.base_bank,'--base-qual',a.base_qual,
             '--bank',a.bank,'--local-report',str(Path(a.producer)/'report.json'),
             '--local-column',str(Path(a.producer)/'model.f32'),'--dataset-manifest',a.dataset_manifest,
             '--output',qual]
    if a.reference_column:command+=['--reference-column',a.reference_column]
    if a.row_map:command+=['--row-map',a.row_map]
    run('append','Verified full-training base insertion; no labels, exact schema and row-map proof',command)
    run('predict','Frozen-model qualifying inference only; verify inputs and freeze prediction hash before audit',
        ['scripts/predict_blender_models.py','--models-dir',a.models_dir,'--qual',qual,
         '--output',final,'--device','cpu','--threads','2'])
    run('audit','Postfreeze recovered-label audit only; no fitting, source split discrepancy explicitly retained',
        ['scripts/audit_qualifying.py','--model-dir',final,'--output-dir',a.audit_output])
    audit=json.loads((Path(a.audit_output)/'report.json').read_text())
    report['audit']=audit;status('target_met' if audit['both_source_groups_below_target'] else 'complete_target_unmet')


if __name__=='__main__':main()
