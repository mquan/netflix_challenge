#!/usr/bin/env python3
"""Compare cached GPU predictions with CPU inference for complete user queries."""
import argparse
import dataclasses
import json
from pathlib import Path
import time
from production_neural import digest,find_recipe,load_trainer,validate_dataset


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--producer',required=True);ap.add_argument('--component',type=int,required=True)
    ap.add_argument('--dataset-manifest',required=True);ap.add_argument('--output',required=True)
    a=ap.parse_args();start=time.monotonic();producer=Path(a.producer)
    cache=producer/'inference-cache'/f'e{a.component}'
    cached=json.loads((cache/'report.json').read_text());source=json.loads((producer/'report.json').read_text())
    if cached['predictions_sha256']!=digest(cache/'model.f32'):raise ValueError('cache changed')
    entry=cached['source_snapshots'][0]
    if entry['sha256']!=digest(entry['path']):raise ValueError('checkpoint changed')
    registry=json.loads(Path('experiments/QUAL-A2h-registry.json').read_text())
    recipe=find_recipe(registry,Path(source['column']).name)
    trainer=load_trainer(recipe,producer/'source');trainer.torch.set_num_threads(1)
    import numpy as np
    data,_=validate_dataset(a.dataset_manifest);d=trainer.load(data)
    if not np.all(d['probe']['rating']==0):raise ValueError('query labels must stay unknown')
    active=np.unique(d['probe']['user']);users=active[np.linspace(0,len(active)-1,64,dtype=int)]
    # Keep every target token for the chosen users: targets interact through
    # label-free attention, so sampling isolated rows would change the inputs.
    rows=np.flatnonzero(np.isin(d['probe']['user'],users))
    saved=trainer.torch.load(entry['path'],map_location='cpu',weights_only=True)
    c=trainer.Config(**saved['config']);c=dataclasses.replace(c,device='cpu',
        batch_tokens=max(1,int(c.batch_tokens*cached['inference_token_multiplier'])))
    model=trainer.SequenceModel(saved['items'],c,saved['weeks']);model.load_state_dict(saved['state'])
    expected=trainer.predict(model,d,queries=d['probe'][rows])
    actual=np.fromfile(cache/'model.f32','<f4')[rows]
    maximum=float(np.max(np.abs(actual-expected)))
    if maximum>1e-5:raise ValueError('GPU/CPU predictions differ beyond tolerance: '+str(maximum))
    report=dict(state='passed',rows=len(rows),users=len(users),maximum_absolute_difference=maximum,
        tolerance=1e-5,cache_report_sha256=digest(cache/'report.json'),
        checkpoint_sha256=entry['sha256'],qualifying_labels_read=False,seconds=time.monotonic()-start)
    Path(a.output).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
