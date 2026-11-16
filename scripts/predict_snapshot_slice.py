#!/usr/bin/env python3
"""Infer one complete-user sequence-length slice in a fresh process."""
import argparse
import dataclasses
import json
from pathlib import Path
import time
from production_neural import digest,find_recipe,load_trainer,validate_dataset


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ['producer','dataset-manifest','output']:
        ap.add_argument('--'+name,required=True)
    for name in ['epoch','lower','upper']:
        ap.add_argument('--'+name,type=int,required=True)
    ap.add_argument('--device',choices=['cpu','mps'],default='mps')
    ap.add_argument('--token-multiplier',type=float,default=.5)
    a=ap.parse_args();start=time.monotonic();producer=Path(a.producer)
    source=json.loads((producer/'report.json').read_text())
    entry=json.loads((producer/'snapshots/report.json').read_text())['snapshots'][str(a.epoch)]
    if digest(entry['path'])!=entry['sha256']:raise ValueError('checkpoint changed')
    registry=json.loads(Path('experiments/QUAL-A2h-registry.json').read_text())
    recipe=find_recipe(registry,Path(source['column']).name)
    trainer=load_trainer(recipe,producer/'source');trainer.torch.set_num_threads(1)
    import numpy as np
    data,manifest=validate_dataset(a.dataset_manifest);d=trainer.load(data)
    if source['data_sha256']!=manifest['outputs'][data.name]:raise ValueError('query data differs')
    if not np.all(d['probe']['rating']==0):raise ValueError('queries must have unknown labels')
    saved=trainer.torch.load(entry['path'],map_location='cpu',weights_only=True)
    if saved['epoch']!=a.epoch or saved['config']!=source['config']:raise ValueError('checkpoint identity differs')
    c=trainer.Config(**saved['config']);c=dataclasses.replace(c,device=a.device,
        batch_tokens=max(1,int(c.batch_tokens*a.token_multiplier)))
    counts=np.bincount(d['probe']['user'],minlength=d['users'])
    lengths=trainer.capped_lengths(d,c,counts)
    selected=(counts>0)&(lengths>=a.lower)&(lengths<a.upper)
    rows=np.flatnonzero(selected[d['probe']['user']]).astype('<u4')
    root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
    if len(rows):
        model=trainer.SequenceModel(saved['items'],c,saved['weeks']).to(a.device)
        model.load_state_dict(saved['state'])
        if a.device=='mps':
            original=trainer.prefetched
            def bounded(*args,**kwargs):
                batches=0
                for group,batch in original(*args,**kwargs):
                    trainer.torch.mps.synchronize();trainer.torch.mps.empty_cache();batches+=1
                    if batches%200==0:
                        print(json.dumps(dict(lower=a.lower,upper=a.upper,batches=batches,
                            driver_mb=trainer.torch.mps.driver_allocated_memory()/2**20)),flush=True)
                    yield group,batch
            trainer.prefetched=bounded
        # Every query for the selected users is retained. Other users never
        # interact through attention or history encoding.
        pred=trainer.predict(model,d,queries=d['probe'][rows])
    else:pred=np.empty(0,dtype='<f4')
    if pred.shape!=(len(rows),) or not np.isfinite(pred).all():raise ValueError('invalid slice prediction')
    rows.tofile(root/'rows.u32');pred.astype('<f4').tofile(root/'predictions.f32')
    report=dict(state='complete',kind='complete_user_snapshot_slice',epoch=a.epoch,
        lower=a.lower,upper=a.upper,rows=len(rows),users=int(selected.sum()),
        checkpoint_sha256=entry['sha256'],data_sha256=source['data_sha256'],
        dataset_manifest_sha256=digest(a.dataset_manifest),qualifying_labels_read=False,
        token_multiplier=a.token_multiplier,device=a.device,
        rows_sha256=digest(root/'rows.u32'),predictions_sha256=digest(root/'predictions.f32'),
        seconds=time.monotonic()-start)
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
