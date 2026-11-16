#!/usr/bin/env python3
"""Export a predeclared snapshot average from a shared fixed-budget fit prefix.

Each component is a completed, hashed epoch checkpoint. Training may continue
beyond the final component; this exporter never fits or scores query labels.
"""
import argparse
import dataclasses
import json
from pathlib import Path
import time
import subprocess
import sys
from production_neural import digest,find_recipe,load_trainer,validate_dataset


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--producer',required=True);ap.add_argument('--reference',required=True)
    ap.add_argument('--dataset-manifest',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--registry',default='experiments/QUAL-A2h-registry.json')
    ap.add_argument('--device',choices=['cpu','mps'],default='mps');ap.add_argument('--threads',type=int,default=1)
    ap.add_argument('--export-component',type=int,help='cache one predeclared checkpoint; not an auditable blend/base')
    ap.add_argument('--inference-token-multiplier',type=float,default=1.,help='batch packing only; history/weights unchanged')
    ap.add_argument('--resume',action='store_true',help='reuse verified slices from the identical unfinished inference plan')
    a=ap.parse_args();start=time.monotonic();producer=Path(a.producer)
    if not 0<a.inference_token_multiplier<=1:raise ValueError('invalid inference batch multiplier')
    reference=json.loads(Path(a.reference).read_text());epochs=reference['epochs']
    if a.export_component is not None:
        if a.export_component not in epochs:raise ValueError('component was not predeclared')
        epochs=[a.export_component]
    if reference['kind']!='snapshot_reference' or reference['weights']!='equal':raise ValueError('invalid reference')
    if digest(reference['path'])!=reference['column_sha256']:raise ValueError('development reference changed')
    source=json.loads((producer/'report.json').read_text())
    if reference['config']!=source.get('archived_config',source['config']):
        raise ValueError('reference and production learner configurations differ')
    if max(epochs)>source['fixed_epochs']:raise ValueError('snapshot exceeds the fixed production budget')
    registry=json.loads(Path(a.registry).read_text());recipe=find_recipe(registry,Path(source['column']).name)
    if source['column_sha256']!=reference['parent_retained_column_sha256'] or source['column_sha256']!=recipe['column_sha256']:
        raise ValueError('different base-model family')
    if reference['data_sha256']!=recipe['original_data_sha256']:raise ValueError('development split differs')
    data,manifest=validate_dataset(a.dataset_manifest)
    if source['dataset_manifest_sha256']!=digest(a.dataset_manifest) or source['data_sha256']!=manifest['outputs'][data.name]:
        raise ValueError('production query order differs')
    captures=json.loads((producer/'snapshots/report.json').read_text())['snapshots']
    selected=[]
    for epoch in epochs:
        entry=captures[str(epoch)]
        if digest(entry['path'])!=entry['sha256']:raise ValueError('captured checkpoint changed')
        selected.append(entry)
    history=json.loads((producer/'model.progress.json').read_text())['records']
    if len(history)<max(epochs):raise ValueError('training prefix incomplete')
    trainer=load_trainer(recipe,producer/'source');trainer.torch.set_num_threads(a.threads)
    if a.device=='mps':
        original_prefetched=trainer.prefetched
        def bounded_prefetched(*args,**kwargs):
            batches=0
            for group,batch in original_prefetched(*args,**kwargs):
                # Inference creates many sequence shapes. Release unoccupied
                # allocator blocks between batches rather than retaining them
                # alongside another model's training cache.
                trainer.torch.mps.synchronize()
                trainer.torch.mps.empty_cache()
                batches+=1
                if batches%200==0:
                    print(json.dumps(dict(state='inference_batches',epoch=epoch,batches=batches,
                        mps_driver_mb=trainer.torch.mps.driver_allocated_memory()/2**20)),flush=True)
                yield group,batch
        trainer.prefetched=bounded_prefetched
    import numpy as np
    d=trainer.load(data)
    if not np.all(d['probe']['rating']==0):raise ValueError('query labels must be unknown sentinels')
    root=Path(a.output);root.mkdir(parents=True,exist_ok=a.resume)
    plan=dict(reference_sha256=digest(a.reference),epochs=epochs,data_sha256=source['data_sha256'],
        dataset_manifest_sha256=source['dataset_manifest_sha256'],device=a.device,
        token_multiplier=a.inference_token_multiplier,exporter_sha256=digest(__file__),
        slice_worker_sha256=digest(Path(__file__).with_name('predict_snapshot_slice.py')))
    planpath=root/'inference-plan.json'
    if a.resume:
        if (root/'report.json').exists() or json.loads(planpath.read_text())!=plan:
            raise ValueError('inference plan changed or already completed')
    else:planpath.write_text(json.dumps(plan,indent=2)+'\n')
    result=np.zeros(len(d['probe']));states=[];config=None;component_inference=[]
    for epoch,entry in zip(epochs,selected):
        saved=trainer.torch.load(entry['path'],map_location='cpu',weights_only=True)
        if saved['epoch']!=epoch or saved['config']!=source['config']:raise ValueError('snapshot identity differs')
        cache=producer/'inference-cache'/f'e{epoch}'
        cr=cache/'report.json'
        if a.export_component is None and cr.exists():
            cached=json.loads(cr.read_text())
            if (cached.get('state')!='complete' or cached.get('kind')!='snapshot_component' or
                cached['snapshot_epochs']!=[epoch] or cached['data_sha256']!=source['data_sha256'] or
                cached['dataset_manifest_sha256']!=source['dataset_manifest_sha256'] or
                cached['parent_retained_column_sha256']!=source['column_sha256'] or
                cached['source_snapshots'][0]['sha256']!=entry['sha256'] or
                cached['predictions_sha256']!=digest(cache/'model.f32')):
                raise ValueError('component inference cache differs')
            pred=np.fromfile(cache/'model.f32',dtype='<f4')
            component_inference.append(dict(epoch=epoch,cache_report_sha256=digest(cr),
                device=cached['inference_device'],token_multiplier=cached['inference_token_multiplier']))
        elif a.device=='mps':
            # MPSGraph retains buffers for each compiled attention shape.
            # Fresh narrow-length workers bound that cache independently of
            # the number of users, without changing any user's query tokens.
            pred=np.empty(len(result),dtype='<f4');covered=np.zeros(len(result),dtype=bool)
            worker=Path(__file__).with_name('predict_snapshot_slice.py');worker_sha=digest(worker)
            parts=[];limit=saved['config']['seq_len']+1
            for lower in range(0,limit,128):
                upper=min(lower+128,limit);part=root/'slices'/f'e{epoch}-{lower}-{upper}'
                if not (part/'report.json').exists():
                    if digest(worker)!=worker_sha:raise ValueError('slice worker changed during inference')
                    subprocess.run([sys.executable,str(worker),'--producer',a.producer,
                        '--dataset-manifest',a.dataset_manifest,'--output',str(part),'--epoch',str(epoch),
                        '--lower',str(lower),'--upper',str(upper),'--device','mps',
                        '--token-multiplier',str(a.inference_token_multiplier)],check=True)
                pr=json.loads((part/'report.json').read_text())
                if (pr['state']!='complete' or pr['epoch']!=epoch or pr['lower']!=lower or pr['upper']!=upper or
                    pr['checkpoint_sha256']!=entry['sha256'] or pr['data_sha256']!=source['data_sha256'] or
                    pr['dataset_manifest_sha256']!=source['dataset_manifest_sha256'] or
                    pr['device']!='mps' or pr['token_multiplier']!=a.inference_token_multiplier or
                    pr['qualifying_labels_read'] is not False or
                    pr['rows_sha256']!=digest(part/'rows.u32') or
                    pr['predictions_sha256']!=digest(part/'predictions.f32')):
                    raise ValueError('inference slice changed')
                rows=np.fromfile(part/'rows.u32',dtype='<u4');values=np.fromfile(part/'predictions.f32',dtype='<f4')
                if len(rows)!=len(values) or np.any(rows>=len(pred)) or covered[rows].any():
                    raise ValueError('inference slices overlap or have invalid rows')
                pred[rows]=values;covered[rows]=True
                parts.append(dict(path=str(part/'report.json'),sha256=digest(part/'report.json')))
                print(json.dumps(dict(state='slice_complete',epoch=epoch,lower=lower,upper=upper,
                    rows=len(rows))),flush=True)
            if not covered.all():raise ValueError('inference slices do not cover every query')
            component_inference.append(dict(epoch=epoch,device='mps',token_multiplier=a.inference_token_multiplier,
                slice_worker_sha256=worker_sha,slices=parts))
        else:
            c=trainer.Config(**saved['config']);c=dataclasses.replace(c,device=a.device,
                batch_tokens=max(1,int(c.batch_tokens*a.inference_token_multiplier)))
            model=trainer.SequenceModel(saved['items'],c,saved['weeks']).to(a.device)
            model.load_state_dict(saved['state']);pred=trainer.predict(model,d)
            del model
            component_inference.append(dict(epoch=epoch,device=a.device,
                token_multiplier=a.inference_token_multiplier))
        if pred.shape!=result.shape or not np.isfinite(pred).all():raise ValueError('invalid component prediction')
        result+=pred.astype(np.float64)/len(epochs)
        pred.astype('<f4').tofile(root/f'e{epoch}.f32')
        states.append(saved);config=saved['config'];del pred
        if a.device=='mps':trainer.torch.mps.empty_cache()
        print(json.dumps(dict(state='component_complete',epoch=epoch)),flush=True)
    result.astype('<f4').tofile(root/'model.f32')
    trainer.torch.save(dict(kind='snapshot_prediction_mean',epochs=epochs,states=states,
                           data_sha256=source['data_sha256']),root/'model.pt')
    component=a.export_component is not None
    report=dict(state='complete',kind='snapshot_component' if component else 'snapshot_ensemble',column=reference['path'],
                column_sha256=None if component else reference['column_sha256'],metric=None,training_requested=not component,
                parent_retained_column_sha256=source['column_sha256'],
                fixed_epochs=max(epochs),records=history[:max(epochs)],snapshot_epochs=epochs,
                training_budget_note='Completed shared prefix of fixed-budget producer; no extra fitting in exporter',
                dataset_manifest_sha256=digest(a.dataset_manifest),data_sha256=source['data_sha256'],
                predictions_sha256=digest(root/'model.f32'),checkpoint_sha256=digest(root/'model.pt'),
                prediction_count=len(result),prediction_min=float(result.min()),prediction_max=float(result.max()),
                config=config,execution_overrides=source.get('execution_overrides'),sources=source['sources'],
                inference_device=a.device,inference_token_multiplier=a.inference_token_multiplier,
                mps_empty_cache_each_batch=a.device=='mps',component_inference=component_inference,
                reference_sha256=digest(a.reference),source_snapshots=selected,
                qualifying_labels_read=False,elapsed_seconds=time.monotonic()-start)
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
