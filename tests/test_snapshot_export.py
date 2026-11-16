import dataclasses
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from production_neural import digest,extract_sources,load_trainer,train_fixed


def worker(directory):
    import numpy as np
    from test_joint import dataset
    root=Path(directory);registry=json.loads((ROOT/'experiments/QUAL-A2h-registry.json').read_text())
    recipe=registry['recipes'][0];producer=root/'producer';producer.mkdir()
    extract_sources(recipe,registry,ROOT/'experiments/QUAL-neural-sources.zip',producer/'source')
    trainer=load_trainer(recipe,producer/'source');trainer.torch.set_num_threads(1)
    path=root/'train-probe-qualifying.bin';dataset(path,correlated_train=True);d=trainer.load(path)
    q=d['probe'].copy();q['rating']=0
    offset=sys.modules['gbdt_blend'].HEADER.itemsize+(d['users']+1)*8+len(d['train_ratings'])*5
    with path.open('r+b') as f:f.seek(offset);q.tofile(f)
    d=trainer.load(path)
    manifest=dict(state='complete',purpose='production_only_unlabeled_queries',outputs={path.name:digest(path)})
    manifest_path=root/'manifest.json';manifest_path.write_text(json.dumps(manifest))
    c=trainer.Config(rank=4,hidden=8,batch_users=32,nnz_budget=512,dropout=0.,device='cpu',
                     d_model=8,heads=2,layers=1,ff=16,seq_len=16,max_targets=4,
                     batch_tokens=256,pad_width=8,pad_rows=8)
    snapshots=producer/'snapshots';snapshots.mkdir();captures={};predictions=[];records=None
    for epoch in [1,2]:
        prefix=root/f'fit{epoch}';_,p,r=train_fixed(trainer,d,c,epoch,prefix)
        checkpoint=snapshots/f'e{epoch}.pt';shutil.copyfile(str(prefix)+'.pt',checkpoint)
        captures[str(epoch)]=dict(path=str(checkpoint),sha256=digest(checkpoint))
        predictions.append(p.astype('f8'));records=r['records']
    (snapshots/'report.json').write_text(json.dumps(dict(snapshots=captures)))
    (producer/'model.progress.json').write_text(json.dumps(dict(records=records)))
    source=dict(state='training',column=recipe['column'],column_sha256=recipe['column_sha256'],
                fixed_epochs=2,config=dataclasses.asdict(c),archived_config=dataclasses.asdict(c),
                dataset_manifest_sha256=digest(manifest_path),data_sha256=digest(path),sources=recipe['sources'])
    (producer/'report.json').write_text(json.dumps(source))
    reference=root/'reference.f32';expected=((predictions[0]+predictions[1])/2).astype('<f4');expected.tofile(reference)
    ref=dict(kind='snapshot_reference',epochs=[1,2],weights='equal',path=str(reference),
             column_sha256=digest(reference),parent_retained_column_sha256=recipe['column_sha256'],
             data_sha256=recipe['original_data_sha256'],config=dataclasses.asdict(c))
    refpath=root/'reference.json';refpath.write_text(json.dumps(ref));output=root/'export'
    command=[sys.executable,str(ROOT/'scripts/predict_snapshot_ensemble.py'),'--producer',str(producer),
             '--reference',str(refpath),'--dataset-manifest',str(manifest_path),'--output',str(output),
             '--device','cpu','--threads','1']
    result=subprocess.run(command,capture_output=True,text=True,cwd=ROOT)
    assert result.returncode==0,result.stdout+result.stderr
    np.testing.assert_array_equal(np.fromfile(output/'model.f32','<f4'),expected)
    report=json.loads((output/'report.json').read_text());assert report['state']=='complete'
    assert report['metric'] is None and report['qualifying_labels_read'] is False
    bundle=trainer.torch.load(output/'model.pt',map_location='cpu',weights_only=True)
    assert bundle['epochs']==[1,2] and len(bundle['states'])==2
    component=producer/'inference-cache/e1'
    cachecommand=command.copy();cachecommand[cachecommand.index('--output')+1]=str(component)
    result=subprocess.run(cachecommand+['--export-component','1','--inference-token-multiplier','0.5'],capture_output=True,text=True,cwd=ROOT)
    assert result.returncode==0,result.stdout+result.stderr
    cached=json.loads((component/'report.json').read_text())
    assert cached['kind']=='snapshot_component' and cached['training_requested'] is False
    assert cached['column_sha256'] is None
    np.testing.assert_allclose(np.fromfile(component/'model.f32','<f4'),predictions[0],rtol=0,atol=1e-6)
    # Complete-user slicing must preserve the same per-user target context.
    assembled=np.empty(len(expected),dtype='<f4');covered=np.zeros(len(expected),dtype=bool)
    for lower,upper in [(0,16),(16,17)]:
        piece=root/f'slice-{lower}'
        result=subprocess.run([sys.executable,str(ROOT/'scripts/predict_snapshot_slice.py'),
            '--producer',str(producer),'--dataset-manifest',str(manifest_path),'--output',str(piece),
            '--epoch','1','--lower',str(lower),'--upper',str(upper),'--device','cpu',
            '--token-multiplier','0.5'],capture_output=True,text=True,cwd=ROOT)
        assert result.returncode==0,result.stdout+result.stderr
        rows=np.fromfile(piece/'rows.u32','<u4');assert not covered[rows].any()
        assembled[rows]=np.fromfile(piece/'predictions.f32','<f4');covered[rows]=True
    assert covered.all()
    np.testing.assert_allclose(assembled,predictions[0],rtol=0,atol=1e-6)
    reuse=command.copy();reuse[reuse.index('--output')+1]=str(root/'cached-export')
    result=subprocess.run(reuse,capture_output=True,text=True,cwd=ROOT)
    assert result.returncode==0,result.stdout+result.stderr
    np.testing.assert_allclose(np.fromfile(root/'cached-export/model.f32','<f4'),expected,rtol=0,atol=1e-6)
    with (component/'model.f32').open('ab') as f:f.write(b'changed')
    reuse[reuse.index('--output')+1]=str(root/'tampered-cache-export')
    result=subprocess.run(reuse,capture_output=True,text=True,cwd=ROOT)
    assert result.returncode!=0 and 'component inference cache differs' in result.stderr
    # Remove only this test cache to exercise the independent checkpoint guard.
    shutil.rmtree(producer/'inference-cache')
    with (snapshots/'e1.pt').open('ab') as f:f.write(b'changed')
    bad=command.copy();bad[bad.index('--output')+1]=str(root/'bad')
    result=subprocess.run(bad,capture_output=True,text=True,cwd=ROOT)
    assert result.returncode!=0 and 'checkpoint changed' in result.stderr


class SnapshotTests(unittest.TestCase):
    def test_complete_epoch_export_matches_fixed_mean_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            r=subprocess.run([sys.executable,__file__,'--worker',directory],capture_output=True,text=True,cwd=ROOT)
            self.assertEqual(r.returncode,0,r.stdout+r.stderr)

    def test_live_capture_pairs_checkpoint_with_published_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);producer=root/'producer';producer.mkdir()
            (producer/'model.pt').write_bytes(b'checkpoint')
            (producer/'model.progress.json').write_text(json.dumps(dict(selected_epoch=1)))
            (producer/'report.json').write_text(json.dumps(dict(state='complete')))
            out=root/'capture'
            r=subprocess.run([sys.executable,str(ROOT/'scripts/capture_epoch_checkpoints.py'),
                '--producer',str(producer),'--output',str(out),'--epochs','1'],capture_output=True,text=True,cwd=ROOT)
            self.assertEqual(r.returncode,0,r.stdout+r.stderr)
            report=json.loads((out/'report.json').read_text());self.assertEqual(report['state'],'complete')
            self.assertEqual(report['snapshots']['1']['sha256'],digest(producer/'model.pt'))
            self.assertEqual((out/'e1.pt').read_bytes(),b'checkpoint')


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--worker':worker(sys.argv[2])
    else:unittest.main()
