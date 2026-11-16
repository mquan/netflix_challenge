"""Production fitting must be independent of qualifying feature values."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'tests'))
from test_joint import dataset
from gbdt_blend import load,blend_fold
from streaming_blend import digest,grouped_statistics
from fit_qual_blend import nn_features
from feature_adaptation import relative_unrated
from round8_nn_blend import net
from streaming_blend import predict_ridge


class FinalFitTests(unittest.TestCase):
    def test_weighted_training_uses_verified_weights_without_query_inputs(self):
        from fit_qual_blend import solve
        from streaming_nn_blend import train_device
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);data=root/'dev.bin';dataset(data,correlated_train=True)
            d=load(data);y=d['probe']['rating'].astype('f8');users=d['probe']['user']
            rng=np.random.default_rng(28)
            p=(y[:,None]+rng.normal(0,.4,(len(y),2))).astype('f4')
            raw=np.column_stack([p,rng.normal(size=(len(y),18))]).astype('f4')
            context=np.column_stack([np.ones(len(y)),rng.normal(size=(len(y),4))]).astype('f4')
            meta=root/'meta.npz';np.savez(meta,context=context)
            weight=1+3*(y<3);wp=root/'weights.npy';np.save(wp,weight)
            bank=str(root/'bank');np.save(bank+'.x.npy',raw)
            groups=blend_fold(users)*2+(blend_fold(users.astype('u8')*np.uint64(7)+np.uint64(1))==0)
            stats=grouped_statistics(p,context,y,groups,chunks=16,sample_weights=weight)
            np.savez(bank+'.stats.npz',gram=stats[0],rhs=stats[1],count=stats[2])
            record=dict(config=dict(data=str(data),meta=str(meta),sample_weights=str(wp)),
                        data_sha256=digest(data),meta_sha256=digest(meta),sample_weights_sha256=digest(wp),
                        columns=2,extra_meta_names=[])
            Path(bank+'.inputs.json').write_text(json.dumps(record))
            output=root/'models'
            command=[sys.executable,str(ROOT/'scripts/train_development_blender.py'),
                     '--bank',bank,'--output',str(output),'--device','cpu','--threads','1',
                     '--seeds','2026','--nn-only']
            result=subprocess.run(command,capture_output=True,text=True,cwd=ROOT)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            report=json.loads((output/'report.json').read_text())
            self.assertEqual(report['state'],'models_ready')
            self.assertEqual(report['sample_weights_sha256'],digest(wp))
            expected=solve(stats,np.ones(10,dtype=bool))
            np.testing.assert_array_equal(np.load(output/'ridge-weights.npz')['full'],expected)
            saved=torch.load(output/'nn-2026.pt',map_location='cpu',weights_only=True)
            torch.set_num_threads(1)
            reference,_=train_device(np.load(output/'nn-normalized.npy'),
                y-predict_ridge(p,context,expected),SimpleNamespace(**saved['config']),
                2026,saved['epochs'],sample_weights=weight)
            for key,value in reference.state_dict().items():
                self.assertTrue(torch.equal(value,saved['state'][key]),key)
            np.save(wp,weight*2)
            command[command.index('--output')+1]=str(root/'tampered')
            result=subprocess.run(command,capture_output=True,text=True,cwd=ROOT)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('training weights changed',result.stderr)

    def test_qualifying_changes_do_not_change_fitted_models_or_stopping(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);data=root/'dev.bin';dataset(data,correlated_train=True)
            d=load(data);y=d['probe']['rating'].astype('f8');users=d['probe']['user']
            rng=np.random.default_rng(12)
            p=(y[:,None]+rng.normal(0,.3,(len(y),2))).astype('f4')
            raw=np.column_stack([p,rng.normal(size=(len(y),18))]).astype('f4')
            raw[:,2]=np.log1p(20+users);raw[:,-4]=np.log1p(9)
            c=np.column_stack([np.ones(len(y)),rng.normal(size=(len(y),4))]).astype('f4')
            meta=root/'meta.npz';np.savez(meta,context=c)
            bank=str(root/'bank')
            np.save(bank+'.x.npy',raw)
            groups=blend_fold(users)*2+(blend_fold(users.astype('u8')*np.uint64(7)+np.uint64(1))==0)
            gram,rhs,count=grouped_statistics(p,c,y,groups,chunks=16)
            np.savez(bank+'.stats.npz',gram=gram,rhs=rhs,count=count)
            record=dict(config=dict(data=str(data),meta=str(meta),metric_label='Synthetic fixture'),
                        data_sha256=digest(data),meta_sha256=digest(meta),columns=2,names=['p0','p1'],
                        extra_meta_names=['movie_age','user_frac1','user_frac5','log_unrated_n',
                                          'log_user_days','same_day_ratio','item_std'])
            Path(bank+'.inputs.json').write_text(json.dumps(record))
            reports=[];models=[];trees=[];scalers=[]
            for variant in range(2):
                qual=str(root/f'qual{variant}')
                qr=rng.normal(3+variant,1,(23,20)).astype('f4')
                qc=np.column_stack([np.ones(23),rng.normal(size=(23,4))]).astype('f4')
                np.save(qual+'.x.npy',qr);np.save(qual+'.context.npy',qc)
                q=dict(state='complete',qualifying_labels_read=False,rows=23,
                       bank_sha256=digest(bank+'.inputs.json'),x_sha256=digest(qual+'.x.npy'),
                       context_sha256=digest(qual+'.context.npy'))
                Path(qual+'.json').write_text(json.dumps(q))
                output=root/f'run{variant}'
                result=subprocess.run([sys.executable,str(ROOT/'scripts/fit_qual_blend.py'),
                    '--bank',bank,'--qual',qual,'--output',str(output),'--device','cpu','--threads','1',
                    '--tree-seeds','2026','--nn-seeds','2026'],capture_output=True,text=True,cwd=ROOT)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                r=json.loads((output/'report.json').read_text());self.assertEqual(r['state'],'complete')
                self.assertFalse(r['qualifying_labels_read']);self.assertIsNone(r['metric'])
                self.assertEqual(len(np.fromfile(output/'qualifying-final.f32','<f4')),23)
                reports.append(r);models.append(torch.load(output/'nn-2026.pt',map_location='cpu',weights_only=True)['state'])
                trees.append((output/'tree-2026.txt').read_text())
                scalers.append(dict(np.load(output/'nn-scaler.npz')))
            self.assertEqual(reports[0]['tree_rounds'],reports[1]['tree_rounds'])
            self.assertEqual(reports[0]['nn_epochs'],reports[1]['nn_epochs'])
            self.assertEqual(trees[0],trees[1])
            for key in models[0]: self.assertTrue(torch.equal(models[0][key],models[1][key]),key)
            for key in scalers[0]: np.testing.assert_array_equal(scalers[0][key],scalers[1][key])
            # Decoupled training receives no qualifying inputs and must match
            # the already-tested fit, including its selected stopping budget.
            features=root/'features.npy';np.save(features,nn_features(raw,2))
            training=root/'training-only'
            result=subprocess.run([sys.executable,str(ROOT/'scripts/train_development_blender.py'),
                '--bank',bank,'--features',str(features),'--output',str(training),'--device','cpu',
                '--threads','1','--seeds','2026'],capture_output=True,text=True,cwd=ROOT)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            r=json.loads((training/'report.json').read_text())
            self.assertEqual(r['state'],'models_ready')
            self.assertFalse(r['qualifying_inputs_provided'])
            self.assertFalse((training/'qualifying-final.f32').exists())
            self.assertEqual(r['nn_epochs'][0],reports[0]['nn_epochs'][0])
            self.assertEqual((training/'tree-2026.txt').read_text(),trees[0])
            model=torch.load(training/'nn-2026.pt',map_location='cpu',weights_only=True)['state']
            for key in model:self.assertTrue(torch.equal(model[key],models[0][key]),key)
            inference=root/'inference'
            result=subprocess.run([sys.executable,str(ROOT/'scripts/predict_blender_models.py'),
                '--models-dir',str(training),'--qual',str(root/'qual0'),'--output',str(inference),
                '--device','cpu','--threads','1'],capture_output=True,text=True,cwd=ROOT)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            np.testing.assert_allclose(np.fromfile(inference/'qualifying-final.f32','<f4'),
                                       np.fromfile(root/'run0/qualifying-final.f32','<f4'),rtol=0,atol=5e-7)
            # The same query matrix can be represented as a verified overlay.
            qr=np.load(str(root/'qual0')+'.x.npy');np.save(root/'base-query.npy',np.delete(qr,2,axis=1))
            qr[:,2].tofile(root/'insert.f32');overlay=str(root/'overlay')
            spec=dict(base_path=str(root/'base-query.npy'),base_sha256=digest(root/'base-query.npy'),
                      column_path=str(root/'insert.f32'),column_sha256=digest(root/'insert.f32'),insert_at=2,shape=list(qr.shape))
            Path(overlay+'.matrix.json').write_text(json.dumps(spec))
            oq=json.loads(Path(str(root/'qual0')+'.json').read_text());oq.pop('x_sha256')
            oq.update(matrix_layout='column_insertion',composite_spec_sha256=digest(overlay+'.matrix.json'))
            Path(overlay+'.json').write_text(json.dumps(oq))
            Path(overlay+'.context.npy').write_bytes(Path(str(root/'qual0')+'.context.npy').read_bytes())
            result=subprocess.run([sys.executable,str(ROOT/'scripts/predict_blender_models.py'),
                '--models-dir',str(training),'--qual',overlay,'--output',str(root/'overlay-predictions'),
                '--device','cpu','--threads','1'],capture_output=True,text=True,cwd=ROOT)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            np.testing.assert_allclose(np.fromfile(root/'overlay-predictions/qualifying-final.f32','<f4'),
                                       np.fromfile(inference/'qualifying-final.f32','<f4'),rtol=0,atol=5e-7)
            nntraining=root/'nn-training'
            result=subprocess.run([sys.executable,str(ROOT/'scripts/train_development_blender.py'),
                '--bank',bank,'--features',str(features),'--output',str(nntraining),'--device','cpu',
                '--threads','1','--seeds','2026','--nn-only'],capture_output=True,text=True,cwd=ROOT)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            result=subprocess.run([sys.executable,str(ROOT/'scripts/predict_blender_models.py'),
                '--models-dir',str(nntraining),'--qual',str(root/'qual0'),'--output',str(root/'nn-inference'),
                '--device','cpu','--threads','1'],capture_output=True,text=True,cwd=ROOT)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            np.testing.assert_allclose(np.fromfile(root/'nn-inference/qualifying-final.f32','<f4'),
                                       np.clip(np.fromfile(root/'run0/nn.f32','<f4'),1,5),rtol=0,atol=5e-7)
            # Extending immutable NN models must reproduce a joint full fit.
            extension=root/'tree-extension'
            result=subprocess.run([sys.executable,str(ROOT/'scripts/extend_blender_with_trees.py'),
                '--models-dir',str(nntraining),'--output',str(extension),'--threads','1'],
                capture_output=True,text=True,cwd=ROOT)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            er=json.loads((extension/'report.json').read_text())
            self.assertEqual(er['state'],'models_ready')
            self.assertEqual(er['outputs']['nn-2026.pt'],json.loads((nntraining/'report.json').read_text())['outputs']['nn-2026.pt'])
            result=subprocess.run([sys.executable,str(ROOT/'scripts/predict_blender_models.py'),
                '--models-dir',str(extension),'--qual',str(root/'qual0'),'--output',str(root/'extended-inference'),
                '--device','cpu','--threads','1'],capture_output=True,text=True,cwd=ROOT)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            np.testing.assert_allclose(np.fromfile(root/'extended-inference/qualifying-final.f32','<f4'),
                                       np.fromfile(root/'run0/qualifying-final.f32','<f4'),rtol=0,atol=5e-7)
            relative=root/'relative-training'
            result=subprocess.run([sys.executable,str(ROOT/'scripts/train_development_blender.py'),
                '--bank',bank,'--features',str(features),'--output',str(relative),'--device','cpu',
                '--threads','1','--seeds','2026','--nn-only','--relative-unrated'],capture_output=True,text=True,cwd=ROOT)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            result=subprocess.run([sys.executable,str(ROOT/'scripts/predict_blender_models.py'),
                '--models-dir',str(relative),'--qual',str(root/'qual0'),'--output',str(root/'relative-inference'),
                '--device','cpu','--threads','1'],capture_output=True,text=True,cwd=ROOT)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            saved=torch.load(relative/'nn-2026.pt',map_location='cpu',weights_only=True)
            model=net(qr.shape[1]+1,128,2,.2);model.load_state_dict(saved['state']);model.eval()
            qraw=np.load(str(root/'qual0')+'.x.npy');f=nn_features(qraw,2).astype('f8')
            index,value=relative_unrated(f,record);f[:,index]=value
            scaler=np.load(relative/'nn-scaler.npz')
            scaled=((np.clip(f,scaler['lo'],scaler['hi'])-scaler['mean'])/scaler['std']).astype('f4')
            with torch.no_grad():correction=model(torch.from_numpy(scaled)).squeeze(1).numpy()
            base=predict_ridge(qraw[:,:2],np.load(str(root/'qual0')+'.context.npy'),np.load(relative/'ridge-weights.npz')['full'])
            np.testing.assert_allclose(np.fromfile(root/'relative-inference/qualifying-final.f32','<f4'),
                                       np.clip(base+correction,1,5),rtol=0,atol=5e-7)
            Path(bank+'.x.npy').unlink()
            result=subprocess.run([sys.executable,str(ROOT/'scripts/predict_blender_models.py'),
                '--models-dir',str(relative),'--qual',str(root/'qual0'),'--output',str(root/'portable-inference'),
                '--device','cpu','--threads','1'],capture_output=True,text=True,cwd=ROOT)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            np.testing.assert_array_equal(np.fromfile(root/'portable-inference/qualifying-final.f32','<f4'),
                                          np.fromfile(root/'relative-inference/qualifying-final.f32','<f4'))


if __name__=='__main__': unittest.main()
