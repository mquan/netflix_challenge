"""Tree blending preserves frozen inputs and ignores shifted count features."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from test_joint import dataset
from gbdt_blend import load,blend_fold
from streaming_blend import digest,grouped_statistics


class CatBoostBlenderTests(unittest.TestCase):
    def test_ignored_query_feature_and_frozen_model_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);data=root/'data.bin';dataset(data,correlated_train=True)
            d=load(data);y=d['probe']['rating'].astype('f8');users=d['probe']['user']
            rng=np.random.default_rng(42)
            p=(y[:,None]+rng.normal(0,.5,(len(y),2))).astype('f4')
            raw=np.column_stack([p,rng.normal(size=(len(y),18))]).astype('f4')
            c=np.column_stack([np.ones(len(y)),rng.normal(size=(len(y),4))]).astype('f4')
            meta=root/'meta.npz';np.savez(meta,context=c)
            bank=str(root/'bank');np.save(bank+'.x.npy',raw)
            inner=blend_fold(users.astype('u8')*np.uint64(7)+np.uint64(1))==0
            stats=grouped_statistics(p,c,y,blend_fold(users)*2+inner,chunks=16)
            np.savez(bank+'.stats.npz',gram=stats[0],rhs=stats[1],count=stats[2])
            b=dict(config=dict(data=str(data),meta=str(meta)),columns=2,support=0,
                data_sha256=digest(data),meta_sha256=digest(meta),
                extra_meta_names=['movie_age','user_frac1','user_frac5','log_unrated_n','log_user_days','same_day_ratio','item_std'])
            Path(bank+'.inputs.json').write_text(json.dumps(b))
            def run(script,*args,ok=True):
                r=subprocess.run([sys.executable,str(ROOT/'scripts'/script),*map(str,args)],capture_output=True,text=True,cwd=ROOT)
                if ok:self.assertEqual(r.returncode,0,r.stdout+r.stderr)
                else:self.assertNotEqual(r.returncode,0)
                return r
            models=root/'neural'
            run('train_development_blender.py','--bank',bank,'--output',models,'--device','cpu','--threads','1','--nn-only','--drop-unrated','--seeds','2026')
            cats=root/'cats'
            run('catboost_blender.py','train','--neural-models',models,'--features',models/'nn-features.npy',
                '--output',cats,'--threads','1','--seeds','2026','--max-rounds','40','--patience','10')
            report=json.loads((cats/'report.json').read_text())
            self.assertEqual(report['state'],'models_ready');self.assertFalse(report['torch_loaded'])
            self.assertFalse(report['qualifying_inputs_provided'])
            qr=rng.normal(size=(23,20)).astype('f4');qc=np.ones((23,5),dtype='f4')
            predictions=[]
            for variant in range(2):
                qual=str(root/f'q{variant}');values=qr.copy();values[:,16]+=variant*1e6
                np.save(qual+'.x.npy',values);np.save(qual+'.context.npy',qc)
                Path(qual+'.json').write_text(json.dumps(dict(state='complete',qualifying_labels_read=False,rows=len(qr),
                    bank_sha256=digest(bank+'.inputs.json'),x_sha256=digest(qual+'.x.npy'),context_sha256=digest(qual+'.context.npy'))))
                neural=root/f'n{variant}'
                run('predict_blender_models.py','--models-dir',models,'--qual',qual,'--output',neural,'--device','cpu','--threads','1')
                output=root/f'out{variant}'
                run('catboost_blender.py','predict','--models',cats,'--qual',qual,'--neural-reference',neural,'--output',output,'--threads','1')
                predictions.append({kind:(output/kind/'qualifying-final.f32').read_bytes() for kind in ['cat-only','hybrid']})
            self.assertEqual(predictions[0],predictions[1])
            (cats/'cat-2026.cbm').write_bytes(b'tampered')
            r=run('catboost_blender.py','predict','--models',cats,'--qual',str(root/'q0'),
                  '--neural-reference',root/'n0','--output',root/'rejected',ok=False)
            self.assertIn('CatBoost model changed',r.stderr)


if __name__=='__main__':unittest.main()
