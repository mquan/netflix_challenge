import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import lightgbm as lgb
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from choice_affinity_column import SOURCES,run,load_bundle,predict_models,fit_models
from gbdt_blend import HEADER,PROBE,load
from headroom_choices import features,read_joint_factors
from train_holdout import digest


class ColumnTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.rng=np.random.default_rng(19)
        self.y=self.rng.integers(1,6,400)
        self.data=self.root/'data.bin';self.write_data(self.data,self.y)
        self.joint=self.root/'joint.bin'
        with self.joint.open('wb') as stream:
            np.array([0,0],dtype='<u8').tofile(stream);np.array([32,1],dtype='<u4').tofile(stream)
            for k in range(12):
                a=self.rng.normal(0,.1,40*32).astype('<f4') if k in (9,10) else np.array([],dtype='<f4')
                np.array([len(a)],dtype='<u8').tofile(stream);a.tofile(stream)
        self.names=[f'{stem}_w{w}' for w in (3,30,100000) for stem in ('n','q_dot_meany','q_dot_meanq','y_dot_meany')]
        self.params=dict(objective='regression',num_leaves=7,min_data_in_leaf=10,seed=19,num_threads=1,verbose=-1)
        self.bank=self.y[:,None]+self.rng.normal(0,.7,(400,3))
        self.bankpaths=[]
        for k in range(3):
            p=self.root/f'bank{k}.f32';self.bank[:,k].astype('<f4').tofile(p);self.bankpaths.append(str(p))
        self.bank=np.column_stack([np.fromfile(p,dtype='<f4').astype(float) for p in self.bankpaths])
        self.iterations=dict(base_tree=2,choice_tree=3)
        self.selection=self.root/'selection.json';self.selection.write_text(json.dumps(dict(selected_iterations=self.iterations)))

    def write_data(self,path,y):
        h=np.zeros(1,dtype=HEADER);h['magic']=b'NFPDAT2';h['users']=200;h['items']=40
        h['train_count']=400;h['probe_count']=400;h['implicit_count']=800;h['mean']=3;h['max_day']=4
        probe=np.zeros(400,dtype=PROBE);probe['user']=np.repeat(np.arange(200),2)
        probe['item']=np.tile([3,4],200);probe['day']=np.tile([3,4],200);probe['rating']=y
        with path.open('wb') as stream:
            for part in (h,np.arange(201,dtype='<u8')*2,np.tile([1,2],200).astype('<u2'),
                np.tile([1,2],200).astype('<u2'),np.tile([2,4],200).astype('u1'),probe,
                np.arange(201,dtype='<u8')*4,np.tile([1,2,3,4],200).astype('<u2'),
                np.tile([1,2,3,4],200).astype('<u2')):part.tofile(stream)

    def plan(self,mode,data=None):
        data=self.data if data is None else data
        sources=Path(__file__).resolve().parents[1]/'scripts'
        paths=[data,self.joint]+[sources/name for name in SOURCES]
        if mode=='train':paths += [self.selection,*map(Path,self.bankpaths)]
        p=dict(mode=mode,data=str(data),joint_model=str(self.joint),feature_names=self.names,
            libraries=dict(numpy=np.__version__,lightgbm=lgb.__version__),
            frozen={str(path):digest(path) for path in paths})
        if mode=='train':p.update(bank=self.bankpaths,bank_names=['a','b','c'],
            selection_result=str(self.selection),iterations=self.iterations,tree_params=self.params)
        path=self.root/(mode+data.stem+'-plan.json');path.write_text(json.dumps(p));return path

    def test_full_bundle_roundtrip_fixed_fit_and_changed_payload_rejection(self):
        with contextlib.redirect_stdout(io.StringIO()):
            report=run(self.plan('train'),self.root/'bundle')
        self.assertEqual(report['iterations'],self.iterations)
        d=load(self.data);q,y=read_joint_factors(self.joint,40);x,names=features(d,q,y)
        w,models=load_bundle(self.root/'bundle',['a','b','c'],names)
        wa,ma=fit_models(self.bank,x,self.y.astype(float),d['probe']['user'],self.params,self.iterations)
        np.testing.assert_array_equal(w,wa)
        a=predict_models(self.bank,x,w,models);b=predict_models(self.bank,x,wa,ma)
        for key in a:np.testing.assert_array_equal(a[key],b[key])
        with self.assertRaisesRegex(ValueError,'schema'):
            load_bundle(self.root/'bundle',['c','b','a'],names)
        with (self.root/'bundle/base_tree.txt').open('a') as f:f.write('changed')
        with self.assertRaisesRegex(ValueError,'payload changed'):
            load_bundle(self.root/'bundle',['a','b','c'],names)

    def test_real_feature_interface_removes_query_rating_field(self):
        import choice_affinity_column as module
        original=module.features
        def check(d,*args):
            self.assertEqual(d['probe'].dtype.names,('user','item','day'))
            return original(d,*args)
        from unittest.mock import patch
        changed=self.root/'changed.bin';self.write_data(changed,(self.y%5)+1)
        with patch.object(module,'features',check),contextlib.redirect_stdout(io.StringIO()):
            run(self.plan('features'),self.root/'features1')
            run(self.plan('features',changed),self.root/'features2')
        np.testing.assert_array_equal(np.load(self.root/'features1/features.npy'),np.load(self.root/'features2/features.npy'))


if __name__=='__main__':unittest.main()
