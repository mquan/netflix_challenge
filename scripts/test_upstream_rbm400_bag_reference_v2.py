"""Bag arithmetic and CPU replay fixtures; no production/MPS admission.

The full fixture retains real widths, five saved pipelines and all seed slots.
Source admission, training and device moves are mocked in a temporary directory.
"""
import copy
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import upstream_rbm400_bag_reference_v2 as bag
import verify_upstream_rbm400_bag_reference_v2 as verification
import test_verify_upstream_rbm400_neural_study_reference_v2 as fixtures


class BagTests(unittest.TestCase):
    def test_original_float32_seed_division_and_zero_slot_denominator(self):
        baseline=np.array([1.04,3.123456789,4.94],dtype='<f8')
        a=np.array([-.1,.123456789,.2],dtype='<f4')
        b=np.array([.3,-.23456789,.1],dtype='<f4')
        actual=bag.fold_values(baseline,[a,None,b])
        expected=np.array([min(4.95,max(1.05,float(base)+float(np.float32(x/np.float32(3)))+
                                   float(np.float32(y/np.float32(3)))))
                           for base,x,y in zip(baseline,a,b)],dtype='<f4')
        self.assertEqual(actual.tobytes(),expected.tobytes())
        self.assertEqual(bag.fold_values(baseline,[None]*3).tobytes(),np.clip(baseline,1.05,4.95).astype('<f4').tobytes())

    def test_five_full_float32_folds_accumulate_in_float64(self):
        rng=np.random.default_rng(128)
        folds=[rng.uniform(1.05,4.95,35).astype('<f4') for _ in range(5)]
        expected=np.array([sum(float(f[i]) for f in folds)/5 for i in range(35)],dtype='<f4')
        self.assertEqual(bag.bag_values(folds).tobytes(),expected.tobytes())
        with self.assertRaises(ValueError):bag.bag_values(folds[:4])
        with self.assertRaises(ValueError):bag.fold_values(folds[0].astype('<f8'),folds[:2])
        with self.assertRaises(ValueError):bag.fold_values(folds[0].astype('<f8'),[folds[1].astype('<f8'),None,None])

    def test_ranking_fixed_half_cannot_become_NN_provider(self):
        nn=dict(path='nn.npy',sha256='a');half=dict(path='half.npy',sha256='b')
        actual=bag.roles({'selected_known_vector':half},{'nn':nn,'fixed_half':half})
        self.assertEqual(actual['known_NN_provider'],nn)
        self.assertEqual(actual['known_ranking_vector'],half)
        with self.assertRaises(ValueError):bag.roles({'selected_known_vector':nn},{'nn':nn,'fixed_half':half})
        with self.assertRaises(ValueError):bag.roles({'selected_known_vector':half},{'nn':half,'fixed_half':half})

    def test_raw_descriptor_identity_checked_on_exceptional_exit(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'tree.f32';path.write_bytes(np.ones(7,dtype='<f4').tobytes())
            identity=bag.admission.file_identity(path.stat())
            with self.assertRaisesRegex(ValueError,'changed during'):
                with bag.raw_vector(path,dict(identity=identity),7,lambda:None):
                    replacement=path.with_suffix('.new');replacement.write_bytes(path.read_bytes());replacement.replace(path)
                    raise RuntimeError('Synthetic inference failure')

    def test_complete_saved_CPU_pipelines_replay_across_original_8192_boundary(self):
        guard=lambda **kw:None
        with fixtures.ReplayTests().complete_fixture(arms=bag.study.ARMS) as (root,results):
            for arm in bag.study.ARMS:bag.canonical.canonical(arm,guard)
            selection=bag.canonical.select(guard);arm=selection['selected_arm']
            manifest,declaration,data,_,v,receipts,_,_=bag.canonical.evidence(arm,guard)
            rows=8193;rng=np.random.default_rng(129)
            z=rng.normal(size=(rows,manifest['dimensions']['design_columns']));z[:,-1]=1
            f=rng.normal(size=(rows,manifest['dimensions']['nn_inputs']-1)).astype('<f4')
            tree=root/'tree-qual.f32';tree.write_bytes(rng.uniform(1.05,4.95,rows).astype('<f4').tobytes())
            bindings=dict(selection['source_sha256']);bindings['tree-qual.f32']=bag.execution.inputs.sha(tree)
            records=bag.study.inputs.cpu.inputs.verify_files(root,bindings,guard)
            data=copy.deepcopy(data);data['virtual_c_order_sha256']['qual']={'arm_operator_sha256':{
                arm:[hashlib.sha256(z.tobytes()).hexdigest(),hashlib.sha256(f.tobytes()).hexdigest()]}}
            source=dict(arm=arm,manifest=manifest,declaration={'source_sha256':bindings},data_manifest=data,
                receipts=receipts,source_sha256=bindings,records=records,roles=bag.roles(selection,v),
                tree_qualifying='tree-qual.f32',actual_models=v['actual_saved_models_reopened'],
                selection_sha256=bag.execution.inputs.sha(root/'experiments/upstream-own-rbm400-neural-selection-reference-v2/selection.json'))
            blocks=[]
            @contextmanager
            def query(q,declaration,records,guard,*,retain_scalar,latent):
                self.assertEqual(q,'qual');self.assertTrue(retain_scalar);self.assertEqual(latent,arm=='latent')
                class Reader:
                    prediction=np.zeros(rows,dtype='<f4')
                    def block(self,lo,hi):blocks.append((lo,hi));return z[lo:hi],f[lo:hi]
                yield Reader()
            with patch.object(bag,'ROOT',root),patch.object(bag,'REPORT',root/'bag-report'),\
                 patch.object(bag,'ART',root/'bag-art'),patch.object(bag,'ROWS',rows),\
                 patch.object(bag,'sources',return_value=source),patch.object(bag.study.inputs.context,'open_query',side_effect=query):
                done=bag.infer(guard);proof=verification.verify(guard)
                self.assertEqual(blocks,[(0,8192),(8192,8193)]*2)
                self.assertEqual(proof['state'],verification.STATE)
                self.assertEqual(proof['actual_models_reopened'],10)
                self.assertEqual(proof['rows'],rows);self.assertEqual(proof['models_fitted'],0)
                self.assertEqual(proof['known_NN_provider'],v['nn'])
                self.assertNotEqual(proof['known_NN_provider'],selection['selected_known_vector'])
                self.assertEqual(proof['qualifying_NN_provider'],done['NN'])
                self.assertTrue(proof['every_full_query_operator_raw_normalized_seed_fold_and_output_bit_verified'])
                self.assertTrue(all(row[0] is None for row in done['seed_sha256']))
                with self.assertRaises(ValueError):bag.infer(guard)
                with self.assertRaises(ValueError):verification.verify(guard)
                # Rehashing a changed output cannot substitute for actual model replay.
                (bag.REPORT/'independent-verification.json').unlink()
                path=bag.ART/'NN-bag.npy';array=np.load(path);array[-1]=np.nextafter(array[-1],np.float32(10));np.save(path,array)
                metadata=json.loads((bag.REPORT/'inference-completion.json').read_text())
                metadata['NN']['sha256']=bag.execution.inputs.sha(path)
                (bag.REPORT/'inference-completion.json').write_text(json.dumps(metadata))
                with self.assertRaisesRegex(ValueError,'NN-only bag bit'):verification.verify(guard)


if __name__=='__main__':unittest.main()
