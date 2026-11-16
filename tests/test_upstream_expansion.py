"""Exercise actual artifact collection across the two expansion workers."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import upstream_expansion as expansion


class ExpansionCollection(unittest.TestCase):
    def collect(self, mismatch=False):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            labels=np.array([1,2,3,4,5],dtype=np.int8)
            users=np.arange(5,dtype=np.int32)
            reference=root/'reference/data/probex'
            reference.mkdir(parents=True)
            np.save(reference/'ratings.npy',labels)
            incoming=root/'incoming'
            incoming.mkdir()
            np.savez_compressed(incoming/'original.npz',prediction=np.full(5,3,dtype=np.float32))
            frozen=root/'control'
            frozen.mkdir()
            expansion.u.write(frozen/'fresh-only-blends.json',{'input_artifact_sha256':{'original':expansion.u.sha(incoming/'original.npz')}})
            tracked=root/'tracked'
            assignments={'m1':['new-a'],'m3':['new-b']}
            for machine,names in assignments.items():
                destination=tracked/machine
                destination.mkdir(parents=True)
                np.savez_compressed(destination/'development.npz',labels=labels,users=users)
                name=names[0]
                values=np.full(5,3,dtype=np.float32)
                np.savez_compressed(destination/(name+'.npz'),prediction=values)
                output='0'*64
                report=dict(source_commit=expansion.u.PIN,
                    artifact_sha256=expansion.u.sha(destination/(name+'.npz')),
                    development_sha256=expansion.u.sha(destination/'development.npz'),
                    prediction_bytes_sha256=expansion.x.array_sha(values),fresh_rmse=expansion.x.metric(values,labels),
                    native_npy_sha256='1'*64 if mismatch and machine=='m3' else output,
                    receipt=dict(source_commit=expansion.u.PIN,job=name,outputs={'preds_new/'+name+'.probex.npy':output}))
                expansion.u.write(destination/(name+'.json'),report)
            with patch.object(expansion,'TRACKED',tracked),patch.object(expansion,'ASSIGNMENTS',assignments),patch.object(expansion.u,'REFERENCE',root/'reference'),patch.object(expansion.x,'JOBS',['original']),patch.object(expansion.x,'LOCAL',root),patch.object(expansion.x,'REPORT',frozen):
                return expansion.collect_expansion()

    def test_two_machine_artifacts_are_verified_and_joined(self):
        labels,users,predictions=self.collect()
        self.assertEqual(set(predictions),{'original','new-a','new-b'})
        self.assertEqual(len(labels),len(users))

    def test_native_receipt_and_export_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'native output receipt'):
            self.collect(mismatch=True)


if __name__=='__main__':unittest.main()
