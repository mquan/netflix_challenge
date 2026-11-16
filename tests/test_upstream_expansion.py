"""Exercise actual artifact collection across the two expansion workers."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import upstream_expansion as expansion


class ExpansionCollection(unittest.TestCase):
    def collect(self, mismatch=False, relocate=False):
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
                destination=tracked/('m1' if relocate else machine)
                destination.mkdir(parents=True,exist_ok=True)
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


    def test_remaining_remote_outputs_can_be_trained_and_collected_on_m1(self):
        _,_,predictions=self.collect(relocate=True)
        self.assertEqual(set(predictions),{'original','new-a','new-b'})

    def test_takeover_keeps_the_same_six_models(self):
        jobs=expansion.execution_jobs('m1',take_over=True)
        self.assertEqual(set(jobs),set(expansion.NEW))
        self.assertEqual(len(jobs),len(set(jobs)))
        with self.assertRaisesRegex(ValueError,'Local takeover'):
            expansion.execution_jobs('m3',take_over=True)

    def test_duplicate_execution_is_not_silently_selected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for machine in ['m1','m3']:
                expansion.u.write(root/machine/'same.json',{})
            with patch.object(expansion,'TRACKED',root):
                with self.assertRaisesRegex(ValueError,'Duplicate expansion'):
                    expansion.report_location('same')

    def test_local_completion_never_waits_for_or_fetches_m3(self):
        with patch.object(expansion,'collect_expansion',return_value=None),patch.object(expansion.x,'git') as git:
            with self.assertRaisesRegex(ValueError,'missing a completed'):
                expansion.watch(False,local_only=True)
            git.assert_not_called()


class PublicationRetry(unittest.TestCase):
    def test_concurrent_ancestor_push_retries_without_merging(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            with patch.object(expansion,'ROOT',root),patch.object(expansion,'TRACKED',root/'tracked'),patch.object(expansion.x,'publish',side_effect=[RuntimeError('concurrent ref update'),None]) as publish,patch.object(expansion.x,'git',return_value=SimpleNamespace(returncode=0)) as git,patch.object(expansion.time,'sleep'):
                expansion.status('m1','diagnostic',pushing=True)
                self.assertEqual(publish.call_count,2)
                self.assertTrue(any(call.args[:2]==('merge-base','--is-ancestor') for call in git.call_args_list))

    def test_divergent_remote_history_is_not_retried_or_merged(self):
        from types import SimpleNamespace
        def git(*args,**kwargs):
            return SimpleNamespace(returncode=1 if args[0]=='merge-base' else 0)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            with patch.object(expansion,'ROOT',root),patch.object(expansion,'TRACKED',root/'tracked'),patch.object(expansion.x,'publish',side_effect=RuntimeError('divergence')) as publish,patch.object(expansion.x,'git',side_effect=git),patch.object(expansion.time,'sleep'):
                with self.assertRaisesRegex(RuntimeError,'divergence'):
                    expansion.status('m1','diagnostic',pushing=True)
                self.assertEqual(publish.call_count,1)


if __name__=='__main__':unittest.main()
