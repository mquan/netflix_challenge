"""Full recipe/trace/guard and lossless return checks without native training."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import upstream_remote_weekend as run


class WeekendTests(unittest.TestCase):
    def test_all_original_recipes_and_complete_epoch_lists(self):
        plan=run.read(run.PLAN)
        registry=run.read(run.ROOT/'experiments/upstream-all-pipeline-coverage/registry.json')['3']['jobs']
        self.assertEqual([j['name'] for j in plan['jobs']],['s3-rbm400pf','s3-tx300r','s3-rbm800p'])
        self.assertEqual([j['epochs_per_phase'] for j in plan['jobs']],[15,24,25])
        for job in plan['jobs']:
            self.assertEqual(job['original_registry_job'],registry[job['name']])
            self.assertEqual(set(job['original_registry_job']['inputs_from']),{'ingest','split_3'})
            self.assertEqual(job['environment']['EPSAVE'],'1')
            self.assertEqual(job['environment']['EPOCHS'],str(job['epochs_per_phase']))
            self.assertEqual(job['published_columns'],[job['name']]+[job['name']+'_ep%02d'%i for i in range(1,job['epochs_per_phase']+1)])
        self.assertEqual(sum(len(j['published_columns']) for j in plan['jobs']),67)
        run.check_files(run.ROOT,plan['controller_sha256'])

    def test_trace_requires_both_full_phases(self):
        phase='\n'.join('Epoch %02d | RMSE .9'%i for i in range(1,16))+'\n'
        run.trace_check(phase*2,15)
        for invalid in [phase,phase+phase.replace('Epoch 09','Epoch 08'),phase*2+'Epoch 01 | RMSE .9\n']:
            with self.assertRaises(ValueError):run.trace_check(invalid,15)

    def test_guard_boundaries(self):
        b={'compressor_bytes':0,'swap_used_bytes':1000}
        w={'1':{'bytes':1,'rss_bytes':1}}
        self.assertTrue(run.capacity(w,b,b,reserve=64*2**30))
        self.assertFalse(run.capacity(w,dict(b,compressor_bytes=12*2**30),b))
        self.assertFalse(run.capacity(w,dict(b,swap_used_bytes=1001+128*2**20),b))
        self.assertFalse(run.capacity(w,b,b,own={'bytes':64*2**30+1,'rss_bytes':1}))
        self.assertFalse(run.capacity({'1':{'bytes':80*2**30,'rss_bytes':1}},b,b))
        self.assertFalse(run.capacity({'1':{'bytes':1,'rss_bytes':80*2**30}},b,b))

    def test_full_training_and_factor_shapes(self):
        plan=run.read(run.PLAN)
        for job in plan['jobs']:
            self.assertEqual(run.native_shape(job['name']+'.train_3.npy',job),(98720013,))
            self.assertEqual(run.native_shape(job['name']+'.fulltrain.npy',job),(100480507,))
            self.assertEqual(run.native_shape(job['name']+'_ep01.qual.npy',job),(2817131,))
        self.assertEqual(run.native_shape('s3-rbm400pf.ifeat.train_3.npy',plan['jobs'][0]),(17770,250))
        self.assertEqual(run.native_shape('s3-rbm800p.ifeat.fulltrain.npy',plan['jobs'][2]),(17770,4000))
        with self.assertRaises(ValueError):run.native_shape('s3-tx300r.unbound.npy',plan['jobs'][1])

    def test_identity_survives_only_the_declared_exec(self):
        captured=dict(pid=20,pgid=20,started='same start',command='python bootstrap',native_command='./target/release/fit rbm original')
        self.assertTrue(run.same_child(captured,captured))
        self.assertTrue(run.same_child(captured,dict(captured,command=captured['native_command'])))
        self.assertFalse(run.same_child(captured,dict(captured,started='reused PID')))
        self.assertFalse(run.same_child(captured,dict(captured,command='unrelated executable')))

    def test_owner_lost_before_launch_gate_never_executes_native(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);tree=root/'source';tree.mkdir()
            owner=dict(pid=10,pgid=10,started='owner start',command='owner')
            run.write(root/'tiny-owner.json',owner)
            plan=dict(controller_sha256={},jobs=[dict(name='tiny',command=['./target/release/fit','rbm','tiny'])])
            real_read=run.read
            def read(path):return plan if Path(path)==run.PLAN else real_read(path)
            captured=dict(pid=20,ppid=10,pgid=20,started='child start',command='python bootstrap')
            def process(pid):return captured if pid==20 else None
            with mock.patch.object(run,'read',side_effect=read),mock.patch.object(run.os,'getpid',return_value=20),mock.patch.object(run.identity,'process',side_effect=process),mock.patch.object(run.os,'killpg',side_effect=RuntimeError('stopped')),mock.patch.object(run.os,'execvpe') as execute:
                with self.assertRaisesRegex(RuntimeError,'stopped'):run.bootstrap(tree,'tiny',10)
                execute.assert_not_called()
            ownership=real_read(root/'tiny-ownership.json')
            self.assertEqual(ownership['pid'],20)
            self.assertEqual(ownership['native_command'],'./target/release/fit rbm tiny')

    def test_archive_fully_replays_and_retains_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);tree=root/'source';tree.mkdir();export=root/'exports/tiny';export.mkdir(parents=True)
            payloads={'receipt.json':b'{"complete":true}', 'native/preds_3/tiny.qual.npy':b'own predictions'*1000,
                      'source/src/main.rs':b'original model code'}
            for rel,payload in payloads.items():
                p=export/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(payload)
            digests={rel:hashlib.sha256(payload).hexdigest() for rel,payload in payloads.items()}
            run.write(export/'verification.json',dict(state='verified_complete_original_remote_weekend_producer',job='tiny',
                files_sha256={'receipt.json':digests['receipt.json']},
                all_native_output_sha256={'native/preds_3/tiny.qual.npy':digests['native/preds_3/tiny.qual.npy']},
                original_source_and_binary_sha256={'source/src/main.rs':digests['source/src/main.rs']}))
            dest=root/'archives/tiny'
            with mock.patch.object(run,'verify_inputs'),mock.patch.object(run.shutil,'disk_usage',return_value=type('D',(),{'free':100*2**30})()):
                run.archive(tree,'tiny',dest)
            index=run.read(dest/'index.json')
            self.assertEqual(index['state'],'complete_all_original_native_outputs_and_all_epoch_pairs_lossless_return')
            self.assertEqual(len(index['parts']),1)
            members=index['parts'][0]['members_sha256']
            for rel,payload in payloads.items():
                self.assertEqual((export/rel).read_bytes(),payload)
                self.assertEqual(members[rel],digests[rel])
            (export/'receipt.json').write_bytes(b'changed')
            with mock.patch.object(run,'verify_inputs'),self.assertRaises(ValueError):run.archive(tree,'tiny',root/'bad-archive')
            self.assertFalse((root/'bad-archive').exists())


if __name__=='__main__':unittest.main()
