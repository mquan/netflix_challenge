"""Complete synthetic terminal predictor and durable output contracts, CPU only."""
import copy
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import test_upstream_factored_rbm_training as training_tests
import upstream_factored_rbm_terminal as terminal
from upstream_factored_rbm_terminal import freeze,replay,validate,durable_payload
DIGEST=training_tests.DIGEST


class TerminalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch=torch;torch.set_num_threads(1)

    def state(self):
        helper=training_tests.TrainingTests();helper.torch=self.torch
        reader,state=helper.setup_state()
        for epoch in (1,2):state.fit_epoch(epoch,lambda:None)
        return reader,state

    def test_complete_predictive_state_replays_all_original_queries_exactly(self):
        r,s=self.state();record=freeze(s,'known',DIGEST,asdict(s.config))
        self.assertFalse(record['training_resume_supported']);self.assertFalse(record['optimizer_state_retained'])
        self.assertEqual(set(record['model']),{'p','q','d','bv','bh','bu','but'})
        np.testing.assert_array_equal(replay(record,r,'known',DIGEST,asdict(s.config),lambda:None),s.predict_queries(lambda:None))
        for k,v in dict(s.shared,**s.bias).items():self.assertTrue(self.torch.equal(record['model'][k],v))

    def test_incomplete_wrong_phase_source_or_missing_tensors_are_rejected(self):
        r,s=self.state();record=freeze(s,'known',DIGEST,asdict(s.config))
        for phase,digest in [('qualifying',DIGEST),('known','2'*64)]:
            with self.assertRaises(ValueError):validate(record,r,phase,digest,asdict(s.config))
        for change in (lambda x:x['model'].pop('q'),lambda x:x.update(completed_epoch=1),lambda x:x.update(training_resume_supported=True)):
            bad=copy.deepcopy(record);change(bad)
            with self.assertRaises(ValueError):validate(bad,r,'known',DIGEST,asdict(s.config))
        with self.assertRaises(ValueError):validate(record,r,'known',DIGEST,asdict(training_tests.Config()))
        s.completed_epoch=1
        with self.assertRaises(ValueError):freeze(s,'known',DIGEST,asdict(s.config))

    def test_durable_complete_model_and_vector_roundtrip_without_overwrite(self):
        t=self.torch;r,s=self.state();record=freeze(s,'known',DIGEST,asdict(s.config));values=s.predict_queries(lambda:None)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);model=root/'model.pt';vector=root/'query.npy'
            receipt=durable_payload(model,lambda stream:t.save(record,stream),root,lambda **kwargs:None,0,0)
            self.assertEqual(receipt['bytes'],model.stat().st_size)
            loaded=t.load(model,map_location='cpu',weights_only=True)
            np.testing.assert_array_equal(replay(loaded,r,'known',DIGEST,asdict(s.config),lambda:None),values)
            durable_payload(vector,lambda stream:np.save(stream,values,allow_pickle=False),root,lambda **kwargs:None,0,0)
            np.testing.assert_array_equal(np.load(vector,allow_pickle=False),values)
            self.assertFalse(model.with_name(model.name+'.staging').exists())
            with self.assertRaises(ValueError):durable_payload(model,lambda stream:stream.write(b'bad'),root,lambda **kwargs:None,0,0)

    def test_failed_write_keeps_staging_and_cannot_be_retried(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);path=root/'partial.bin'
            def fail(stream):stream.write(b'partial');raise RuntimeError('synthetic fatal writer')
            with self.assertRaises(RuntimeError):durable_payload(path,fail,root,lambda **kwargs:None,0,0)
            self.assertFalse(path.exists());self.assertTrue(path.with_name(path.name+'.staging').exists())
            with self.assertRaises(ValueError):durable_payload(path,lambda stream:stream.write(b'retry'),root,lambda **kwargs:None,0,0)

    def test_each_payload_chunk_requests_original_reserve_and_all_buffer_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);path=root/'bytes.bin';value=np.arange(700000,dtype=np.uint16);reservations=[]
            def guard(**kwargs):
                if 'extra_disk_bytes' in kwargs:reservations.append(kwargs['extra_disk_bytes'])
            durable_payload(path,lambda stream:stream.write(memoryview(value)),root,guard,0,64)
            self.assertEqual(path.read_bytes(),value.tobytes())
            self.assertEqual(reservations,[1048576+64,value.nbytes-1048576+64])

    def test_new_directory_chain_is_flushed_before_any_payload_write(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);path=root/'new'/'nested'/'value.bin';expected=[path.parent.resolve(),path.parent.parent.resolve(),root.resolve(),root.resolve().parent]
            with patch.object(terminal,'sync_directory',wraps=terminal.sync_directory) as synced:
                def writer(stream):
                    observed=[Path(call.args[0]).resolve() for call in synced.call_args_list]
                    self.assertEqual(observed,expected)
                    stream.write(b'durable')
                durable_payload(path,writer,root,lambda **kwargs:None,0,0)
            self.assertEqual(path.read_bytes(),b'durable')
