from pathlib import Path
import dataclasses
import contextlib
import io
import json
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2_code_cache import CacheWriter, CodeCache, export_chunks, run
from gbdt_blend import HEADER, PROBE
from train_holdout import digest
from e2_joint_context import peer_code_batch, frozen_peer_vectors
import masked_sequence as ms
from test_selection_objective import fixture


class CodeCacheTests(unittest.TestCase):
    def test_resume_discards_uncommitted_tail_and_matches_direct_codes(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        d = fixture()
        c = ms.Config(rank=4,hidden=8,d_model=8,heads=2,ff=16,layers=1,
            context='session',time_bins=4,dropout=.1,seq_len=24,max_targets=8,
            pad_width=4,pad_rows=1,device='cpu')
        model = ms.SequenceModel(d['items'], c, ms.weeks(d)).eval()
        rows = np.array([0,1,5,11,22], dtype=np.int64)
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            output = Path(temp)/'cache'
            writer = CacheWriter(output, {'checkpoint':'fixture'},len(rows),c.rank,2,rows=rows)
            export_chunks(writer,d,model,2,max_chunks=1)
            self.assertEqual(writer.state['completed'],2)
            with self.assertRaisesRegex(ValueError,'incomplete'):
                CodeCache(output,binding={'checkpoint':'fixture'})
            # Simulate death after writing data but before committing its receipt.
            writer.vectors[2:]=np.nan
            writer.vectors.flush()
            resumed = CacheWriter(output,{'checkpoint':'fixture'},len(rows),c.rank,2,rows=rows,resume=True)
            export_chunks(resumed,d,model,1)
            resumed.seal()
            cache = CodeCache(output,binding={'checkpoint':'fixture'})
            direct,_ = frozen_peer_vectors(model,peer_code_batch(d,rows,c))
            np.testing.assert_allclose(cache(rows),direct.numpy(),rtol=0,atol=1e-6)
            np.testing.assert_array_equal(cache(rows[[4,0,4]]),np.asarray(resumed.vectors)[[4,0,4]])
            self.assertEqual(cache(np.array([],dtype=np.int64)).shape,(0,c.rank))
            for missing in [-1,2,23]:
                with self.assertRaisesRegex(ValueError,'uncached'):
                    cache(np.array([missing]))
            with self.assertRaises(FileExistsError):
                CacheWriter(output,{},len(rows),c.rank,2,rows=rows)
            with self.assertRaisesRegex(ValueError,'changed'):
                CacheWriter(output,{'checkpoint':'other'},len(rows),c.rank,2,rows=rows,resume=True)
            with self.assertRaisesRegex(ValueError,'binding'):
                CodeCache(output,binding={})

    def test_frozen_plan_cli_path_exports_and_resumes_without_training(self):
        torch.set_num_threads(1)
        d=fixture()
        c=ms.Config(rank=4,hidden=8,d_model=8,heads=2,ff=16,layers=1,
            context='session',time_bins=4,dropout=.1,seq_len=24,max_targets=8,
            pad_width=4,pad_rows=1,device='cpu')
        model=ms.SequenceModel(d['items'],c,ms.weeks(d)).eval()
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root=Path(temp); data=root/'data.bin'
            header=np.zeros(1,dtype=HEADER);header['magic']=b'NFPDAT2'
            for key in ('users','items','mean','min_day','max_day'):header[key]=d[key]
            for family in ('train','probe','implicit'):
                header[family+'_count']=len(d['probe'] if family=='probe' else d[family+'_items'])
            with data.open('wb') as stream:
                header.tofile(stream)
                for key,dtype in [('train_offsets','<u8'),('train_items','<u2'),('train_days','<u2'),
                        ('train_ratings','u1'),('probe',PROBE),('implicit_offsets','<u8'),
                        ('implicit_items','<u2'),('implicit_days','<u2')]:
                    d[key].astype(dtype).tofile(stream)
            checkpoint=root/'model.pt';report=root/'report.json';rowfile=root/'rows.npy'
            torch.save(dict(epoch=1,config=dataclasses.asdict(c),items=d['items'],
                            weeks=ms.weeks(d),state=model.state_dict()),checkpoint)
            report.write_text(json.dumps(dict(selected_epoch=1,config=dataclasses.asdict(c),
                                              selection_sha256=digest(data))))
            rows=np.array([0,1,5,11,22],dtype=np.int64);np.save(rowfile,rows)
            sources=Path(__file__).resolve().parents[1]/'scripts'
            frozen=[data,checkpoint,report,rowfile]+[sources/name for name in
                ('e2_code_cache.py','e2_joint_context.py','e2_query_codes.py','masked_sequence.py',
                 'masked_history.py','gbdt_blend.py','train_holdout.py')]
            plan=dict(data=str(data),checkpoint=str(checkpoint),warm_report=str(report),
                source_rows=str(rowfile),frozen={str(p):digest(p) for p in frozen},
                device='cpu',cpu_threads=1,pad_rows=1,chunk_rows=2,batch_pairs=2)
            planfile=root/'plan.json';planfile.write_text(json.dumps(plan))
            state=run(planfile,root/'cache',max_chunks=1)
            self.assertFalse(state['complete']);self.assertEqual(state['completed'],2)
            state=run(planfile,root/'cache',resume=True)
            self.assertTrue(state['complete'])
            cache=CodeCache(root/'cache',binding=state['identity']['binding'])
            expected,_=frozen_peer_vectors(model,peer_code_batch(d,rows,c))
            np.testing.assert_allclose(cache(rows),expected.numpy(),rtol=0,atol=1e-6)
            with data.open('ab') as stream:stream.write(b'changed')
            with self.assertRaisesRegex(ValueError,'frozen input changed'):
                run(planfile,root/'cache',resume=True)

    def test_committed_corruption_and_changed_row_map_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'cache'; rows=np.array([1,3,5])
            writer=CacheWriter(output,{},3,2,2,rows=rows)
            writer.append(np.ones((2,2)))
            writer.append(np.ones((1,2)))
            writer.seal()
            np.save(output/'rows.npy', np.array([1,4,5]))
            with self.assertRaisesRegex(ValueError,'row map'):
                CodeCache(output,binding={})
            np.save(output/'rows.npy',rows)
            writer.vectors[0,0]=2
            writer.vectors.flush()
            with self.assertRaisesRegex(ValueError,'chunk changed'):
                CodeCache(output,binding={})
            with self.assertRaisesRegex(ValueError,'chunk changed'):
                CacheWriter(output,{},3,2,2,rows=rows,resume=True)

    def test_dense_identity_and_invalid_progress_or_payload(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'cache'
            writer=CacheWriter(output,{},3,2,2)
            with self.assertRaisesRegex(ValueError,'incomplete'):
                writer.seal()
            with self.assertRaisesRegex(ValueError,'chunk required'):
                writer.append(np.ones((1,2)))
            with self.assertRaisesRegex(ValueError,'nonfinite'):
                writer.append(np.full((2,2),np.nan))
            writer.append(np.array([[1,2],[3,4]]))
            writer.append(np.array([[5,6]]))
            writer.seal()
            cache=CodeCache(output,binding={})
            np.testing.assert_array_equal(cache(np.array([2,0])),np.array([[5,6],[1,2]]))
            with self.assertRaisesRegex(ValueError,'chunk required'):
                writer.append(np.ones((1,2)))
            state=json.loads((output/'manifest.json').read_text());state['completed']=2
            (output/'manifest.json').write_text(json.dumps(state))
            with self.assertRaisesRegex(ValueError,'progress'):
                CodeCache(output,binding={})


if __name__ == '__main__':
    unittest.main()
