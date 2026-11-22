"""Full-width/block-boundary tests with synthetic data and original wire sender."""
import io
import sys
import unittest
import numpy as np
import upstream_full_context_catboost_query_stream as module
import test_upstream_full_context_catboost_wire as fixtures


class QueryTests(unittest.TestCase):
    def test_every_original_row_and_ten_baselines_all_widths_and_8192_boundary(self):
        for keep,latent,n in [(False,False,19),(True,False,19),(False,True,19),(True,True,8193)]:
            with self.subTest(keep=keep,latent=latent,n=n):
                payload,contract,expected,raw=fixtures.WireTests().payload(keep,latent,n)
                seen=[]
                def consume(lo,hi,x,b):
                    self.assertEqual(lo,sum(seen));self.assertLessEqual(hi-lo,8192)
                    self.assertEqual(x[:,:-1].tobytes(),raw[lo:hi].tobytes())
                    self.assertEqual(b.tobytes(),expected[:,:,lo:hi].tobytes())
                    self.assertFalse(x.flags.writeable);self.assertFalse(b.flags.writeable)
                    seen.append(hi-lo)
                proof=module.receive(io.BytesIO(payload),contract,consume,guard=lambda:None)
                self.assertEqual(sum(seen),n);self.assertEqual(proof['rows'],n)
                self.assertEqual(proof['blocks'],len(seen));self.assertTrue(proof['exact_footer_and_EOF'])
                self.assertFalse(proof['prediction_publication_authorized'])
                self.assertFalse(proof['actual_producer_successful_exit_verified'])
                self.assertNotIn('torch',sys.modules)

    def test_partial_trailing_and_mutated_stream_cannot_authorize_predictions(self):
        payload,contract,_,_=fixtures.WireTests().payload()
        reader=io.BytesIO(payload);reader.read(len(module.wire.MAGIC));module.wire.read_metadata(reader,lambda:None)
        changed=bytearray(payload);changed[reader.tell()+4]^=1
        for data in [payload[:-1],payload[:100],payload+b'x',changed]:
            with self.subTest(size=len(data)):
                with self.assertRaises(ValueError):module.receive(io.BytesIO(data),contract,lambda *a:None,guard=lambda:None)

    def test_consumer_mutation_and_interrupt_do_not_produce_complete_proof(self):
        payload,contract,_,_=fixtures.WireTests().payload()
        def changed(lo,hi,x,b):
            x.flags.writeable=True;x[0,0]=np.nextafter(x[0,0],np.float32(np.inf))
        with self.assertRaisesRegex(ValueError,'mutate'):module.receive(io.BytesIO(payload),contract,changed,guard=lambda:None)
        def interrupt(*args):raise RuntimeError('preserve same numerical call stack')
        with self.assertRaisesRegex(RuntimeError,'same numerical call stack'):
            module.receive(io.BytesIO(payload),contract,interrupt,guard=lambda:None)


if __name__=='__main__':unittest.main()
