import io
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import upstream_tree_pipe as wire


class ShortIO(io.BytesIO):
    def write(self,raw):return super().write(memoryview(raw)[:7])
    def readinto(self,raw):return super().readinto(memoryview(raw)[:5])


class PipeProtocol(unittest.TestCase):
    def test_partial_reads_and_writes_preserve_metadata_and_float_bits(self):
        stream=ShortIO();metadata=dict(rows=23,width=9)
        data=np.arange(207,dtype='<f4').reshape(23,9);data[0,:3]=[np.float32(-0.),np.nextafter(np.float32(1),np.float32(2)),np.float32(-1.e-20)]
        baseline=np.linspace(1.05,4.95,230,dtype='<f8').reshape(23,10)
        wire.write_json(stream,metadata);wire.write_array(stream,data);wire.write_array(stream,baseline);wire.write_json(stream,dict(done=True))
        stream.seek(0);self.assertEqual(wire.read_json(stream),metadata)
        np.testing.assert_array_equal(wire.read_array(stream,(23,9),'<f4').view('u4'),data.view('u4'))
        np.testing.assert_array_equal(wire.read_array(stream,(23,10),'<f8').view('u8'),baseline.view('u8'))
        self.assertEqual(wire.read_json(stream),dict(done=True));self.assertEqual(stream.read(),b'')

    def test_noncontiguous_arrays_have_declared_c_order(self):
        data=np.arange(99,dtype=np.float32).reshape(9,11)[:,::2];stream=io.BytesIO()
        wire.write_array(stream,data);stream.seek(0)
        np.testing.assert_array_equal(wire.read_array(stream,data.shape,'<f4'),data)

    def test_truncated_array_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'Truncated'):
            wire.read_array(io.BytesIO(b'\0'*15),(4,),'<f4')

    def test_unbounded_or_truncated_metadata_is_rejected(self):
        for raw in [(2**20+1).to_bytes(8,'little'),(9).to_bytes(8,'little')+b'{}',b'\0'*8]:
            with self.subTest(raw=raw),self.assertRaises(ValueError):wire.read_json(io.BytesIO(raw))


if __name__=='__main__':unittest.main()
