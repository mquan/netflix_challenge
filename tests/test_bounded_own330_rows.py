import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from upstream_own330_native149_data import Rows
from upstream_own330_native149_queue import safe


class BoundedNpyTransport(unittest.TestCase):
    def test_nonsequential_windows_preserve_full_precision_and_header_offset(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'source.npy'
            source=np.arange(17*7,dtype=np.float64).reshape(17,7)/13
            source[3,2]=np.nextafter(1.0,2.0)
            np.save(path,source)
            reader=Rows(path,source.shape,np.float64)
            try:
                for lo,hi in [(5,13),(0,3),(16,17),(2,9)]:
                    self.assertTrue(np.array_equal(reader.read(lo,hi).view('u8'),source[lo:hi].view('u8')))
            finally:reader.close()

    def test_short_or_transposed_payload_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'source.npy';source=np.arange(24,dtype=np.float32).reshape(6,4)
            np.save(path,np.asfortranarray(source))
            with self.assertRaises(ValueError):Rows(path,source.shape,np.float32)
            np.save(path,source)
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaises(ValueError):Rows(path,source.shape,np.float32)

    def test_original_native_and_bounded_diagnostic_fit_but_oversubscription_does_not(self):
        current=dict(swap_used_bytes=6*2**30)
        workers={'original':dict(bytes=21.6*2**30,rss_bytes=18*2**30)}
        self.assertTrue(safe(workers,current,current,23*2**30,1.5*2**30))
        self.assertFalse(safe(workers,current,current,23*2**30,3*2**30))
        self.assertFalse(safe(workers,dict(swap_used_bytes=current['swap_used_bytes']+129*2**20),current,23*2**30))


if __name__=='__main__':unittest.main()
