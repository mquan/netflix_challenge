import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import upstream_own176_recent10_queue as queue

class WaitingObserved(Exception):pass

class DependencyLockRegression(unittest.TestCase):
    def test_unfinished_gpu_dependency_keeps_lock_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(queue,'REPORT',Path(tmp)/'report'),patch.object(queue,'ready',return_value=False),\
                patch.object(queue.signal,'signal'),patch.object(queue.fcntl,'flock') as gpu_lock,\
                patch.object(queue.resources,'footprints',return_value={}),\
                patch.object(queue.memory,'memory',return_value=dict(swap_used_bytes=0,compressor_bytes=0)),\
                patch.object(queue.time,'sleep',side_effect=WaitingObserved),\
                patch.object(queue.subprocess,'Popen') as fit:
                with self.assertRaises(WaitingObserved):queue.main()
                gpu_lock.assert_not_called()
                fit.assert_not_called()

if __name__=='__main__':unittest.main()
