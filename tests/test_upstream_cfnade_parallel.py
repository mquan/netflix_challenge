import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import upstream_cfnade_parallel as p


class ParallelChecks(unittest.TestCase):
    def test_reserves_memory_for_each_stage_and_prioritizes_blender(self):
        g=2**30
        self.assertFalse(p.admit('training',5*g,[],40*g))
        self.assertTrue(p.admit('complete',10*g,[5*g],40*g))
        self.assertFalse(p.admit('complete',15*g,[6*g],40*g))
        self.assertFalse(p.admit('complete',5*g,[],20*g))

    def test_terminal_file_is_not_complete_until_successful_exit_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base=root/'base';tree=root/'tree';report=root/'report'
            output=tree/'preds_new/cfnade-96.probex.npy';output.parent.mkdir(parents=True)
            np.save(output,np.full(1128164,3.0,dtype=np.float32))
            p.u.write(base/'parallel-checkpoints/trainx/latest.json',dict(epoch=95))
            p.u.write(report/'plan.json',dict(source='frozen'))
            with patch.object(p,'BASE',base),patch.object(p,'TREE',tree),patch.object(p,'REPORT',report):
                self.assertFalse(p.phase_complete('x'))
                p.finalize_phase('x');self.assertTrue(p.phase_complete('x'))
                with output.open('r+b') as stream:
                    stream.seek(-1,2);stream.write(b'X')
                with self.assertRaisesRegex(ValueError,'receipt changed'):p.phase_complete('x')


if __name__=='__main__':unittest.main()
