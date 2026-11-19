"""Resource admission and refusal of partial/tampered CF48 completion claims."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import upstream_cfnade48 as m


class Cf48Checks(unittest.TestCase):
    def test_two_phase_admission_reserves_constructor_memory(self):
        gib=2**30
        self.assertTrue(m.admission(17*gib,[],40*gib))
        self.assertFalse(m.admission(18*gib,[3*gib],40*gib))
        self.assertTrue(m.admission(18*gib,[gib],40*gib))
        self.assertFalse(m.admission(10*gib,[],20*gib))

    def test_global_rss_includes_other_native_and_python_groups_once(self):
        rows='1 10 100 /repo/target/release/native\n2 10 30 worker\n3 20 200 '+str(m.u.ROOT/'.venv/bin/python')+' --family nn\n4 20 10 worker\n5 30 500 /repo/target/release/self\n'
        with patch.object(m.subprocess,'check_output',return_value=rows),patch.object(m.os,'getpgrp',return_value=99):
            self.assertEqual(m.other_rss({30}),(100+30+200+10)*1024)

    def test_partial_epoch_and_changed_marker_cannot_claim_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'source/preds_preset48';out.mkdir(parents=True)
            report=root/'report';report.mkdir();plan=report/'plan.json';m.u.write(plan,{})
            terminal=out/'cfnade-48.probex.npy';terminal.write_bytes(b'paired prediction')
            outputs={str(terminal.relative_to(root/'source')):m.u.sha(terminal)}
            outputs.update({f'snapshot{i}':'bound in final validation' for i in range(95)})
            marker=dict(plan_sha256=m.u.sha(plan),epochs=95,outputs=outputs)
            m.u.write(report/'x-complete.json',marker)
            with patch.object(m,'REPORT',report),patch.object(m,'TREE',root/'source'),patch.object(m,'OUT',out):
                with patch.object(m,'epoch',return_value=94):
                    with self.assertRaisesRegex(ValueError,'incomplete'):m.phase_complete('x')
                with patch.object(m,'epoch',return_value=95):
                    self.assertTrue(m.phase_complete('x'))
                    terminal.write_bytes(b'changed prediction')
                    with self.assertRaisesRegex(ValueError,'changed'):m.phase_complete('x')


if __name__=='__main__':unittest.main()
