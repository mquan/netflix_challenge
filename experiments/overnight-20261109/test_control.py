import datetime as dt
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('overnight_control',Path(__file__).with_name('control.py'))
control=importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)


class ScheduledPauseTests(unittest.TestCase):
    def test_collects_only_owned_process_trees(self):
        table={1:dict(ppid=0),2:dict(ppid=1),3:dict(ppid=2),4:dict(ppid=99),5:dict(ppid=4)}
        self.assertEqual(control.descendants(table,[1]),{1,2,3})
        self.assertEqual(control.descendants(table,[1,4]),{1,2,3,4,5})
        self.assertEqual(control.descendants(table,[77]),set())

    def test_no_early_pause_or_signals(self):
        future=(dt.datetime.now(dt.timezone.utc)+dt.timedelta(days=1)).isoformat()
        config=dict(frozen={},cutoff_utc=future)
        with patch.object(control.helper,'read',return_value=config), patch.object(control.os,'kill') as kill:
            with self.assertRaisesRegex(ValueError,'before requested cutoff'):
                control.pause()
            kill.assert_not_called()

    def test_partial_or_holey_receipts_are_not_completed(self):
        self.assertEqual(control.verified_prefix([True,True,False]),2)
        self.assertEqual(control.verified_prefix([False,False]),0)
        self.assertEqual(control.verified_prefix([True,True]),2)
        with self.assertRaises(ValueError):
            control.verified_prefix([True,False,True])

    def test_identity_reuse_excluded(self):
        row=dict(pid=12,name='GPU',command='original',started='old')
        with patch.object(control.subprocess,'check_output',return_value='new'):
            self.assertFalse(control.matched_controller(row,dict(state='S',command='original')))
        with patch.object(control.subprocess,'check_output',return_value='old'):
            self.assertFalse(control.matched_controller(row,dict(state='S',command='unrelated')))
            self.assertTrue(control.matched_controller(row,dict(state='S',command='original')))
        self.assertFalse(control.matched_controller(row,None))


if __name__=='__main__':
    unittest.main()
