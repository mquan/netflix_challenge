import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch
import upstream_source_extension as e


class SourceExtensionChecks(unittest.TestCase):
    def test_partial_or_unreceipted_phase_cannot_enter_extension(self):
        ready = dict.fromkeys(e.NEW,True)
        ready['cfnade-12'] = False
        with patch.object(e,'inventory',return_value=ready), patch.object(e.np,'load') as load:
            with self.assertRaisesRegex(ValueError,'cfnade-12'):
                e.added_values(3)
            load.assert_not_called()

    def test_parent_binding_includes_new_implementation_and_frozen_parent(self):
        hashes = e.code_hashes()
        self.assertIn('scripts/upstream_source_extension.py',hashes)
        self.assertIn('scripts/upstream_voting_blend.py',hashes)
        self.assertIn('scripts/upstream_nonlinear.py',hashes)

    def test_inventory_is_fast_but_preparation_requests_full_hash_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); tree = base/'fresh'
            jobs = {name:dict(command='native '+name,outputs=[name+'.probex.npy',name+'.qual.npy']) for name in e.NEW}
            e.u.write(base/'registry.json',dict(jobs=jobs))
            name = e.NEW[0]; outputs = jobs[name]['outputs']
            for filename in outputs:
                path = tree/filename; path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'partial')
            e.u.write(base/'fresh-receipts'/(name+'.json'),dict(source_commit=e.u.PIN,job=name,
                command=jobs[name]['command'],outputs=dict.fromkeys(outputs,'unverified')))
            with patch.object(e.u,'BASE',base),patch.object(e.u,'FRESH',tree):
                self.assertTrue(e.inventory()[name])
                with patch.object(e,'receipt_valid',return_value=False) as checked:
                    with self.assertRaisesRegex(ValueError,'Missing complete paired'):
                        e.added_values(3)
                    self.assertEqual(checked.call_count,len(e.NEW))
                (tree/outputs[-1]).unlink()
                self.assertFalse(e.inventory()[name])


if __name__ == '__main__':
    unittest.main()
