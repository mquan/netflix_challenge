import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from local_ensemble_assignment import assignment_ready


class AssignmentTests(unittest.TestCase):
    def test_handoff_requires_every_successful_unchanged_report(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'logs').mkdir()
            stages = [dict(name=name, report=str(root/(name+'.json'))) for name in ['baseline', 'treatment']]
            self.assertFalse(assignment_ready(root, stages))
            for stage in stages:
                report = Path(stage['report'])
                report.write_text('{}')
                receipt = dict(returncode=0, report_sha256=hashlib.sha256(report.read_bytes()).hexdigest())
                (root/'logs'/(stage['name']+'.receipt.json')).write_text(json.dumps(receipt))
            self.assertTrue(assignment_ready(root, stages))
            Path(stages[-1]['report']).write_text('{"changed":true}')
            with self.assertRaises(ValueError):
                assignment_ready(root, stages)


if __name__ == '__main__':
    unittest.main()
