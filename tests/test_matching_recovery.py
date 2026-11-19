from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import upstream_cf12_matching_recovery as r


class MatchingRecoveryTest(unittest.TestCase):
    def test_refuses_live_predecessor_changed_code_or_existing_fit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);original=root/'original';original.mkdir();script=root/'matching.py'
            script.write_text('# preserved source\n')
            (original/'launch.log').write_text('ValueError: Production bank still incomplete\n')
            child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)','upstream_cf12_matching_production.py'])
            try:
                r.u.write(original/'launch.json',dict(pid=child.pid,script_sha256=r.u.sha(script)))
                with patch.object(r,'ORIGINAL',original),patch.object(r,'SCRIPT',script),patch.object(r.u,'ROOT',root):
                    with self.assertRaisesRegex(ValueError,'still live'):r.preflight()
                    child.terminate();child.wait(timeout=5)
                    self.assertEqual(r.preflight()['pid'],child.pid)
                    script.write_text('# changed code\n')
                    with self.assertRaisesRegex(ValueError,'code changed'):r.preflight()
                    script.write_text('# preserved source\n');(original/'plan.json').write_text('{}')
                    with self.assertRaisesRegex(ValueError,'Existing fit preparation'):r.preflight()
            finally:
                if child.poll() is None:child.terminate();child.wait(timeout=5)


if __name__=='__main__':unittest.main()
