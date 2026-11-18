from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import upstream_dnn_backfill as d


class DnnBackfillChecks(unittest.TestCase):
    def test_resource_count_includes_both_canonical_phases_and_final_fitting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            d.u.write(root/'experiments/upstream-cfnade-parallel/status.json',dict(active={'x':11,'qual':12}))
            d.u.write(root/'experiments/upstream-production/status.json',dict(state='full_probe_refitting',pid=13))
            with patch.object(d.u,'ROOT',root),patch.object(d,'group_rss',side_effect=lambda pid:pid*1024):
                self.assertEqual(d.extra_rss(),(11+12+13)*1024)


if __name__=='__main__':unittest.main()
