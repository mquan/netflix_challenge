"""Keep the simplified overview tied to verified best IDs without dropping ledger rows."""
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import build_research_journey as journey


class OverviewSelectionTests(unittest.TestCase):
    def test_current_representatives_follow_evidence_paths(self):
        record = json.loads((ROOT / 'experiments/fresh-reproduction-status.json').read_text())
        before = copy.deepcopy(journey.MILESTONES)
        selection = journey.overview_selection(journey.MILESTONES, record)
        expected = {'H8 full', 'QUAL619 ridge'}
        for key in ['development_verification_path', 'latest_qualifying_verification_path']:
            expected.add(next(row[0] for row in journey.MILESTONES if row[9] == record[key]))
        self.assertEqual(set(selection['ids']), expected)
        self.assertEqual(journey.MILESTONES, before)

    def test_future_best_updates_id_and_pacific_date(self):
        def row(name, timestamp, path):
            return (name, '.85', timestamp, '', '', '', '', '', '', path, '')
        rows = [row('new CV', '2026-11-21T03:00:00+00:00', 'known'),
                row('new qualifying', '2026-11-22T01:00:00+00:00', 'qual')]
        selected = journey.overview_selection(rows, dict(development_verification_path='known', latest_qualifying_verification_path='qual'))
        self.assertIn('new CV', selected['ids'])
        self.assertIn('new qualifying', selected['ids'])
        self.assertEqual(set(selected['days']), {'2026-10-31', '2026-11-15', '2026-11-20', '2026-11-21'})

    def test_missing_or_ambiguous_evidence_fails_closed(self):
        record = dict(development_verification_path='missing', latest_qualifying_verification_path='qual')
        with self.assertRaises(ValueError):
            journey.overview_selection([], record)
        row = ('duplicate', '.85', '2026-11-20T03:00:00+00:00', '', '', '', '', '', '', 'missing', '')
        with self.assertRaises(ValueError):
            journey.overview_selection([row, row], record)


if __name__ == '__main__':
    unittest.main()
