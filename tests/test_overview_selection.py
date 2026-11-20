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
        expected = {'H8 full'}
        for key in ['development_verification_path', 'latest_qualifying_verification_path']:
            expected.add(next(row[0] for row in journey.MILESTONES if row[9] == record[key]))
        self.assertTrue(expected.issubset(selection['ids']))
        self.assertEqual(len(selection['ids']), 23)
        self.assertEqual(journey.MILESTONES, before)

    def test_busy_november_days_keep_one_best_per_population(self):
        record = json.loads((ROOT / 'experiments/fresh-reproduction-status.json').read_text())
        ids = set(journey.overview_selection(journey.MILESTONES, record)['ids'])
        recent = {row[0] for row in journey.MILESTONES if row[0] in ids and row[4].startswith('fresh_')}
        self.assertEqual(recent, {
            'FRESH86 matched', 'FRESH85 qualifying',
            'FRESH native NN pool6 tree31', 'NNPOOL6 own7 qualifying',
            'FRESH paired391 H128 tree154 half', 'FRESH complete13 qualifying',
        })
        self.assertNotIn('FRESH paired391 H512 control', ids)

    def test_pacific_midnight_series_authors_and_ties(self):
        def row(name, score, timestamp, metric='fresh_cv', credit='codex', source='other'):
            return (name, score, timestamp, 'Fresh training', metric, credit, '', '', '', source, '')
        rows = [
            row('previous day', '.88', '2026-11-20T07:59:00+00:00'),
            row('morning', '.87', '2026-11-20T08:01:00+00:00'),
            row('evidence best', '.86', '2026-11-20T19:00:00+00:00', source='known'),
            row('later equal', '.86', '2026-11-20T20:00:00+00:00'),
            row('late regression', '.89', '2026-11-20T21:00:00+00:00'),
            row('qualifying', '.90', '2026-11-20T19:00:00+00:00', metric='fresh_qualifying', source='qual'),
            row('other author', '.85', '2026-11-20T19:00:00+00:00', credit='claude'),
        ]
        record = dict(development_verification_path='known', latest_qualifying_verification_path='qual')
        result = journey.overview_selection(rows, record)
        self.assertEqual(set(result['ids']), {'previous day', 'evidence best', 'qualifying', 'other author'})
        self.assertEqual(result, journey.overview_selection(list(reversed(rows)), record))
        self.assertEqual(set(result['days']), {'2026-11-19', '2026-11-20'})

    def test_public_bank_results_are_excluded_from_published_data(self):
        published = journey.own_training_milestones(journey.MILESTONES)
        self.assertFalse(any(row[3] == 'Public bank' or row[4] == 'qualifying' for row in published))
        self.assertFalse({'PUB466 NN2', 'QUAL619 ridge'} & {row[0] for row in published})
        self.assertEqual(len(published), len(journey.MILESTONES) - 6)
        self.assertEqual(sum(row[0] == 'QUAL619 ridge' for row in journey.MILESTONES), 1)

    def test_future_best_updates_id_and_pacific_date(self):
        def row(name, timestamp, path):
            return (name, '.85', timestamp, '', '', '', '', '', '', path, '')
        rows = [row('new CV', '2026-11-21T03:00:00+00:00', 'known'),
                row('new qualifying', '2026-11-22T01:00:00+00:00', 'qual')]
        selected = journey.overview_selection(rows, dict(development_verification_path='known', latest_qualifying_verification_path='qual'))
        self.assertIn('new CV', selected['ids'])
        self.assertIn('new qualifying', selected['ids'])
        self.assertEqual(set(selected['days']), {'2026-11-20', '2026-11-21'})

    def test_missing_or_ambiguous_evidence_fails_closed(self):
        record = dict(development_verification_path='missing', latest_qualifying_verification_path='qual')
        with self.assertRaises(ValueError):
            journey.overview_selection([], record)
        row = ('duplicate', '.85', '2026-11-20T03:00:00+00:00', '', '', '', '', '', '', 'missing', '')
        with self.assertRaises(ValueError):
            journey.overview_selection([row, row], record)


if __name__ == '__main__':
    unittest.main()
