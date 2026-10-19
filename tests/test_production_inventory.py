"""Keep input references from masquerading as output provenance."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from production_inventory import output_digests


class InventoryTests(unittest.TestCase):
    def test_input_hashes_and_checkpoints_are_not_prediction_producers(self):
        self.assertEqual(output_digests(dict(inputs={'prediction.f32': 'input'},
            model_artifact={'sha256': 'model'}, outputs={'checkpoint.pt': 'checkpoint'})), set())

    def test_explicit_prediction_output_schemas_are_recognized(self):
        record = dict(predictions_sha256='a', prediction_sha256='b',
                      predictions_artifact={'sha256': 'c'}, outputs={'column.f32': 'd', 'state.pt': 'e'})
        self.assertEqual(output_digests(record), {'a', 'b', 'c', 'd'})

    def test_missing_or_unrelated_output_metadata_is_not_evidence(self):
        self.assertEqual(output_digests(dict(outputs=['column.f32'], predictions_artifact=None)), set())


if __name__ == '__main__':
    unittest.main()
