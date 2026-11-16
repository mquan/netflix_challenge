"""End-to-end outer-label poisoning through the portable execution adapters."""
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from test_local_ensemble_isolation import OuterIsolation

ROOT = Path(__file__).resolve().parents[1]


class DeviceIsolation(unittest.TestCase):
    def check_device(self, device):
        original = subprocess.run
        mapping = {'round4_boost.py': 'tree', 'round8_nn_blend.py': 'nn', 'round9_enn_blend.py': 'enn'}
        def adapted(command, *args, **kwargs):
            if isinstance(command, list) and len(command) > 1 and Path(command[1]).name in mapping:
                command = [command[0], str(ROOT/'scripts/local_ensemble_device.py'),
                           mapping[Path(command[1]).name], '--device', device,
                           '--tree-threads', '2', *command[2:]]
            return original(command, *args, **kwargs)
        with patch('subprocess.run', side_effect=adapted):
            OuterIsolation().test_poisoned_outer_labels_do_not_change_training_or_predictions()

    def test_cpu_adapter_outer_label_isolation(self):
        self.check_device('cpu')

    def test_available_gpu_adapter_outer_label_isolation(self):
        import torch
        device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else None)
        if device is None:
            self.skipTest('No GPU backend available')
        self.check_device(device)


if __name__ == '__main__':
    unittest.main()
