"""Full-width wire, failure barriers and actual isolated-process fixtures."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock
import numpy as np
import upstream_full_context_catboost_wire as module
import test_upstream_full_context_catboost_stream as fixtures


class WireTests(unittest.TestCase):
    def fixture(self, keep=False, latent=False, n=19):
        reader, w, x, b, hashes, expected, raw = fixtures.StreamTests().fixture(keep, latent, n)
        declaration = module.contract(n, retain_scalar=keep, latent=latent,
            weights_sha256=hashlib.sha256(w.tobytes()).hexdigest(), admission_sha256='a'*64, operator_sha256=hashes)
        return reader, w, declaration, expected, raw

    def payload(self, keep=False, latent=False, n=19):
        reader, weights, declaration, expected, raw = self.fixture(keep, latent, n)
        output = io.BytesIO()
        module.send(reader, weights, declaration, output, verify_sources=lambda: 'a'*64, guard=lambda: None)
        return output.getvalue(), declaration, expected, raw

    def test_every_raw_and_ten_baseline_bits_all_widths_and_block_boundary(self):
        for keep, latent, n in [(False, False, 19), (True, False, 19), (False, True, 19), (True, True, 8193)]:
            with self.subTest(keep=keep, latent=latent, n=n):
                payload, declaration, expected, raw = self.payload(keep, latent, n)
                x, b, proof = module.receive(io.BytesIO(payload), declaration, guard=lambda: None)
                self.assertEqual(x[:, :-1].tobytes(), raw.tobytes())
                self.assertTrue((x[:, -1] == 0).all())
                self.assertEqual(b.tobytes(), expected.tobytes())
                self.assertTrue(proof['exact_footer_and_EOF'])
                self.assertFalse(proof['actual_producer_successful_exit_verified'])
                self.assertFalse(proof['production_data_admitted'])

    def test_truncation_trailing_bytes_and_feature_bit_mutation_rejected(self):
        payload, declaration, _, _ = self.payload()
        for data in [payload[:-1], payload[:100], payload+b'x']:
            with self.assertRaises(ValueError): module.receive(io.BytesIO(data), declaration, guard=lambda: None)
        source = io.BytesIO(payload); source.read(len(module.MAGIC)); module.read_metadata(source, lambda: None)
        changed = bytearray(payload); changed[source.tell()+4] ^= 1
        with self.assertRaises(ValueError): module.receive(io.BytesIO(changed), declaration, guard=lambda: None)

    def test_header_types_and_changed_source_or_weight_never_complete(self):
        payload, declaration, _, _ = self.payload()
        bad = dict(declaration, qualifying_labels_read=0)
        with self.assertRaises(ValueError): module.receive(io.BytesIO(payload), bad, guard=lambda: None)
        reader, w, declaration, _, _ = self.fixture()
        source_calls = [0]
        def source_check():
            source_calls[0] += 1
            return 'a'*64 if source_calls[0] == 1 else 'b'*64
        output = io.BytesIO()
        with self.assertRaises(ValueError):
            module.send(reader, w, declaration, output, verify_sources=source_check, guard=lambda: None)
        with self.assertRaises(ValueError): module.receive(io.BytesIO(output.getvalue()), declaration, guard=lambda: None)
        reader, w, declaration, _, _ = self.fixture(); w[0, 0, 0] += .001
        with self.assertRaises(ValueError):
            module.send(reader, w, declaration, io.BytesIO(), verify_sources=lambda: 'a'*64, guard=lambda: None)

    def test_partial_IO_and_header_rejection_before_matrix_allocation(self):
        reader, weights, declaration, expected, raw = self.fixture()
        class Partial(io.BytesIO):
            def write(self, value): return super().write(value[:7])
            def readinto(self, value): return super().readinto(value[:11])
        output = Partial()
        module.send(reader, weights, declaration, output, verify_sources=lambda: 'a'*64, guard=lambda: None)
        output.seek(0)
        x, b, _ = module.receive(output, declaration, guard=lambda: None)
        self.assertEqual(x[:, :-1].tobytes(), raw.tobytes())
        self.assertEqual(b.tobytes(), expected.tobytes())
        wrong = dict(declaration, admission_sha256='b'*64)
        with mock.patch.object(module.np, 'empty', side_effect=AssertionError('matrix allocated')):
            with self.assertRaises(ValueError):
                module.receive(io.BytesIO(output.getvalue()), wrong, guard=lambda: None)
            huge = module.MAGIC + module.struct.pack('!I', module.METADATA_LIMIT+1)
            with self.assertRaises(ValueError):
                module.receive(io.BytesIO(huge), declaration, guard=lambda: None)

    def test_actual_producer_and_Torch_free_receiver_over_pipe(self):
        _, _, declaration, _, _ = self.fixture()
        directory = str(Path(__file__).resolve().parent)
        producer_code = '''import sys
sys.path.insert(0, sys.argv[1])
import torch
import upstream_full_context_catboost_wire as wire
import test_upstream_full_context_catboost_wire as fixture
reader, weights, declaration, _, _ = fixture.WireTests().fixture()
wire.send(reader, weights, declaration, sys.stdout.buffer, verify_sources=lambda: 'a'*64, guard=lambda: None)
'''
        receiver_code = '''import sys, json
sys.path.insert(0, sys.argv[1])
import upstream_full_context_catboost_wire as wire
assert 'torch' not in sys.modules
x, b, proof = wire.receive(sys.stdin.buffer, json.loads(sys.argv[2]), guard=lambda: None)
assert 'torch' not in sys.modules
print(json.dumps(proof))
'''
        producer = subprocess.Popen([sys.executable, '-c', producer_code, directory], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        receiver = subprocess.Popen([sys.executable, '-c', receiver_code, directory, module.encoded(declaration).decode()],
            stdin=producer.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        producer.stdout.close()
        try:
            received, errors = receiver.communicate(timeout=30)
            self.assertEqual(producer.wait(timeout=30), 0, producer.stderr.read().decode())
            self.assertEqual(receiver.returncode, 0, errors.decode())
            proof = json.loads(received)
            self.assertTrue(proof['exact_footer_and_EOF'])
            # The parent just checked the true exit; the binary receiver alone
            # correctly refused to claim it or admit a production study.
            self.assertFalse(proof['actual_producer_successful_exit_verified'])
        finally:
            for process in (producer, receiver):
                if process.poll() is None:
                    process.terminate(); process.wait(timeout=10)
            producer.stderr.close()


if __name__ == '__main__': unittest.main()
