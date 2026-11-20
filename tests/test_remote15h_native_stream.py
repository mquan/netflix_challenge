import gzip
import hashlib
import io
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import upstream_remote15h_native_stream as stream


class NativeArchiveTransport(unittest.TestCase):
    def fixture(self, unsafe=False):
        files = {'source/preds_3/test.probe_3.npy': b'own native query'*100,
                 'source/preds_3/test.train_3.npy': b'large own training prediction'*1000}
        if unsafe:
            files['../escape'] = b'never write'
        raw = io.BytesIO()
        with tarfile.open(fileobj=raw, mode='w') as archive:
            for name, payload in files.items():
                info = tarfile.TarInfo(name); info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
        compressed = gzip.compress(raw.getvalue()); boundary = len(compressed)//2
        chunks = dict(a=compressed[:boundary], b=compressed[boundary:])
        parts = [(name, dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())) for name,data in chunks.items()]
        members = {name:dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest()) for name,data in files.items()}
        return chunks, parts, members

    def test_checks_both_parts_and_unmaterialized_members(self):
        chunks, parts, members = self.fixture()
        reader = stream.Parts(parts, lambda name:io.BytesIO(chunks[name]))
        selected = {'source/preds_3/test.probe_3.npy'}
        with tempfile.TemporaryDirectory() as directory, patch.object(stream.shutil, 'disk_usage') as disk:
            disk.return_value.free = 32*2**30
            seen, _ = stream.extract(reader, members, selected, Path(directory))
            self.assertEqual(set(seen), set(members))
            self.assertEqual(len(reader.verified), 2)
            self.assertTrue((Path(directory)/next(iter(selected))).exists())
            self.assertFalse((Path(directory)/'source/preds_3/test.train_3.npy').exists())

    def test_corrupt_complete_part_is_rejected(self):
        chunks, parts, members = self.fixture()
        parts[-1][1]['sha256'] = 'wrong'
        reader = stream.Parts(parts, lambda name:io.BytesIO(chunks[name]))
        with tempfile.TemporaryDirectory() as directory, patch.object(stream.shutil, 'disk_usage') as disk:
            disk.return_value.free = 32*2**30
            with self.assertRaises(ValueError):
                stream.extract(reader, members, set(), Path(directory))

    def test_unsafe_declared_path_is_still_rejected(self):
        chunks, parts, members = self.fixture(unsafe=True)
        reader = stream.Parts(parts, lambda name:io.BytesIO(chunks[name]))
        with tempfile.TemporaryDirectory() as directory, patch.object(stream.shutil, 'disk_usage') as disk:
            disk.return_value.free = 32*2**30
            with self.assertRaises(ValueError):
                stream.extract(reader, members, set(members), Path(directory))


if __name__ == '__main__':
    unittest.main()
