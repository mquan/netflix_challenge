import io
from pathlib import Path
import tarfile
import tempfile
import unittest
import verify_remote_temporal_archive as v


class IntakeArchiveTest(unittest.TestCase):
    def test_only_complete_regular_payload_can_be_extracted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);archive=root/'return.tar.gz'
            names=[v.TOP+'/manifest.json',v.TOP+'/native-receipt.json',
                   v.TOP+'/outputs/preds_new/tsvdx5-150o.out']
            def build(extra=None,missing=False):
                with tarfile.open(archive,'w:gz') as stream:
                    for name in names[:-1] if missing else names:
                        member=tarfile.TarInfo(name);member.size=2
                        stream.addfile(member,io.BytesIO(b'ok'))
                    if extra:stream.addfile(extra)
            outputs={'preds_new/tsvdx5-150o.out':'unused'}
            for index,member in enumerate([tarfile.TarInfo('../escape'),tarfile.TarInfo(names[0])]):
                build(member)
                with self.assertRaises(ValueError):v.extract_exact(archive,root/f'bad{index}',outputs)
            link=tarfile.TarInfo(names[-1]);link.type=tarfile.SYMTYPE;link.linkname='/tmp/outside'
            build(link,missing=True)
            with self.assertRaises(ValueError):v.extract_exact(archive,root/'link',outputs)
            build(missing=True)
            with self.assertRaisesRegex(ValueError,'Incomplete'):v.extract_exact(archive,root/'missing',outputs)
            build();stage=v.extract_exact(archive,root/'good',outputs)
            self.assertEqual((stage/'outputs/preds_new/tsvdx5-150o.out').read_bytes(),b'ok')
            with self.assertRaisesRegex(ValueError,'Existing'):v.extract_exact(archive,root/'good',outputs)


if __name__=='__main__':unittest.main()
