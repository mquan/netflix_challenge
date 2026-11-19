from pathlib import Path
import tempfile
import unittest
import numpy as np
import upstream_remote_cf96 as r


class TransferCheckpointTest(unittest.TestCase):
    def test_owner_and_all84_tensor_checks_reject_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);directory=root/'epoch-00003';directory.mkdir()
            files={}
            for index in range(84):
                p=directory/f'field-{index}.npy';np.save(p,np.array([.5,1.5],dtype=np.float32));files[p.name]=r.io.sha(p,'md5')
            identity=dict(source='a'*64,data='b'*64,query='probex',query_rows=3,
                config='CfNadeConfig { n_hidden: 320, rank: 96, n_epochs: 95, }')
            state=dict(epoch=3,identity=identity,files=files,timestep=10,swa_count=0,
                rng_seed=[0]*32,rng_stream=0,rng_word_pos='100',swapped_in=False)
            r.io.write(directory/'state.json',state);r.io.write(root/'owner.json',identity)
            r.io.write(root/'latest.json',dict(epoch=3,manifest_md5=r.io.sha(directory/'state.json','md5')))
            pointer,loaded,_=r.phase_checkpoint(root,'a'*64,'b'*64,'probex',3)
            self.assertEqual(pointer['epoch'],3);self.assertEqual(loaded,state)
            r.io.write(root/'owner.json',dict(identity,data='c'*64))
            with self.assertRaisesRegex(ValueError,'owner/config/identity'):r.phase_checkpoint(root,'a'*64,'b'*64,'probex',3)
            r.io.write(root/'owner.json',identity)
            np.save(directory/'field-83.npy',np.array([.5,2.5],dtype=np.float32))
            with self.assertRaisesRegex(ValueError,'tensor differs'):r.phase_checkpoint(root,'a'*64,'b'*64,'probex',3)

    def test_rejects_escaping_or_symlinked_members(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'link').symlink_to(root/'outside')
            for name in ['../outside','/tmp/outside','link']:
                with self.assertRaises(ValueError):r.safe_member(root,name)


if __name__=='__main__':unittest.main()
