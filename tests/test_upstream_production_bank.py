from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import upstream_production_bank as p


class ProductionBankChecks(unittest.TestCase):
    def test_block_assembly_matches_frozen_centering_across_chunk_sizes(self):
        rng=np.random.default_rng(2026);raw=rng.normal(3.6,2,(29,85)).astype(np.float32)
        clip=np.ones(85,dtype=bool);clip[[55,56,57,59,60]]=False
        context=rng.normal(size=(29,5));votes=rng.normal(size=(29,130)).astype(np.float32)
        clipped=raw.copy();clipped[:,clip]=np.clip(clipped[:,clip],0,6)
        original=clipped[:,:62].copy();mean=original[:,clip[:62]].mean(1,keepdims=True)
        nonlin=original.copy();nonlin[:,clip[:62]]-=mean
        expected=np.column_stack([mean-3.6,nonlin,context,clipped[:,62:]-3.6,votes]).astype(np.float32)
        z=p.b.design(original,clip[:62],context,tuple(range(14)))
        expected_z=np.column_stack([z[:,:-1],clipped[:,62:],z[:,-1:]])
        for size in [1,7,29]:
            pieces=[p.blocks(raw[lo:lo+size],clip,context[lo:lo+size],votes[lo:lo+size]) for lo in range(0,len(raw),size)]
            self.assertTrue(np.array_equal(np.vstack([a for a,b in pieces]),expected_z))
            self.assertTrue(np.array_equal(np.vstack([b for a,b in pieces]),expected))

    def test_receipt_pair_must_be_complete_and_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);tree=root/'fresh';receipts=root/'receipts';(tree/'preds_new').mkdir(parents=True)
            paths={key:f'preds_new/model.{key}.npy' for key in ['probex','qual']}
            for path in paths.values():np.save(tree/path,np.arange(4,dtype=np.float32))
            record=dict(source_commit=p.u.PIN,job='model',command='train',outputs={paths['probex']:p.u.sha(tree/paths['probex'])})
            p.u.write(receipts/'model.json',record)
            row=dict(name='model',producer='model',clip=True);jobs={'model':{'command':'train'}}
            with patch.object(p,'LOCATIONS',[(tree,receipts,'native')]),patch.object(p.u,'ROOT',root):
                self.assertIsNone(p.choose_pair(row,jobs,True))
                record['outputs'][paths['qual']]=p.u.sha(tree/paths['qual']);p.u.write(receipts/'model.json',record)
                self.assertIsNotNone(p.choose_pair(row,jobs,True))
                np.save(tree/paths['qual'],np.ones(4,dtype=np.float32))
                with self.assertRaisesRegex(ValueError,'Changed production prediction'):p.choose_pair(row,jobs,True)


if __name__=='__main__':unittest.main()
