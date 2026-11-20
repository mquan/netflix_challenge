#!/usr/bin/env python3
"""Admit complete own330 design/nonlinear inputs with bounded exact NPY reads."""
import os
for key in ['OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS']:os.environ[key]='2'
import hashlib
from pathlib import Path
import numpy as np
import upstream_own330_native149_data as raw

u=raw.u
RUN=raw.RUN
REPORT=u.ROOT/'experiments/upstream-own330-neural-inputs'
ROWS=raw.ROWS
DESIGN=406
FEATURES=556
check=raw.check
score=raw.score


def sources():
    m=u.read(raw.REPORT/'manifest.json');proof=u.read(raw.REPORT/'independent-verification.json')
    if (proof['state']!='complete_all_original330_ridge_inputs_statistics_solves_and_prediction_bits' or
        proof['manifest_sha256']!=u.sha(raw.REPORT/'manifest.json') or
        m['raw_predictions']!=330 or len(m['control_rows'])!=15 or len(m['added_rows'])!=149):
        raise ValueError('Complete all149 original input/ridge replay required')
    bindings=dict(m['source_sha256'])
    for name,digest in m['prepared_sha256'].items():bindings[str((RUN/name).relative_to(u.ROOT))]=digest
    for path in [Path(__file__),Path(raw.__file__),raw.REPORT/'manifest.json',raw.REPORT/'independent-verification.json',
                 u.ROOT/'OWN330_NEURAL_PLAN.md',u.ROOT/'scripts/verify_upstream_own330_neural_inputs.py']:
        bindings[str(path.relative_to(u.ROOT))]=u.sha(path)
    return bindings,m['control_rows']+m['added_rows']


class Inputs:
    def __init__(self,rows):
        if len(rows)!=164:raise ValueError('All164 fixed additions required')
        self.design=raw.Rows(u.ROOT/'artifacts/upstream-warm166/design.npy',(ROWS,242),np.float64)
        self.features=raw.Rows(u.ROOT/'artifacts/upstream-warm166/features.npy',(ROWS,392),np.float32)
        self.raw=[raw.Rows(u.ROOT/r['paired_files']['probex'],(ROWS,),np.float32) for r in rows]

    def block(self,lo,hi):
        zold=self.design.read(lo,hi);fold=self.features.read(lo,hi)
        added=np.column_stack([np.clip(a.read(lo,hi),0,6) for a in self.raw])
        z=np.column_stack([zold[:,:-1],added,zold[:,-1:]])
        f=np.column_stack([fold,added-np.float32(3.6)])
        if z.shape!=(hi-lo,DESIGN) or f.shape!=(hi-lo,FEATURES) or f.dtype!=np.float32 or not np.isfinite(z).all() or not np.isfinite(f).all():
            raise ValueError('Complete finite original330 input operators required')
        return z,f

    def close(self):
        self.design.close();self.features.close()
        for value in self.raw:value.close()


def main():
    if REPORT.exists():raise ValueError('Existing full neural input admission preserved')
    bindings,rows=sources();check(bindings);REPORT.mkdir()
    hashes=[hashlib.sha256(),hashlib.sha256()];inputs=Inputs(rows)
    try:
        for lo in range(0,ROWS,8192):
            hi=min(lo+8192,ROWS);z,f=inputs.block(lo,hi)
            hashes[0].update(z.tobytes(order='C'));hashes[1].update(f.tobytes(order='C'))
    finally:inputs.close()
    m=u.read(raw.REPORT/'manifest.json')
    if hashes[0].hexdigest()!=m['full_design_c_order_sha256']:raise ValueError('Complete original406-column design must match prior actual replay')
    check(bindings)
    u.write(REPORT/'manifest.json',dict(state='complete_virtual_original330_design_and_nonlinear_operators_pending_replay',
            rows=ROWS,raw_predictions=330,context_columns=90,design_columns=DESIGN,nonlinear_columns=FEATURES,nn_inputs=557,
            recent_rows=rows,source_sha256=bindings,prepared_sha256=m['prepared_sha256'],
            virtual_c_order_sha256=[h.hexdigest() for h in hashes],qualifying_labels_read=False,neural_models_fitted=0,
            scope='All original392 nonlinear inputs retained, all164 fixed own predictors appended. Complete virtual input transport only; separate operator replay and full neural fits/scalers/models still required.'))
    print('Complete own330 virtual406/556 input operators; separate replay required',flush=True)


if __name__=='__main__':main()
