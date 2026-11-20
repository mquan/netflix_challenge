#!/usr/bin/env python3
"""Reconstruct every retained and added own330 neural input bit independently."""
import os
for key in ['OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS']:os.environ[key]='2'
import hashlib
from pathlib import Path
import numpy as np
import upstream_own330_neural_inputs as data

u=data.u


def main():
    path=data.REPORT/'independent-verification.json'
    if path.exists():raise ValueError('Existing complete own330 neural input replay preserved')
    m=u.read(data.REPORT/'manifest.json');data.check(m['source_sha256'])
    _,rows=data.sources()
    if m['recent_rows']!=rows:raise ValueError('Every fixed original input and order required')
    files=[u.ROOT/'artifacts/upstream-warm166/design.npy',u.ROOT/'artifacts/upstream-warm166/features.npy']+[u.ROOT/r['paired_files']['probex'] for r in rows]
    streams=[]
    for file in files:
        stream=file.open('rb');version=np.lib.format.read_magic(stream)
        if version!=(1,0):raise ValueError('Original NPY header version required')
        shape,fortran,dtype=np.lib.format.read_array_header_1_0(stream)
        if fortran:raise ValueError('Original C order required')
        streams.append((stream,dtype))
    actual=data.Inputs(rows);hashes=[hashlib.sha256(),hashlib.sha256()]
    try:
        for lo in range(0,data.ROWS,8192):
            hi=min(lo+8192,data.ROWS);count=hi-lo
            old_z=np.fromfile(streams[0][0],dtype=streams[0][1],count=count*242).reshape(count,242)
            old_f=np.fromfile(streams[1][0],dtype=streams[1][1],count=count*392).reshape(count,392)
            z=np.empty((count,406),dtype=np.float64);f=np.empty((count,556),dtype=np.float32)
            z[:,:241]=old_z[:,:241];z[:,-1]=old_z[:,-1];f[:,:392]=old_f
            for j,(stream,dtype) in enumerate(streams[2:]):
                values=np.fromfile(stream,dtype=dtype,count=count)
                if values.shape!=(count,):raise ValueError('Every complete original scalar payload required')
                clipped=np.maximum(np.minimum(values,np.float32(6)),np.float32(0))
                z[:,241+j]=clipped;f[:,392+j]=clipped-np.float32(3.6)
            az,af=actual.block(lo,hi)
            if not np.array_equal(z.view('u8'),az.view('u8')) or not np.array_equal(f.view('u4'),af.view('u4')):
                raise ValueError('Every retained/new design and nonlinear input bit must match')
            hashes[0].update(z.tobytes(order='C'));hashes[1].update(f.tobytes(order='C'))
    finally:
        actual.close()
        for stream,_ in streams:stream.close()
    if [h.hexdigest() for h in hashes]!=m['virtual_c_order_sha256']:raise ValueError('Complete original input fingerprints differ')
    if hashes[0].hexdigest()!=u.read(data.raw.REPORT/'manifest.json')['full_design_c_order_sha256']:
        raise ValueError('Original complete ridge design differs')
    data.check(m['source_sha256'])
    u.write(path,dict(state='complete_all_original330_design_and_nonlinear_input_bits_replayed',
            manifest_sha256=u.sha(data.REPORT/'manifest.json'),verifier_sha256=u.sha(Path(__file__)),rows=data.ROWS,
            raw_predictions=330,context_columns=90,design_columns=406,nonlinear_columns=556,nn_inputs=557,
            all_original392_nonlinear_columns_preserved=True,all164_scalar_transforms_bitwise_equal=True,
            complete_virtual_c_order_sha256=[h.hexdigest() for h in hashes],qualifying_labels_read=False,neural_models_fitted=0,
            scope='Independent complete input operators only. Full original eligible557-input scalers, fivefold/three-seed fitting and all actual saved-model replays remain required.'))
    print('Independent complete own330406/556 operators: every retained/new bit matches',flush=True)


if __name__=='__main__':main()
