#!/usr/bin/env python3
"""Prototype contiguous sparse Adam factor rows with exact checkpoint parity."""
from pathlib import Path
import os
import subprocess
import upstream_repro as u
import upstream_cfnade_checkpoint as inherited

BASE=u.ROOT/'artifacts/upstream-cfnade-sparse-rows'
TREE=BASE/'source'
REPORT=u.ROOT/'experiments/upstream-cfnade-sparse-rows'


def prepare():
    inherited.BASE,inherited.TREE,inherited.REPORT=BASE,TREE,REPORT
    previous=inherited.prepare()
    path=TREE/'src/cfnade.rs';text=path.read_text()
    (TREE/'src/cfnade_baseline.rs').write_text(text)
    begin=text.index('    fn apply_sparse_row_updates(')
    end=text.index('    /// Adam step for sparse rows of a 2D matrix',begin)
    before=text[begin:end]
    header=before[:before.index('        let bc =')]
    replacement=header+'''        let shape=param.dim();
        assert_eq!(shape,m.dim());assert_eq!(shape,v.dim());assert_eq!(shape.0,K);
        let items=shape.1;let rank=shape.2;
        let params=param.as_slice_mut().expect("contiguous factor rows");
        let moments=m.as_slice_mut().expect("contiguous first moments");
        let variances=v.as_slice_mut().expect("contiguous second moments");
        let bc = batch_count.max(1) as f32;
        let t = timestep as i32;
        let corr1 = 1.0 - beta1.powi(t);
        let corr2 = 1.0 - beta2.powi(t);
        for rating in 0..K {
            for (&item,row_grad) in &grads[rating] {
                assert!(item<items);assert_eq!(row_grad.len(),rank);
                let start=(rating*items+item)*rank;let end=start+rank;
                for (((p,m_t),v_t),&gradient) in params[start..end].iter_mut()
                    .zip(moments[start..end].iter_mut()).zip(variances[start..end].iter_mut())
                    .zip(row_grad.iter()) {
                    let g = gradient / bc;
                    *m_t = beta1 * *m_t + (1.0 - beta1) * g;
                    *v_t = beta2 * *v_t + (1.0 - beta2) * g * g;
                    let mhat = *m_t / corr1;
                    let vhat = *v_t / corr2;
                    *p -= lr * mhat / (vhat.sqrt() + eps);
                    *p -= lr * weight_decay * *p;
                }
            }
        }
    }

'''
    path.write_text(text[:begin]+replacement+text[end:])
    lib=TREE/'src/lib.rs';lib.write_text(lib.read_text()+'\npub mod cfnade_baseline;\n')
    patch=subprocess.check_output(['git','diff','--','src/cfnade.rs','src/lib.rs','Cargo.toml'],cwd=TREE)
    (REPORT/'candidate.patch').write_bytes(patch)
    record=dict(previous,script_sha256=u.sha(Path(__file__)),inherited_preparation=previous,
        patch_sha256=u.sha(REPORT/'candidate.patch'),
        files={name:u.sha(TREE/name) for name in list(previous['files'])+['src/cfnade_baseline.rs']},
        scope='Only sparse Adam factor-row traversal changes. Identical scalar arithmetic/order, shape guards, source model/config and checkpoint layout. No current training tree modified.')
    u.write(REPORT/'preparation.json',record);return record


def verify():
    record=prepare();env=u.environment();env.update(CARGO_TARGET_DIR=str(BASE/'target'),CARGO_BUILD_JOBS='2')
    for key in ['CF_CHECKPOINT_DIR','CF_CHECKPOINT_SOURCE','CF_CHECKPOINT_DATA','CF_EXIT_AFTER_CHECKPOINT']:env.pop(key,None)
    with (REPORT/'build.log').open('w') as log:
        subprocess.run([str(u.BASE/'toolchain/cargo/bin/cargo'),'build','--release','--offline',
            '--bin','cfnade-fast-parity','--bin','cfnade-checkpoint-proof'],cwd=TREE,env=env,
            stdout=log,stderr=subprocess.STDOUT,check=True)
    executable=BASE/'target/release/cfnade-fast-parity'
    child=subprocess.run([str(executable)],cwd=TREE,env=env,capture_output=True,text=True,timeout=600)
    (REPORT/'parity.log').write_text(child.stdout+child.stderr)
    if any(u.sha(TREE/name)!=digest for name,digest in record['files'].items()):raise ValueError('Sparse candidate source changed')
    u.write(REPORT/'verification.json',dict(pass_all=child.returncode==0,preparation=record,
        executable_sha256=u.sha(executable),log_sha256=u.sha(REPORT/'parity.log'),
        scope='Inherited24-case source parity through7 epochs, all84 parameter/Adam/SWA arrays and RNG, direct and grouped predictions. Checkpoint executable built; migration/full-data execution and relative speed not established.'))
    if child.returncode:raise ValueError('Sparse row parity failed; preserve artifacts')
    print('Sparse row parity passed',flush=True)


if __name__=='__main__':verify()
