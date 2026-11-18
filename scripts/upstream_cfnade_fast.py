#!/usr/bin/env python3
"""Prototype contiguous CFNADE gradient loops; require exact source parity."""
import re
import shutil
import subprocess
from pathlib import Path
import upstream_repro as u

BASE=u.ROOT/'artifacts/upstream-cfnade-fast'
TREE=BASE/'source'
REPORT=u.ROOT/'experiments/upstream-cfnade-fast'


def once(text,old,new):
    if text.count(old)!=1:raise ValueError('CF source anchor differs: '+old[:100])
    return text.replace(old,new)


def prepare():
    BASE.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    if not TREE.exists():
        subprocess.run(['git','clone','--shared',str(u.SOURCE),str(TREE)],check=True)
        subprocess.run(['git','checkout','--detach',u.PIN],cwd=TREE,check=True)
    def pinned(path):return subprocess.check_output(['git','show',u.PIN+':'+path],cwd=u.SOURCE,text=True)
    original=pinned('src/cfnade.rs')
    fields=re.findall(r'^    (\w+): Array[123]<f32>',original.split('pub struct CfNadeModel {',1)[1].split('\n}',1)[0],re.M)
    audit='\nimpl CfNadeModel { pub fn audit_parameters(&self)->Vec<u32> {\n'
    audit+='let mut out=vec![self.timestep as u32,self.swa_count as u32,self.swapped_in as u32];\n'
    audit+='\n'.join('out.extend(self.'+name+'.iter().map(|v|v.to_bits()));' for name in fields)
    audit+='\nlet mut rng=self.rng.clone();for _ in 0..8 {out.push(rng.random::<u32>());}\nout\n}}\n'
    (TREE/'src/cfnade_original.rs').write_text(original+audit)
    patched=once(original,'''            for j in 0..rank {
                for i in 0..top.len() {
                    grads.dec_proj[[j, i]] += (grad_z[j] * top[i] as f64) as f32;
                }
            }''','''            for j in 0..rank {
                let mut row=grads.dec_proj.row_mut(j);
                for (g,&value) in row.iter_mut().zip(top.iter()) {
                    *g += (grad_z[j] * value as f64) as f32;
                }
            }''')
    patched=once(patched,'''            } else {
                for i in 0..hidden {
                    let mut s = 0.0;
                    for j in 0..rank {
                        s += self.dec_proj[[j, i]] as f64 * grad_z[j];
                    }
                    grad_h[i] = s * (1.0 - h[i] as f64 * h[i] as f64);
                    grads.enc_bias[i] += grad_h[i] as f32;
                }
            }''','''            } else {
                // Each output still sums j in exactly the original order.
                for j in 0..rank {
                    let row=self.dec_proj.row(j);
                    for (sum,&weight) in grad_h.iter_mut().zip(row.iter()) {
                        *sum += weight as f64 * grad_z[j];
                    }
                }
                for i in 0..hidden {
                    grad_h[i] *= 1.0 - h[i] as f64 * h[i] as f64;
                    grads.enc_bias[i] += grad_h[i] as f32;
                }
            }''')
    patched=once(patched,'''            for j in 0..rank {
                for i in 0..hidden {
                    grads.enc_proj[[i, j]] += (grad_h[i] * ctx_sum[j]) as f32;
                }
            }''','''            for i in 0..hidden {
                let mut row=grads.enc_proj.row_mut(i);
                for (g,&value) in row.iter_mut().zip(ctx_sum.iter()) {
                    *g += (grad_h[i] * value) as f32;
                }
            }''')
    patched=once(patched,'''            let mut grad_ctx_sum = vec![0.0; rank];
            for j in 0..rank {
                let mut s = 0.0;
                for i in 0..hidden {
                    s += self.enc_proj[[i, j]] as f64 * grad_h[i];
                }
                grad_ctx_sum[j] = s * ctx_scale as f64;
            }''','''            let mut grad_ctx_sum = vec![0.0; rank];
            for i in 0..hidden {
                let row=self.enc_proj.row(i);
                for (sum,&weight) in grad_ctx_sum.iter_mut().zip(row.iter()) {
                    *sum += weight as f64 * grad_h[i];
                }
            }
            for value in &mut grad_ctx_sum { *value *= ctx_scale as f64; }''')
    (TREE/'src/cfnade.rs').write_text(patched+audit)
    (TREE/'src/lib.rs').write_text(pinned('src/lib.rs')+'\npub mod cfnade_original;\n')
    shutil.copyfile(u.FRESH/'data/qual_ratings/qual_ratings.csv.gz',TREE/'data/qual_ratings/qual_ratings.csv.gz')
    shutil.copyfile(u.ROOT/'research/cfnade_fast/parity.rs',TREE/'src/bin/cfnade-fast-parity.rs')
    patch=subprocess.check_output(['git','diff','--','src/cfnade.rs','src/lib.rs'],cwd=TREE)
    (REPORT/'candidate.patch').write_bytes(patch)
    record=dict(source_pin=u.PIN,patch_sha256=u.sha(REPORT/'candidate.patch'),
        script_sha256=u.sha(Path(__file__)),audit_parameter_arrays=len(fields),
        files={name:u.sha(TREE/name) for name in ['src/cfnade.rs','src/cfnade_original.rs','src/lib.rs','src/bin/cfnade-fast-parity.rs']},
        scope='Candidate loop-layout optimization only; preserve per-cell accumulation order. No training speed or quality claim before measurement.')
    u.write(REPORT/'preparation.json',record)
    return record


def verify():
    record=prepare();env=u.environment();env.update(CARGO_TARGET_DIR=str(BASE/'target'),CARGO_BUILD_JOBS='2')
    with (REPORT/'build.log').open('w') as log:
        subprocess.run([str(u.BASE/'toolchain/cargo/bin/cargo'),'build','--release','--offline','--bin','cfnade-fast-parity'],
                        cwd=TREE,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    binary=BASE/'target/release/cfnade-fast-parity'
    with (REPORT/'parity.log').open('w') as log:
        result=subprocess.run([str(binary)],cwd=TREE,env=env,stdout=log,stderr=subprocess.STDOUT)
    u.write(REPORT/'verification.json',dict(pass_all=result.returncode==0,preparation=record,
        executable_sha256=u.sha(binary),log_sha256=u.sha(REPORT/'parity.log')))
    if result.returncode:raise ValueError('CFNADE exact parity failed')


if __name__=='__main__':verify()
