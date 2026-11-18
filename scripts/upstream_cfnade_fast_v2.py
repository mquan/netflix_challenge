#!/usr/bin/env python3
"""Exact CFNADE projections, per-user forward reuse, and grouped prediction."""
import shutil
import subprocess
from pathlib import Path
import upstream_repro as u
import upstream_cfnade_fast as first

BASE=u.ROOT/'artifacts/upstream-cfnade-fast-v2'
TREE=BASE/'source'
REPORT=u.ROOT/'experiments/upstream-cfnade-fast-v2'


def prepare():
    first.BASE,first.TREE,first.REPORT=BASE,TREE,REPORT
    first.prepare();text=(TREE/'src/cfnade.rs').read_text()
    helper='''    fn training_forward(&self,ctx_sum:&[f64])->(Vec<f32>,Vec<f32>,Vec<f32>){
        let h:Vec<f32>=exact_matvec(&self.enc_proj,ctx_sum,Some(self.enc_bias.as_slice().unwrap()))
            .into_iter().map(|s|tanh(s as f32)).collect();
        let top=if self.cfg.n_hidden2>0{
            let input:Vec<f64>=h.iter().map(|&v|v as f64).collect();
            exact_matvec(&self.hidden2_proj,&input,Some(self.hidden2_bias.as_slice().unwrap()))
                .into_iter().map(|s|tanh(s as f32)).collect()
        }else{h.clone()};
        let input:Vec<f64>=top.iter().map(|&v|v as f64).collect();
        let z=exact_matvec(&self.dec_proj,&input,None).into_iter().map(|s|s as f32).collect();
        (h,top,z)
    }

'''
    start=text.index('    fn encode_ctx_sum(');end=text.index('    /// Sums the A factors',start)
    text=text[:start]+helper+'''    fn encode_ctx_sum(&self,ctx_sum:&[f64])->Array1<f32>{
        let (_,_,z)=self.training_forward(ctx_sum);Array1::from_vec(z)
    }

'''+text[end:]
    process=text.index('    fn process_user(')
    start=text.index('            let mut h = vec![0.0; hidden];',process)
    end=text.index('            let mut grad_z =',start)
    text=text[:start]+'''            let (h,top,z)=if !self.cfg.use_ctx_target_day_scale {
                if let Some(ref cached)=shared_forward {cached.clone()} else {
                    let values=self.training_forward(&ctx_sum);
                    shared_forward=Some(values.clone());values
                }
            }else{self.training_forward(&ctx_sum)};

'''+text[end:]
    anchor='        for target_day_index in 0..DAY_BINS {\n            if suffix_by_day[target_day_index].is_empty()'
    text=first.once(text,anchor,
        '        let mut shared_forward:Option<(Vec<f32>,Vec<f32>,Vec<f32>)>=None;\n'+anchor)
    grouped='''    fn groups_by_user(&self)->bool{true}

    fn predict_user_rows(&self,ds:&Dataset,u:usize,rows:&[usize],out:&mut[f32]){
        let ratings=&self.user_ratings[u];
        let mut latents:HashMap<usize,Array1<f32>>=HashMap::new();
        for (value,&idx) in out.iter_mut().zip(rows){
            let day=ds.dates[idx] as i32;
            let key=if self.cfg.use_ctx_target_day_scale{self.day_index(day)}else{0};
            let z=latents.entry(key).or_insert_with(||self.encode_user_for_day(u,ratings,day));
            *value=self.predict_with_z(z.as_slice().unwrap(),u,ds.item_idxs[idx] as usize,day);
        }
    }

'''
    anchor='    /// Predicts a rating, reusing the cached per-user/day-bin latent z when possible.'
    text=first.once(text,anchor,grouped+anchor)
    text+='\ninclude!("cfnade_kernels.rs");\n'
    (TREE/'src/cfnade.rs').write_text(text)
    shutil.copyfile(u.ROOT/'research/cfnade_fast/kernels.rs',TREE/'src/cfnade_kernels.rs')
    proof=(u.ROOT/'research/cfnade_fast/parity.rs').read_text()
    proof=proof.replace('for extra in [false,true]','for style in 0..4')
    proof=proof.replace('config!(cfnade_original,hidden,rank,h2,extra)','config!(cfnade_original,hidden,rank,h2,style==1)')
    proof=proof.replace('config!(cfnade,hidden,rank,h2,extra)','config!(cfnade,hidden,rank,h2,style==1)')
    proof=proof.replace('        let mut a=','        let extra=style==1;\n        let mut a=')
    # Additional independent context/implicit combinations exercise cache guards.
    proof=proof.replace('        for epoch in 0..=7{','''        if style==2 || style==3 {
            let mut ac=config!(cfnade_original,hidden,rank,h2,false);
            let mut bc=config!(cfnade,hidden,rank,h2,false);
            ac.use_implicit_probe_ctx=style==2;bc.use_implicit_probe_ctx=style==2;
            ac.use_ctx_target_day_scale=style==3;bc.use_ctx_target_day_scale=style==3;
            a=netflix_prize::cfnade_original::CfNadeModel::new(&tr,&masked,ac);
            b=netflix_prize::cfnade::CfNadeModel::new(&tr,&masked,bc);
        }
        for epoch in 0..=7{''')
    proof=proof.replace('            }}\n        }','''            }
              for u in 0..ds.n_users{
                let rows:Vec<usize>=(0..ds.n_ratings).filter(|&idx|ds.user_idxs[idx] as usize==u).collect();
                let mut out=vec![0.0;rows.len()];b.predict_user_rows(ds,u,&rows,&mut out);
                for (&idx,&value) in rows.iter().zip(&out){
                    assert_eq!(a.predict(u,ds.item_idxs[idx] as usize,ds.dates[idx] as i32).to_bits(),value.to_bits(),"grouped prediction epoch{epoch} style{style}");
                }
              }
            }
        }''')
    (TREE/'src/bin/cfnade-fast-parity.rs').write_text(proof)
    patch=subprocess.check_output(['git','diff','--','src/cfnade.rs','src/lib.rs'],cwd=TREE)
    (REPORT/'candidate.patch').write_bytes(patch)
    record=dict(source_pin=u.PIN,script_sha256=u.sha(Path(__file__)),base_script_sha256=u.sha(Path(first.__file__)),
        patch_sha256=u.sha(REPORT/'candidate.patch'),files={p:u.sha(TREE/p) for p in
          ['src/cfnade.rs','src/cfnade_original.rs','src/cfnade_kernels.rs','src/lib.rs','src/bin/cfnade-fast-parity.rs']},
        scope='Candidate exact arithmetic optimization; model capacities, update order and optimizers unchanged. No full-scale performance claim.')
    u.write(REPORT/'preparation.json',record);return record


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
    if result.returncode:raise ValueError('CFNADE v2 exact parity failed')


if __name__=='__main__':verify()
