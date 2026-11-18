#!/usr/bin/env python3
"""Create an isolated storage-only temporal implementation and parity executable."""
from pathlib import Path
import shutil
import subprocess
import upstream_repro as u

BASE=u.ROOT/'artifacts/full-upstream-mapped'
TREE=BASE/'source'
SUPPORT=u.ROOT/'research/mapped_tx'


def pinned(path):
    return subprocess.check_output(['git','show',u.PIN+':'+path],cwd=u.SOURCE,text=True)


def replace_once(text,old,new):
    if text.count(old)!=1:raise ValueError(f'Expected one source anchor: {old[:100]}')
    return text.replace(old,new)


def prepare():
    if not TREE.exists():
        subprocess.run(['git','clone','--shared',str(u.SOURCE),str(TREE)],check=True)
        subprocess.run(['git','checkout','--detach',u.PIN],cwd=TREE,check=True)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=TREE,text=True).strip()
    if head!=u.PIN:raise ValueError('Incorrect isolated source pin')
    sentinel='data/qual_ratings/qual_ratings.csv.gz'
    shutil.copyfile(u.FRESH/sentinel,TREE/sentinel)
    original=pinned('src/tx.rs')
    audit='''
impl TxModel {
    pub fn audit_parameters(&self)->Vec<u32> {
        let mut out=vec![self.gbias.to_bits()];
        MATRICES
        VECTORS
        out
    }
}
'''
    matrices=['ufeat','ufeat2','ifeat','yfeat','ycache','yfeat_day','but_bin','bit_bin','ufeat_day','ibias_freq','ifeat_freq']
    vectors=['ubias','ibias','tu_mean','alpha_u','ybias','ycache_bias','ubias_day','cu','cut','w','c','w_day','c_day']
    for marker,fields in [('MATRICES',matrices),('VECTORS',vectors)]:
        audit=audit.replace(marker,'\n'.join(f'out.extend(self.{field}.iter().map(|v|v.to_bits()));' for field in fields))
    (TREE/'src/tx_original.rs').write_text(original+audit)
    patched=original
    patched=replace_once(patched,'use indicatif::ProgressIterator;','use indicatif::ProgressIterator;\nuse crate::tx_storage::StorageMatrix;')
    patched=replace_once(patched,'pub struct TxModel {','pub struct TxModel {\n    storage_cache_ready: bool,')
    helper='''
    fn storage_release_user(&self,u:usize) {
        self.ufeat.release_rows(u,u+1);
        self.ufeat2.release_rows(u,u+1);
        self.ycache.release_rows(u,u+1);
        self.ufeat_day.release_rows(self.ud.starts[u],self.ud.starts[u+1]);
    }
'''
    patched=replace_once(patched,'impl TxModel {\n    #[inline]\n    fn has_nbr',
                         'impl TxModel {\n'+helper+'    #[inline]\n    fn has_nbr')
    patched=replace_once(patched,'    fn predict_day(&self, u: usize, i: usize, day: i32, day_su: &[f32]) -> f32 {',
        '    fn predict_day(&self, u: usize, i: usize, day: i32, day_su: &[f32]) -> f32 {\n        self.storage_release_user(u);')
    patched=replace_once(patched,'    fn predict_subscores(&self, u: usize, i: usize, day: i32) -> Array1<f32> {',
        '    fn predict_subscores(&self, u: usize, i: usize, day: i32) -> Array1<f32> {\n        self.storage_release_user(u);')
    fields=['ufeat','ufeat2','ycache','ufeat_day']
    for field in fields:
        import re
        patched,n=re.subn(r'(    '+field+r': )Array2<f32>',r'\1StorageMatrix',patched)
        if n!=1:raise ValueError('Field anchor '+field)
        patched,n=re.subn(r'(    '+field+r',\s+\w+:\s+)Array2<f32>;',r'\1StorageMatrix;',patched)
        if n!=1:raise ValueError('Accessor anchor '+field)
    patched=replace_once(patched,'ufeat: rand_array2(n_users, cfg.n_feat, &mut rng, cfg.sigma_u)',
                         'ufeat: StorageMatrix::normal(n_users, cfg.n_feat, &mut rng, cfg.sigma_u)')
    for field in ['ufeat2','ycache']:
        patched=replace_once(patched,f'{field}: Array2::zeros((n_users, cfg.n_feat))',f'{field}: StorageMatrix::zeros((n_users, cfg.n_feat))')
    patched=replace_once(patched,'ufeat_day: if low_mem { Array2::zeros((0, ud_dim)) }\n                       else { Array2::zeros((n_ud, ud_dim)) }',
        'ufeat_day: if low_mem { StorageMatrix::zeros((0, ud_dim)) }\n                       else { StorageMatrix::zeros((n_ud, ud_dim)) }')
    patched=replace_once(patched,'fn new(tr: &Dataset, pr: &MaskedDataset, cfg: Self::Config) -> Self {',
        'fn new(tr: &Dataset, pr: &MaskedDataset, mut cfg: Self::Config) -> Self {\n        let storage_cache_ready = cfg.lazy_day;\n        if !cfg.low_memory && std::env::var_os("TX_LAZY_DAY").is_some() { cfg.lazy_day = true; }')
    patched=replace_once(patched,'        Self {\n            cfg,','        Self {\n            storage_cache_ready,\n            cfg,')
    patched=replace_once(patched,'        out.fill(0.0);\n        let mut cnt',
        '        out.fill(0.0);\n        if !self.storage_cache_ready { return; }\n        let mut cnt')
    patched=replace_once(patched,'        if !low_mem && !cfg.lazy_day { self.rebuild_ycache_day(tr, pr); }',
        '        if !low_mem && !cfg.lazy_day { self.rebuild_ycache_day(tr, pr); }\n        self.storage_cache_ready = true;')
    patched=replace_once(patched,'&(&self.ufeat + &self.ycache)','&(&self.ufeat.view() + &self.ycache.view())')
    patched=replace_once(patched,'        self.rebuild_ycache(tr, pr);\n        if !low_mem',
        '        self.ycache.release_rows(0,tr.n_users);\n        self.rebuild_ycache(tr, pr);\n        self.ycache.release_rows(0,tr.n_users);\n        if !low_mem')
    anchor='''                self.ybias[j] -= cfg.lr_yb * (sum_err / norm);
            }
        }
'''
    patched=replace_once(patched,anchor,anchor[:-10]+'''            self.storage_release_user(u);
        }
''')
    (TREE/'src/tx.rs').write_text(patched+audit)
    shutil.copyfile(SUPPORT/'storage.rs',TREE/'src/tx_storage.rs')
    lib=pinned('src/lib.rs')+'\npub mod tx_storage;\npub mod tx_original;\n'
    (TREE/'src/lib.rs').write_text(lib)
    cargo=pinned('Cargo.toml').replace('[dependencies]','[dependencies]\nlibc = "0.2"')
    (TREE/'Cargo.toml').write_text(cargo)
    # Use the published ordinal temporal configuration for both implementations.
    source=pinned('src/bin/tsvdx5-new.rs')
    cfg=source.split('fn run_ord(',1)[1].split('    let cfg = ',1)[1].split('    fit2!(',1)[0]
    cfg=cfg.strip().removesuffix(';')
    proof=(SUPPORT/'parity.rs').read_text().replace('PUBLISHED_CONFIG',cfg)
    (TREE/'src/bin/tx-storage-parity.rs').write_text(proof)
    patch=subprocess.check_output(['git','diff','--','src/tx.rs','src/lib.rs','Cargo.toml'],cwd=TREE,text=True)
    (BASE/'storage.patch').write_text(patch)
    u.write(BASE/'preparation.json',dict(source_pin=u.PIN,patch_sha256=u.sha(BASE/'storage.patch'),
        support={p.name:u.sha(p) for p in SUPPORT.iterdir() if p.is_file()},
        generated={p:u.sha(TREE/p) for p in ['src/tx.rs','src/tx_original.rs','src/tx_storage.rs',
                    'src/lib.rs','src/bin/tx-storage-parity.rs','Cargo.toml',sentinel]},
        note='Isolated candidate only. No production claim before bitwise parity and full-scale resource validation.'))
    print(TREE)


if __name__=='__main__':prepare()
