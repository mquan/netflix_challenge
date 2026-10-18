// Adapter for gtakacs/netflix_prize commit 855c7404; model source unmodified.
use netflix_prize::{Dataset, MaskedDataset, Regressor, tx::{TxConfig, TxModel}};
use ndarray::Array1;
use std::{fs, io::{Write, BufWriter}, time::Instant};
fn u16at(b: &[u8], p: usize) -> usize { u16::from_le_bytes(b[p..p+2].try_into().unwrap()) as usize }
fn u32at(b: &[u8], p: usize) -> usize { u32::from_le_bytes(b[p..p+4].try_into().unwrap()) as usize }
fn u64at(b: &[u8], p: usize) -> usize { u64::from_le_bytes(b[p..p+8].try_into().unwrap()) as usize }
fn ds(users: usize, items: usize, us: Vec<i32>, it: Vec<i32>, days: Vec<i16>, ratings: Vec<i8>) -> Dataset {
 let n=us.len(); let mut uc=vec![0;users]; let mut ic=vec![0;items];
 for (&u,&i) in us.iter().zip(&it) { uc[u as usize]+=1; ic[i as usize]+=1; }
 Dataset { user_idxs:us.into(), item_idxs:it.into(), dates:days.into(), residuals:ratings.iter().map(|&r|r as f32).collect(), raw_ratings:ratings.into(), user_cnts:uc.into(), item_cnts:ic.into(), item_years:Array1::zeros(items), is_test:Array1::zeros(n), name:"our-split".into(), n_users:users,n_items:items,n_ratings:n,transposed:false }
}
fn main() {
 let args:Vec<String>=std::env::args().collect();
 let bytes=fs::read(&args[1]).unwrap(); let b=&bytes;
 assert_eq!(&b[..8],b"NFPDAT2\0");
 let users=u32at(b,8); let items=u32at(b,12); let n=u64at(b,16); let pn=u64at(b,24); let ni=u64at(b,32);
 let offsets=56; let ti=offsets+8*(users+1); let td=ti+2*n; let tr=td+2*n; let pp=tr+n;
 let io=pp+12*pn; let ii=io+8*(users+1); let id=ii+2*ni; assert_eq!(id+2*ni,b.len());
 let mut us=Vec::with_capacity(n); let mut its=Vec::with_capacity(n); let mut days=Vec::with_capacity(n); let mut rs=Vec::with_capacity(n);
 let mut hu=Vec::new(); let mut hi=Vec::new(); let mut hd=Vec::new(); let mut seen=vec![usize::MAX;items];
 for u in 0..users {
  for p in u64at(b,offsets+8*u)..u64at(b,offsets+8*(u+1)) { let i=u16at(b,ti+2*p)-1; seen[i]=u; us.push(u as i32);its.push(i as i32);days.push(u16at(b,td+2*p) as i16);rs.push(b[tr+p] as i8); }
  for p in u64at(b,io+8*u)..u64at(b,io+8*(u+1)) { let i=u16at(b,ii+2*p)-1; if seen[i]!=u {hu.push(u as i32);hi.push(i as i32);hd.push(u16at(b,id+2*p) as i16);} }
 }
 let train=ds(users,items,us,its,days,rs); let hn=hu.len(); let implicit=ds(users,items,hu,hi,hd,vec![0;hn]);
 let probe:Vec<(usize,usize,i32,f64)>=(0..pn).map(|p| {let x=pp+p*12;(u32at(b,x),u16at(b,x+4)-1,u16at(b,x+6) as i32,b[x+8] as f64)}).collect();
 drop(bytes);
 println!("train={} probe={} heldout_identities={} factors=60 seed=42",n,pn,hn);
 let n_feat=60; let n_epochs=args.get(3).map(|s|s.parse().unwrap()).unwrap_or(20); let seed=42;let n_time_bins=29;let save_ifeat=false;let low_memory=false;
    let cfg = TxConfig {
        n_feat, n_epochs, seed,
        shuffle_users: true,
        n_time_bins,
        beta: 0.3, n_freq_bins: 16,
        lr_u: 0.003, lr_ud: 0.00125, lr_u2: 7e-6,
        lr_ub: 0.0031, lr_ubd: 0.003,
        lr_i: 0.0036, lr_ib: 0.0036,
        lr_y: 0.0005, lr_yb: 2.5e-5, lr_yd: 0.000267,
        lr_tu: 0.0, lr_ti: 0.000225, lr_ta: 2.25e-5,
        lr_ibf: 5e-5, lr_iqf: 5e-6, lr_cu: 0.002,
        reg_iqf: 0.007, reg_cu: 0.01,
        reg_u: 0.0504, reg_u2: 0.4, reg_ud: 0.04,
        reg_i: 0.00735, reg_y: 0.04, reg_yd: 0.02667,
        sigma_iqf: 0.005, sigma_u: 0.0015, sigma_i: 0.005,
        sigma_y: 0.00333, sigma_yd: 0.009,
        reset_u_epoch: 13,
        max_neighbors: 0,
        lr_w: 0.0, lr_c: 0.0, reg_w: 0.0, reg_c: 0.0,
        lr_w_day: 0.0, lr_c_day: 0.0, reg_w_day: 0.0, reg_c_day: 0.0,
        w_bias: 1.0, w_factor: 1.0, w_nbr: 1.0,
        sum_err_bug: false, lambda1: 0.0, lambda2: 0.0,
        ordinal_head: None,
        save_ifeat, low_memory, full_su: true,
    };

 let masked=MaskedDataset::from(&implicit); let mut model=TxModel::new(&train,&masked,cfg);let mut best=f64::INFINITY;
 for epoch in 1..=n_epochs {
 let start=Instant::now(); model.fit_epoch(&train,&masked,epoch);
 let pred:Vec<f32>=probe.iter().map(|&(u,i,d,_)|model.predict(u,i,d)).collect();
 assert!(pred.iter().all(|p|p.is_finite()));
 let rmse=(pred.iter().zip(&probe).map(|(&p,&(_,_,_,r))|(p.clamp(1.,5.) as f64-r).powi(2)).sum::<f64>()/pn as f64).sqrt();
 if rmse<best {best=rmse;let mut out=BufWriter::new(fs::File::create(&args[2]).unwrap());for p in &pred {out.write_all(&p.to_le_bytes()).unwrap();}out.flush().unwrap();}
 println!("epoch={} probe_rmse={:.10} best={:.10} seconds={:.2}",epoch,rmse,best,start.elapsed().as_secs_f64());
 }
}
