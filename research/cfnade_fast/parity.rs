use netflix_prize::{Dataset,MaskedDataset,Regressor};
use ndarray::Array1;
use std::time::Instant;

fn dataset(probe:bool)->Dataset{
    let n_users=9;let n_items=31;
    let mut users=vec![];let mut items=vec![];let mut dates=vec![];let mut ratings=vec![];
    let days=[1,37,99,388,901,1600,2242];
    if !probe {
        for u in 0..8{for t in 0..27{
            users.push(u as i32);items.push(((t+3*u)%n_items) as i32);
            dates.push(days[(t/4)%days.len()]);ratings.push((1+(u+t)%5) as i8);
        }}
    }else{
        for i in 0..n_items{for u in 0..n_users{if (i+u)%7==0{
            users.push(u as i32);items.push(i as i32);dates.push(days[i%days.len()]);ratings.push(0i8);
        }}}
    }
    let mut uc=vec![0;n_users];let mut ic=vec![0;n_items];
    for &u in &users{uc[u as usize]+=1;}
    for &i in &items{ic[i as usize]+=1;}
    let n_ratings=users.len();
    Dataset{user_idxs:Array1::from_vec(users),user_cnts:Array1::from_vec(uc),item_idxs:Array1::from_vec(items),
        item_cnts:Array1::from_vec(ic),item_years:Array1::from_elem(n_items,2000),
        residuals:Array1::from_iter(ratings.iter().map(|&r|r as f32)),raw_ratings:Array1::from_vec(ratings),
        dates:Array1::from_vec(dates),is_test:Array1::zeros(n_ratings),name:if probe{"probex"}else{"trainx"}.into(),
        n_users,n_items,n_ratings,transposed:false}
}

macro_rules! config{($module:ident,$hidden:expr,$rank:expr,$h2:expr,$extra:expr)=>{{
    let mut cfg=netflix_prize::$module::CfNadeConfig::default();
    cfg.n_hidden=$hidden;cfg.rank=$rank;cfg.n_hidden2=$h2;cfg.n_epochs=7;cfg.batch_size=3;
    cfg.swa_start=3;cfg.milestones=[4,6,0,0];
    if $extra{
        cfg.use_ctx_target_day_scale=true;cfg.use_implicit_probe_ctx=true;cfg.use_user_item_scale=true;
        cfg.use_user_item_scale_bin=true;cfg.use_user_day_bias_bin=true;cfg.use_user_day_scale_bin=true;
        cfg.use_day_bias=true;cfg.use_item_time_bias=true;cfg.use_day_freq_bias=true;cfg.use_side_features=true;
    }
    cfg
}};}

fn main(){
    netflix_prize::suppress_progress();let tr=dataset(false);let pr=dataset(true);let masked=MaskedDataset::from(&pr);
    let mut original_seconds=0.0;let mut candidate_seconds=0.0;let mut count=0;
    for (hidden,rank) in [(4,3),(40,12),(320,96)]{for h2 in [0,5]{for extra in [false,true]{
        let mut a=netflix_prize::cfnade_original::CfNadeModel::new(&tr,&masked,config!(cfnade_original,hidden,rank,h2,extra));
        let mut b=netflix_prize::cfnade::CfNadeModel::new(&tr,&masked,config!(cfnade,hidden,rank,h2,extra));
        for epoch in 0..=7{
            if epoch>0{
                let clock=Instant::now();a.fit_epoch(&tr,&masked,epoch);original_seconds+=clock.elapsed().as_secs_f64();
                let clock=Instant::now();b.fit_epoch(&tr,&masked,epoch);candidate_seconds+=clock.elapsed().as_secs_f64();
            }
            assert_eq!(a.audit_parameters(),b.audit_parameters(),"state: H{hidden} R{rank} H2{h2} extras{extra} epoch{epoch}");
            for ds in [&tr,&pr]{for i in 0..ds.n_ratings{
                let u=ds.user_idxs[i] as usize;let j=ds.item_idxs[i] as usize;let d=ds.dates[i] as i32;
                assert_eq!(a.predict(u,j,d).to_bits(),b.predict(u,j,d).to_bits(),"prediction H{hidden} R{rank} H2{h2} extras{extra} epoch{epoch} row{i}");
            }}
        }
        count+=1;
    }}}
    println!("PASS: {count} cases, initialization + 7 epochs, live/SWA/Adam arrays and RNG bits, train/query predictions");
    println!("Synthetic training-only time: original {original_seconds:.6}s, candidate {candidate_seconds:.6}s, ratio {:.3}; not a full-scale runtime estimate",original_seconds/candidate_seconds);
}
