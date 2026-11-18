use netflix_prize::{Dataset,MaskedDataset,Regressor};
use ndarray::Array1;

macro_rules! config {
    ($module:ident,$features:expr,$chunks:expr,$sequential:expr) => {{
        use netflix_prize::$module::Tsvdx4pConfig;
        let n_feat=$features;let save_ifeat=false;
        let seq_epochs=if $sequential{[1,10,20,0,0]}else{[0,0,0,0,0]};
        let mut cfg=PUBLISHED_CONFIG;
        cfg.n_threads=$chunks;
        if n_feat==17{cfg.reset_u_epoch=3;}
        cfg
    }};
}

fn dataset(probe:bool)->Dataset{
    let n_users=32;let n_items=32;
    let mut users=vec![];let mut items=vec![];let mut dates=vec![];let mut ratings=vec![];
    if probe{
        for i in 0..n_items{for u in 0..n_users{
            if (i+u)%5!=0{continue;}
            users.push(u as i32);items.push(i as i32);
            dates.push(if i%2==0{1}else{8});ratings.push(0i8);
        }}
    }else{
        for u in 0..30{for t in 0..8{
            users.push(u as i32);items.push(((u*3+t)%31) as i32);
            dates.push((1+t/2) as i16);ratings.push((1+(u+t)%5) as i8);
        }}
    }
    let mut user_cnts=vec![0;n_users];let mut item_cnts=vec![0;n_items];
    for &u in &users{user_cnts[u as usize]+=1;}
    for &i in &items{item_cnts[i as usize]+=1;}
    let n_ratings=users.len();
    Dataset{user_idxs:Array1::from_vec(users),user_cnts:Array1::from_vec(user_cnts),
        item_idxs:Array1::from_vec(items),item_cnts:Array1::from_vec(item_cnts),
        item_years:Array1::from_elem(n_items,2000),
        residuals:Array1::from_iter(ratings.iter().map(|&v|v as f32)),raw_ratings:Array1::from_vec(ratings),
        dates:Array1::from_vec(dates),is_test:Array1::zeros(n_ratings),
        name:if probe{"probex"}else{"trainx"}.into(),n_users,n_items,n_ratings,transposed:false}
}

fn check(a:&netflix_prize::tsvdx4p_original::Tsvdx4pModel,
         b:&netflix_prize::tsvdx4p::Tsvdx4pModel,tr:&Dataset,pr:&Dataset,case:&str,epoch:usize){
    let aa=a.audit_parameters();let bb=b.audit_parameters();
    assert_eq!(aa.len(),bb.len(),"parameter shape {case} epoch {epoch}");
    if let Some(i)=aa.iter().zip(bb.iter()).position(|(x,y)|x!=y){
        panic!("parameter bits differ {case} epoch {epoch} index {i}: {} != {}",aa[i],bb[i]);
    }
    for ds in [tr,pr]{
        for idx in 0..ds.n_ratings{
            let u=ds.user_idxs[idx] as usize;let i=ds.item_idxs[idx] as usize;let d=ds.dates[idx] as i32;
            assert_eq!(a.predict(u,i,d).to_bits(),b.predict(u,i,d).to_bits(),"prediction {case} epoch {epoch} row {idx}");
            assert_eq!(a.predict_subscores(u,i,d).mapv(f32::to_bits),b.predict_subscores(u,i,d).mapv(f32::to_bits),
                       "components {case} epoch {epoch} row {idx}");
        }
        for u in 0..ds.n_users{
            let rows:Vec<usize>=(0..ds.n_ratings).filter(|&i|ds.user_idxs[i] as usize==u).collect();
            let mut x=vec![0.;rows.len()];let mut y=x.clone();
            a.predict_user_rows(ds,u,&rows,&mut x);b.predict_user_rows(ds,u,&rows,&mut y);
            assert_eq!(x.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),y.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),
                       "grouped prediction {case} epoch {epoch} user {u}");
        }
    }
}

fn main(){
    netflix_prize::suppress_progress();
    let tr=dataset(false);let pr=dataset(true);let masked=MaskedDataset::from(&pr);let mut count=0;
    for nf in [0,1,17,1000,4000,16000]{for chunks in [1,16]{for sequential in [false,true]{
        let ac=config!(tsvdx4p_original,nf,chunks,sequential);
        let bc=config!(tsvdx4p,nf,chunks,sequential);
        assert_eq!(format!("{:?}",ac),format!("{:?}",bc),"configuration differs");
        let case=format!("nf={nf},chunks={chunks},sequential={sequential}");
        let mut a=netflix_prize::tsvdx4p_original::Tsvdx4pModel::new(&tr,&masked,ac);
        let mut b=netflix_prize::tsvdx4p::Tsvdx4pModel::new(&tr,&masked,bc);
        check(&a,&b,&tr,&pr,&case,0);
        for epoch in 1..=20{
            a.fit_epoch(&tr,&masked,epoch);b.fit_epoch(&tr,&masked,epoch);
            check(&a,&b,&tr,&pr,&case,epoch);
        }
        let bytes:Vec<u8>=a.audit_parameters().iter().flat_map(|v|v.to_le_bytes()).collect();
        println!("CASE {case}: {:x}",md5::compute(bytes));count+=1;
    }}}
    println!("PASS: {count} cases, initialization plus20 epochs; all inner parameter/cache bits, row/grouped predictions and components");
}
