use netflix_prize::{Dataset,MaskedDataset,Regressor,OrdinalHeadConfig};
use ndarray::Array1;
use rand::{rngs::StdRng,SeedableRng};

macro_rules! config {
    ($module:ident,$features:expr) => {{
        use netflix_prize::$module::TxConfig;
        let n_feat=$features;
        let n_epochs=5;
        let max_neighbors=7;
        let w_factor=0.0015;
        PUBLISHED_CONFIG
    }};
}

fn dataset(probe:bool)->Dataset {
    let n_users=9;let n_items=11;
    let mut users=vec![];let mut items=vec![];let mut dates=vec![];let mut ratings=vec![];
    if probe {
        // Item-major prediction rows, shared days, unseen days, and a probe-only user.
        for i in 0..n_items {for u in 0..n_users {
            if (i+u)%3!=0 {continue;}
            users.push(u as i32);items.push(i as i32);
            dates.push(if i%2==0{1}else{6});ratings.push(0i8);
        }}
    } else {
        for u in 0..7 {for t in 0..13 {
            users.push(u as i32);items.push(((u*3+t)%n_items) as i32);
            dates.push((1+t/3) as i16);ratings.push((1+(u+t)%5) as i8);
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
      dates:Array1::from_vec(dates),is_test:Array1::zeros(n_ratings),name:if probe{"probex"}else{"trainx"}.into(),
      n_users,n_items,n_ratings,transposed:false}
}

fn check(a:&netflix_prize::tx_original::TxModel,b:&netflix_prize::tx::TxModel,tr:&Dataset,pr:&Dataset,case:&str,epoch:usize){
    assert_eq!(a.audit_parameters(),b.audit_parameters(),"parameters: {case}, epoch {epoch}");
    for ds in [tr,pr] {
        for idx in 0..ds.n_ratings {
            let u=ds.user_idxs[idx] as usize;let i=ds.item_idxs[idx] as usize;let d=ds.dates[idx] as i32;
            assert_eq!(a.predict(u,i,d).to_bits(),b.predict(u,i,d).to_bits(),"prediction: {case} epoch {epoch} row {idx}");
            let aa=a.predict_subscores(u,i,d);let bb=b.predict_subscores(u,i,d);
            assert_eq!(aa.mapv(f32::to_bits),bb.mapv(f32::to_bits),"components: {case} epoch {epoch} row {idx}");
        }
        for u in 0..ds.n_users {
            let rows:Vec<usize>=(0..ds.n_ratings).filter(|&i|ds.user_idxs[i] as usize==u).collect();
            let mut aa=vec![0.0;rows.len()];let mut bb=aa.clone();
            a.predict_user_rows(ds,u,&rows,&mut aa);b.predict_user_rows(ds,u,&rows,&mut bb);
            assert_eq!(aa.iter().map(|x|x.to_bits()).collect::<Vec<_>>(),bb.iter().map(|x|x.to_bits()).collect::<Vec<_>>(),
                "grouped prediction: {case} epoch {epoch} user {u}");
        }
    }
}

fn main(){
    netflix_prize::suppress_progress();
    // This crosses the eviction threshold and verifies persistence after flush/advice.
    let mut rng=StdRng::seed_from_u64(100);
    let mut expected=netflix_prize::rand_array2(6000,3000,&mut rng,0.005);
    let mut rng=StdRng::seed_from_u64(100);
    let mut actual=netflix_prize::tx_storage::StorageMatrix::normal(6000,3000,&mut rng,0.005);
    for row in (0..6000).step_by(53){
        expected.row_mut(row).mapv_inplace(|v|v+0.25);
        actual.row_mut(row).mapv_inplace(|v|v+0.25);
        actual.release_rows(row,row+1);
    }
    actual.release_rows(0,6000);
    assert!(actual.iter().zip(expected.iter()).all(|(a,b)|a.to_bits()==b.to_bits()),"72 MB matrix persistence");
    drop(actual);drop(expected);
    let tr=dataset(false);let pr=dataset(true);let masked=MaskedDataset::from(&pr);
    let mut cases=0;
    for nf in [0,1,6,17,60,150,400,1200,4800] {for low_mem in [false,true] {for full_su in [false,true] {for ordinal in [false,true] {
      for bug in [false,true] {for neighbors in [0,7] {
        let mut ac=config!(tx_original,nf);let mut bc=config!(tx,nf);
        ac.low_memory=low_mem;bc.low_memory=low_mem;ac.full_su=full_su;bc.full_su=full_su;
        ac.reset_u_epoch=3;bc.reset_u_epoch=3;
        ac.sum_err_bug=bug;bc.sum_err_bug=bug;
        ac.max_neighbors=neighbors;bc.max_neighbors=neighbors;
        if !ordinal{ac.ordinal_head=None;bc.ordinal_head=None;}
        let case=format!("nf={nf}, low_mem={low_mem}, full_su={full_su}, ordinal={ordinal}, bug={bug}, neighbors={neighbors}");
        let mut a=netflix_prize::tx_original::TxModel::new(&tr,&masked,ac);
        let mut b=netflix_prize::tx::TxModel::new(&tr,&masked,bc);
        check(&a,&b,&tr,&pr,&case,0);
        for epoch in 1..=5{
            a.fit_epoch(&tr,&masked,epoch);b.fit_epoch(&tr,&masked,epoch);
            check(&a,&b,&tr,&pr,&case,epoch);
        }
        cases+=1;
    }}}}}}
    println!("PASS: {cases} cases; initialization plus five epochs; bitwise parameters, row/grouped predictions, components");
}
