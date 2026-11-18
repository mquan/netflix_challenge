// Exact same initialized model/data: current trainer versus contiguous Adam rows.
use netflix_prize::{Dataset,MaskedDataset,Regressor};
use ndarray::Array1;

fn data(probe:bool)->Dataset{
    let n_users=512;let n_items=2048;let count=if probe{10}else{200};
    let mut users=Vec::new();let mut items=Vec::new();let mut dates=Vec::new();let mut ratings=Vec::new();
    for u in 0..n_users{for t in 0..count{
        users.push(u as i32);items.push(((u*31+t*13+if probe{7}else{0})%n_items) as i32);
        dates.push(((t*71+u*13)%2243) as i16);ratings.push(if probe{0}else{(1+(u+t*3)%5) as i8});
    }}
    let mut uc=vec![0;n_users];let mut ic=vec![0;n_items];
    for &u in &users{uc[u as usize]+=1;}for &i in &items{ic[i as usize]+=1;}
    let n_ratings=users.len();
    Dataset{user_idxs:Array1::from_vec(users),user_cnts:Array1::from_vec(uc),item_idxs:Array1::from_vec(items),
        item_cnts:Array1::from_vec(ic),item_years:Array1::from_elem(n_items,2000),
        residuals:Array1::from_iter(ratings.iter().map(|&r|r as f32)),raw_ratings:Array1::from_vec(ratings),
        dates:Array1::from_vec(dates),is_test:Array1::zeros(n_ratings),n_users,n_items,n_ratings,
        name:if probe{"probex"}else{"trainx"}.into(),transposed:false}
}

fn main(){
    netflix_prize::suppress_progress();let tr=data(false);let pr=data(true);let masked=MaskedDataset::from(&pr);
    for repeat in 0..3{
        let mut cfg=netflix_prize::cfnade_baseline::CfNadeConfig::default();cfg.n_epochs=3;
        let mut cc=netflix_prize::cfnade::CfNadeConfig::default();cc.n_epochs=3;
        let mut baseline=netflix_prize::cfnade_baseline::CfNadeModel::new(&tr,&masked,cfg);
        let mut candidate=netflix_prize::cfnade::CfNadeModel::new(&tr,&masked,cc);
        assert_eq!(baseline.audit_parameters(),candidate.audit_parameters());
        for epoch in 1..=3{
            if repeat%2==0{
                let start=std::time::Instant::now();baseline.fit_epoch(&tr,&masked,epoch);
                println!("BENCH baseline {repeat} {epoch} {:.9}",start.elapsed().as_secs_f64());
                let start=std::time::Instant::now();candidate.fit_epoch(&tr,&masked,epoch);
                println!("BENCH candidate {repeat} {epoch} {:.9}",start.elapsed().as_secs_f64());
            }else{
                let start=std::time::Instant::now();candidate.fit_epoch(&tr,&masked,epoch);
                println!("BENCH candidate {repeat} {epoch} {:.9}",start.elapsed().as_secs_f64());
                let start=std::time::Instant::now();baseline.fit_epoch(&tr,&masked,epoch);
                println!("BENCH baseline {repeat} {epoch} {:.9}",start.elapsed().as_secs_f64());
            }
            assert_eq!(baseline.audit_parameters(),candidate.audit_parameters(),"state repeat{repeat} epoch{epoch}");
            assert_eq!(netflix_prize::gen_preds_parallel(&baseline,&pr).mapv(f32::to_bits),
                       netflix_prize::gen_preds_parallel(&candidate,&pr).mapv(f32::to_bits),"query repeat{repeat} epoch{epoch}");
        }
    }
    println!("PASS: canonical320/96,batch8;512 users x200 ratings,2048 items;three alternating-order repeats of three epochs;all84 state arrays,RNG and query prediction bits equal");
}
