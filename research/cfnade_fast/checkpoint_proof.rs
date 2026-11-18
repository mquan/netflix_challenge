use netflix_prize::{Dataset,MaskedDataset,Regressor,Fit2Opts,SPLIT_NEW};
use ndarray::Array1;
use ndarray_npy::{write_npy,read_npy};
use std::path::{Path,PathBuf};

mod fixture{
    include!("cfnade-fast-parity.rs");
    pub fn data(probe:bool)->netflix_prize::Dataset{dataset(probe)}
}
macro_rules! config{($module:ident)=>{{
    let mut cfg=netflix_prize::$module::CfNadeConfig::default();
    cfg.n_hidden=40;cfg.rank=12;cfg.n_epochs=7;cfg.batch_size=3;cfg.swa_start=3;cfg.milestones=[4,6,0,0];cfg
}};}
fn save_dataset(ds:&Dataset,name:&str){
    let root=PathBuf::from("data").join(name);std::fs::create_dir_all(&root).unwrap();
    write_npy(root.join("user_idxs.npy"),&ds.user_idxs).unwrap();
    write_npy(root.join("user_cnts.npy"),&ds.user_cnts).unwrap();
    write_npy(root.join("item_idxs.npy"),&ds.item_idxs.mapv(|i|i as i16)).unwrap();
    write_npy(root.join("item_cnts.npy"),&ds.item_cnts).unwrap();
    write_npy(root.join("item_years.npy"),&ds.item_years).unwrap();
    write_npy(root.join("ratings.npy"),&ds.raw_ratings).unwrap();
    write_npy(root.join("dates.npy"),&ds.dates).unwrap();
    write_npy(root.join("is_test.npy"),&ds.is_test).unwrap();
}
fn driver(){
    let options=Fit2Opts{save_probe_each_epoch:true,..Fit2Opts::default()};
    netflix_prize::fit2_inner::<netflix_prize::cfnade::CfNadeModel>(config!(cfnade),"rtg","checkpoint-driver",SPLIT_NEW,options);
}
fn native_fixture(probe:bool)->Dataset{
    // The pinned file loader fixes the full Netflix user/movie universe.
    // Keep toy observed rows while padding their count/year metadata accordingly.
    let mut ds=fixture::data(probe);
    let mut users=Array1::zeros(netflix_prize::N_USERS);
    users.slice_mut(ndarray::s![..ds.user_cnts.len()]).assign(&ds.user_cnts);
    let mut items=Array1::zeros(17_770);
    items.slice_mut(ndarray::s![..ds.item_cnts.len()]).assign(&ds.item_cnts);
    ds.user_cnts=users;ds.item_cnts=items;ds.item_years=Array1::from_elem(17_770,2000);
    ds.n_users=netflix_prize::N_USERS;ds.n_items=17_770;ds
}
fn main(){
    netflix_prize::suppress_progress();
    let mode=std::env::args().nth(1).unwrap();let out=PathBuf::from(std::env::var_os("CF_PROOF_OUTPUT").unwrap());
    let interrupted:usize=std::env::var("CF_PROOF_INTERRUPT").unwrap_or_else(|_|"3".into()).parse().unwrap();
    let tr=native_fixture(false);let pr=native_fixture(true);let masked=MaskedDataset::from(&pr);
    if mode=="create"{
        for name in ["trainx","fulltrain"]{save_dataset(&tr,name);}
        for name in ["probex","qual"]{save_dataset(&pr,name);}
        let mut baseline=netflix_prize::cfnade_original::CfNadeModel::new(&tr,&masked,config!(cfnade_original));
        for epoch in 1..=7{
            baseline.fit_epoch(&tr,&masked,epoch);
            if epoch==interrupted{write_npy(out.join("golden-restored-epoch.npy"),&netflix_prize::gen_preds_parallel(&baseline,&pr)).unwrap();}
        }
        write_npy(out.join("golden-state.npy"),&Array1::from_vec(baseline.audit_parameters())).unwrap();
        write_npy(out.join("golden-prediction.npy"),&netflix_prize::gen_preds_parallel(&baseline,&pr)).unwrap();
        drop(baseline);
        let mut raw_cfg=config!(cfnade_original);raw_cfg.swa_start=usize::MAX;
        let mut raw=netflix_prize::cfnade_original::CfNadeModel::new(&tr,&masked,raw_cfg);
        for epoch in 1..=7{raw.fit_epoch(&tr,&masked,epoch);}
        write_npy(out.join("golden-raw-prediction.npy"),&netflix_prize::gen_preds_parallel(&raw,&pr)).unwrap();
        drop(raw);driver();panic!("expected deliberate exit after epoch {interrupted}");
    }else if mode=="resume"{
        // Simulate an absent last-epoch export while the durable optimizer state survives.
        let snapshot=PathBuf::from(format!("preds_new/checkpoint-driver_ep{interrupted:02}.probex.npy"));
        std::fs::remove_file(&snapshot).unwrap();
        driver();
        let golden:Array1<f32>=read_npy(out.join("golden-prediction.npy")).unwrap();
        for split in ["probex","qual"]{
            let pred:Array1<f32>=read_npy(format!("preds_new/checkpoint-driver.{split}.npy")).unwrap();
            assert_eq!(golden.mapv(f32::to_bits),pred.mapv(f32::to_bits),"native driver final {split}");
        }
        let original:Array1<f32>=read_npy(out.join("golden-restored-epoch.npy")).unwrap();
        let restored:Array1<f32>=read_npy(&snapshot).unwrap();
        assert_eq!(original.mapv(f32::to_bits),restored.mapv(f32::to_bits),"regenerated snapshot");
        let state:Array1<u32>=read_npy(out.join("golden-state.npy")).unwrap();
        let mut loaded=netflix_prize::cfnade::CfNadeModel::new(&tr,&masked,config!(cfnade));
        assert_eq!(loaded.resumed_epoch(),7);assert_eq!(loaded.audit_parameters(),state.to_vec());
        let expected_raw:Array1<f32>=read_npy(out.join("golden-raw-prediction.npy")).unwrap();
        let actual_raw=loaded.predict_raw_sgd(&pr);
        assert_eq!(expected_raw.mapv(f32::to_bits),actual_raw.mapv(f32::to_bits),"derived raw SGD matches independent no-SWA source");
        assert_eq!(loaded.audit_parameters(),state.to_vec(),"raw export preserves averaged/live/Adam/RNG state");drop(loaded);
        let root=PathBuf::from(std::env::var_os("CF_CHECKPOINT_DIR").unwrap());
        for split in ["trainx","fulltrain"]{
            let count=std::fs::read_dir(root.join(split)).unwrap().filter(|e|e.as_ref().unwrap().file_name().to_string_lossy().starts_with("epoch-")).count();
            assert_eq!(count,2,"bounded complete checkpoints");
        }
        println!("PASS: interrupted native two-phase driver equals uninterrupted pinned model; 84 parameter/SWA/Adam arrays, exact RNG, both final vectors and restored epoch snapshot; two checkpoints retained; raw SGD export equals independent no-SWA source and preserves state");
    }else if mode=="load"{
        let _model=netflix_prize::cfnade::CfNadeModel::new(&tr,&masked,config!(cfnade));
        println!("Checkpoint accepted; rejection test must fail if this exits successfully");
    }else{panic!("unknown proof stage");}
}
