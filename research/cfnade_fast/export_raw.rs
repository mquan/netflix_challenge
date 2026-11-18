//! Export the independently verified raw-SGD counterpart of a completed SWA run.
use netflix_prize::{Dataset,MaskedDataset,Regressor};
use netflix_prize::cfnade::{CfNadeConfig,CfNadeModel};
use ndarray_npy::write_npy;

fn main(){
    netflix_prize::suppress_progress();
    let cfg=CfNadeConfig{n_hidden:320,rank:96,n_epochs:95,..CfNadeConfig::default()};
    for (train,query) in [("trainx","probex"),("fulltrain","qual")]{
        let tr=Dataset::load(train,"rtg","preds_new");let pr=Dataset::load(query,"rtg","preds_new");
        let mut model=CfNadeModel::new(&tr,&MaskedDataset::from(&pr),cfg);
        assert_eq!(model.resumed_epoch(),95,"raw export requires complete canonical training");
        let prediction=model.predict_raw_sgd(&pr);
        write_npy(format!("preds_new/cfnade-96r.{query}.npy"),&prediction).unwrap();
        write_npy(format!("preds_new/cfnade-96r_ep95.{query}.npy"),&prediction).unwrap();
        println!("Exported raw SGD trajectory: {train} => {query}, {} rows",prediction.len());
    }
    std::fs::write("preds_new/cfnade-96r.out",
        "Fresh canonical cfnade-96 SGD trajectory, exported from saved raw weights before averaging.\nSame gradient trajectory as independent cfnade-96r (swa_start=usize::MAX), verified in interrupted/native-driver parity tests.\nNo second training run or imported prediction features.\n").unwrap();
}
