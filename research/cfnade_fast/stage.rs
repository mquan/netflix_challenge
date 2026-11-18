//! Run one independent phase through the pinned generic source training driver.
use netflix_prize::{fit,LOG_FILE};
use netflix_prize::cfnade::{CfNadeConfig,CfNadeModel};
use std::io::BufWriter;

fn main(){
    netflix_prize::suppress_progress();
    let phase=std::env::args().nth(1).expect("phase x or qual");
    let (train,query)=match phase.as_str(){"x"=>("trainx","probex"),"qual"=>("fulltrain","qual"),_=>panic!("unknown phase")};
    std::fs::create_dir_all("preds_new").unwrap();
    *LOG_FILE.lock().unwrap()=Some(BufWriter::new(std::fs::File::create(format!("preds_new/cfnade-96.phase-{query}.out")).unwrap()));
    let cfg=CfNadeConfig{n_hidden:320,rank:96,n_epochs:95,..CfNadeConfig::default()};
    netflix_prize::teeln!("[cfnade-96 independent phase] {:?}",cfg);
    fit::<CfNadeModel>(cfg,"rtg",train,query,"cfnade-96",false,false,true,"preds_new",false,false);
    LOG_FILE.lock().unwrap().take();
}
