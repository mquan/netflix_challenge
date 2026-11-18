// Included inside the optimized CFNADE module. Checkpoints own a dedicated,
// identity-bound directory and an OS writer lock for the model's lifetime.
fn checkpoint_md5(path:&std::path::Path)->String{
    let bytes=std::fs::read(path).unwrap();format!("{:x}",md5::compute(bytes))
}
fn checkpoint_json(path:&std::path::Path)->serde_json::Value{
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}
fn checkpoint_sync(path:&std::path::Path){
    std::fs::OpenOptions::new().read(true).write(true).open(path).unwrap().sync_all().unwrap();
}
impl CfNadeModel {
    /// The averaging driver restores the same SGD trajectory before each epoch.
    /// Export its raw state without changing live parameters, optimizer, or RNG.
    pub fn predict_raw_sgd(&mut self,ds:&Dataset)->Array1<f32>{
        let averaged=self.swapped_in;
        if averaged{self.swap_with_swa();}
        let output=crate::gen_preds_parallel(self,ds);
        if averaged{self.swap_with_swa();}
        output
    }

    fn checkpoint_open(&mut self,tr:&Dataset,pr:&MaskedDataset){
        use std::os::fd::AsRawFd;
        let Some(base)=std::env::var_os("CF_CHECKPOINT_DIR") else{return;};
        let source=std::env::var("CF_CHECKPOINT_SOURCE").expect("checkpoint source identity required");
        let data=std::env::var("CF_CHECKPOINT_DATA").expect("checkpoint data identity required");
        for id in [&source,&data]{assert!(id.len()==64 && id.bytes().all(|b|b.is_ascii_hexdigit()));}
        assert!(tr.name.bytes().all(|b|b.is_ascii_alphanumeric() || b==b'_' || b==b'-'));
        let root=std::path::PathBuf::from(base).join(&tr.name);
        std::fs::create_dir_all(&root).unwrap();
        assert!(!std::fs::symlink_metadata(&root).unwrap().file_type().is_symlink());
        let lock=std::fs::OpenOptions::new().create(true).read(true).write(true).open(root.join("writer.lock")).unwrap();
        assert_eq!(unsafe{libc::flock(lock.as_raw_fd(),libc::LOCK_EX|libc::LOCK_NB)},0,"another checkpoint writer owns this phase");
        self.checkpoint_lock=Some(lock);
        let identity=serde_json::json!({"protocol":1,"source":source,"data":data,
            "train":tr.name,"query":pr.name,"train_rows":tr.n_ratings,"query_rows":pr.n_ratings,
            "users":tr.n_users,"items":tr.n_items,"config":format!("{:?}",self.cfg)});
        let owner=root.join("owner.json");
        if owner.exists(){assert_eq!(checkpoint_json(&owner),identity,"checkpoint identity differs");}
        else{
            assert!(std::fs::read_dir(&root).unwrap().all(|e|e.unwrap().file_name()=="writer.lock"),"refuse unowned checkpoint contents");
            std::fs::write(&owner,serde_json::to_vec_pretty(&identity).unwrap()).unwrap();checkpoint_sync(&owner);
        }
        self.checkpoint_directory=Some(root.clone());self.checkpoint_identity=identity.clone();
        let latest=root.join("latest.json");if !latest.exists(){return;}
        let pointer=checkpoint_json(&latest);let epoch=pointer["epoch"].as_u64().unwrap() as usize;
        assert!(epoch>0 && epoch<=self.cfg.n_epochs);
        let directory=root.join(format!("epoch-{epoch:05}"));
        assert!(!std::fs::symlink_metadata(&directory).unwrap().file_type().is_symlink());
        let manifest=directory.join("state.json");assert_eq!(checkpoint_md5(&manifest),pointer["manifest_md5"].as_str().unwrap());
        let state=checkpoint_json(&manifest);assert_eq!(state["identity"],identity);assert_eq!(state["epoch"].as_u64().unwrap() as usize,epoch);
        self.checkpoint_read_arrays(&directory,state["files"].as_object().unwrap());
        self.timestep=state["timestep"].as_u64().unwrap() as usize;
        self.swa_count=state["swa_count"].as_u64().unwrap() as usize;assert!(self.swa_count<=epoch);
        self.swapped_in=state["swapped_in"].as_bool().unwrap();
        let seed:[u8;32]=serde_json::from_value(state["rng_seed"].clone()).unwrap();
        self.rng=StdRng::from_seed(seed);self.rng.set_stream(state["rng_stream"].as_u64().unwrap());
        self.rng.set_word_pos(state["rng_word_pos"].as_str().unwrap().parse().unwrap());
        self.checkpoint_loaded_epoch=epoch;self.clear_cache();
        crate::teeln!("Exact CFNADE checkpoint restored: {} epoch {}",tr.name,epoch);
    }

    fn checkpoint_save(&mut self,epoch:usize){
        let Some(root)=self.checkpoint_directory.as_ref() else{return;};
        assert!(epoch>0 && epoch<=self.cfg.n_epochs);
        let unique=std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let temp=root.join(format!("writing-epoch-{epoch:05}-{}-{unique}",std::process::id()));
        std::fs::create_dir(&temp).unwrap();
        let files=self.checkpoint_write_arrays(&temp);
        let state=serde_json::json!({"identity":self.checkpoint_identity,"epoch":epoch,"files":files,
            "timestep":self.timestep,"swa_count":self.swa_count,"swapped_in":self.swapped_in,
            "rng_seed":self.rng.get_seed(),"rng_stream":self.rng.get_stream(),"rng_word_pos":self.rng.get_word_pos().to_string()});
        let manifest=temp.join("state.json");std::fs::write(&manifest,serde_json::to_vec_pretty(&state).unwrap()).unwrap();checkpoint_sync(&manifest);
        let digest=checkpoint_md5(&manifest);let directory=root.join(format!("epoch-{epoch:05}"));
        if directory.exists(){
            assert!(!std::fs::symlink_metadata(&directory).unwrap().file_type().is_symlink());
            assert_eq!(checkpoint_json(&directory.join("state.json"))["identity"],self.checkpoint_identity);
            std::fs::rename(&directory,root.join(format!("preserved-epoch-{epoch:05}-{unique}"))).unwrap();
        }
        std::fs::rename(&temp,&directory).unwrap();
        let pointer=root.join(format!("latest-{}-{unique}.tmp",std::process::id()));
        std::fs::write(&pointer,serde_json::to_vec_pretty(&serde_json::json!({"epoch":epoch,"manifest_md5":digest})).unwrap()).unwrap();checkpoint_sync(&pointer);
        std::fs::rename(&pointer,root.join("latest.json")).unwrap();
        let _=std::fs::File::open(root).and_then(|f|f.sync_all());
        // Retain the current and preceding complete epochs. Other directories,
        // including interrupted writes, are preserved for inspection.
        for entry in std::fs::read_dir(root).unwrap(){
            let entry=entry.unwrap();let name=entry.file_name();let Some(name)=name.to_str() else{continue;};
            let Some(number)=name.strip_prefix("epoch-") else{continue;};
            if number.len()!=5 || !number.bytes().all(|b|b.is_ascii_digit()){continue;}
            let old=number.parse::<usize>().unwrap();if old+1>=epoch{continue;}
            let metadata=std::fs::symlink_metadata(entry.path()).unwrap();
            if metadata.file_type().is_symlink() || !metadata.is_dir(){continue;}
            let path=entry.path().join("state.json");
            if path.is_file() && checkpoint_json(&path)["identity"]==self.checkpoint_identity{
                std::fs::remove_dir_all(entry.path()).unwrap();
            }
        }
        self.checkpoint_loaded_epoch=epoch;
        if std::env::var("CF_EXIT_AFTER_CHECKPOINT").ok().and_then(|v|v.parse::<usize>().ok())==Some(epoch){
            crate::teeln!("Deliberate restart test after durable checkpoint {}",epoch);
            std::process::exit(73);
        }
    }
}
