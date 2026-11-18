// Appended to the parity driver's fixture/config definitions. No training change.
struct Rows<'a>(&'a netflix_prize::tsvdx4p::Tsvdx4pModel);
impl<'a> Regressor for Rows<'a> {
    type Config=();
    fn new(_: &Dataset,_: &MaskedDataset,_: ())->Self {unreachable!()}
    fn fit_epoch(&mut self,_: &Dataset,_: &MaskedDataset,_: usize){unreachable!()}
    fn predict(&self,u:usize,i:usize,day:i32)->f32{self.0.predict(u,i,day)}
}

fn pad_users(mut ds:Dataset)->Dataset{
    ds.n_users=20_000;
    let mut counts=vec![0;ds.n_users];
    for &u in &ds.user_idxs{counts[u as usize]+=1;}
    ds.user_cnts=Array1::from_vec(counts);ds
}

fn main(){
    netflix_prize::suppress_progress();
    let tr=pad_users(dataset(false));let pr=pad_users(dataset(true));
    let mut model=netflix_prize::tsvdx4p::Tsvdx4pModel::new(&tr,&MaskedDataset::from(&pr),config!(tsvdx4p,1000,16,false));
    model.fit_epoch(&tr,&MaskedDataset::from(&pr),1);
    let indices:Vec<usize>=(0..200_000).map(|i|i%pr.n_ratings).collect();
    let users=pr.user_idxs.select(ndarray::Axis(0),&indices);
    let items=pr.item_idxs.select(ndarray::Axis(0),&indices);
    let mut user_counts=vec![0;pr.n_users];let mut item_counts=vec![0;pr.n_items];
    for &u in &users{user_counts[u as usize]+=1;}
    for &i in &items{item_counts[i as usize]+=1;}
    let query=Dataset{user_idxs:users,user_cnts:Array1::from_vec(user_counts),
        item_idxs:items,item_cnts:Array1::from_vec(item_counts),item_years:pr.item_years.clone(),
        residuals:Array1::zeros(indices.len()),raw_ratings:Array1::zeros(indices.len()),
        dates:pr.dates.select(ndarray::Axis(0),&indices),is_test:Array1::zeros(indices.len()),
        name:"synthetic-timing-query".into(),n_users:pr.n_users,n_items:pr.n_items,
        n_ratings:indices.len(),transposed:false};
    let rows=Rows(&model);let mut reference=None;
    for round in 0..3{
        let order=if round%2==0{[false,true]}else{[true,false]};
        for grouped in order{
            let start=std::time::Instant::now();
            let predictions=if grouped{netflix_prize::gen_preds_parallel(&model,&query)}else{netflix_prize::gen_preds_parallel(&rows,&query)};
            let seconds=start.elapsed().as_secs_f64();
            let bits:Vec<u32>=predictions.iter().map(|v|v.to_bits()).collect();
            if let Some(expected)=&reference{assert_eq!(&bits,expected,"timed prediction bits differ");}
            else{reference=Some(bits.clone());}
            let bytes:Vec<u8>=bits.iter().flat_map(|v|v.to_le_bytes()).collect();
            println!("BENCH {} {} {:.9} {:x}",if grouped{"grouped"}else{"row"},round,seconds,md5::compute(bytes));
        }
    }
    println!("PASS: 200000 synthetic query rows,1000 factors,20000 padded users;80MB user allocations exercise mapped eviction; alternating traversal order,all output bits equal");
}
