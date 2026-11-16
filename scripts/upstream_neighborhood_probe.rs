//! Diagnostic only: hold the residual model fixed and change support provenance.
//! No tracked upstream model code is changed; qualifying targets must be zeros.
use netflix_prize::{Dataset, MaskedDataset, Regressor, gen_preds_parallel};
use netflix_prize::knns::{KnnsConfig, KnnsModel, SuppSource};
use netflix_prize::nbstats::build_bin_nbstats;
use ndarray_npy::{write_npy, read_npy};
use ndarray::Array2;
use std::path::Path;

fn main() {
    std::fs::create_dir_all("investigation").unwrap();
    let support_path = "investigation/bin_supp.fulltrain-plus-qual.npy";
    if !Path::new(support_path).exists() {
        let full = Dataset::load("fulltrain", "rtg", "preds_new");
        let qual = Dataset::load("qual", "rtg", "preds_new");
        assert!(qual.raw_ratings.iter().all(|r| *r == 0), "qual targets must be zero sentinels");
        let matrix = build_bin_nbstats(&full, &MaskedDataset::from(&qual), "supp");
        write_npy(support_path, &matrix).unwrap();
    }
    // Verify the support file can be loaded before fitting either diagnostic.
    let matrix: Array2<f32> = read_npy(support_path).unwrap();
    assert_eq!(matrix.shape(), &[17770, 17770]);
    drop(matrix);
    let tr = Dataset::load("trainx", "1.0*tsvdx4-60", "preds_new");
    let pr = Dataset::load("probex", "1.0*tsvdx4-60", "preds_new");
    for (name, source) in [("same-recipe", SuppSource::Compute),
                            ("support-with-qual", SuppSource::Path(support_path))] {
        let cfg = KnnsConfig { k:25, shrinkage:100.0, scaling:1.0,
                               tau:0.025, supp_source:source };
        let model = KnnsModel::new(&tr, &MaskedDataset::from(&pr), cfg);
        let predictions = gen_preds_parallel(&model, &pr);
        write_npy(format!("investigation/{}.probex.npy", name), &predictions).unwrap();
    }
}
