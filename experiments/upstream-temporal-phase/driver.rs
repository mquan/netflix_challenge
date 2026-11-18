use netflix_prize::{OrdinalHeadConfig,tx::{TxModel,TxConfig}};
fn main(){ let job_name="tsvdx5-150o"; run_ord(150, 25, 120, 1.2, job_name); }
fn run_ord(n_feat: usize, n_epochs: usize, max_neighbors: usize, w_factor: f32, job_name: &str) {
    let cfg = TxConfig {
        n_feat, n_epochs,
        seed: 64, shuffle_users: true,
        n_time_bins: 20, beta: 0.3, n_freq_bins: 16,
        lr_u: 0.003, lr_ud: 0.0015, lr_u2: 1e-5,
        lr_ub: 0.006, lr_ubd: 0.006,
        lr_i: 0.0072, lr_ib: 0.0072,
        lr_y: 0.0012, lr_yb: 2e-6, lr_yd: 0.000267,
        lr_tu: 0.0, lr_ti: 0.0001, lr_ta: 1.5e-5,
        lr_ibf: 5e-5, lr_iqf: 5e-6, lr_cu: 0.002,
        reg_iqf: 0.007, reg_cu: 0.01,
        reg_u: 0.04, reg_u2: 0.4, reg_ud: 0.04,
        reg_i: 0.007, reg_y: 0.04, reg_yd: 0.04,
        sigma_iqf: 0.005, sigma_u: 0.0, sigma_i: 0.005,
        sigma_y: 0.005, sigma_yd: 0.005,
        reset_u_epoch: 1024,
        max_neighbors,
        lr_w: 0.0015, lr_c: 0.0015, reg_w: 0.002, reg_c: 0.002,
        lr_w_day: 0.0, lr_c_day: 0.0, reg_w_day: 0.0, reg_c_day: 0.0,
        w_bias: w_factor, w_factor, w_nbr: w_factor,
        sum_err_bug: false,
        lambda1: 25.0, lambda2: 10.0,
        ordinal_head: Some(OrdinalHeadConfig {
            th_init: [0.5, 1.25, 3.75, 5.5],
            th_gap: 0.001,
            lr_t: 0.0,
            reg_t: 0.0,
        }),
        save_ifeat: false,
        save_ufeat: false,
        low_memory: false,
        full_su: true,
        lazy_day: false,
        ud_dim: None,
    };
    if std::env::args().any(|v| v == "--describe") { println!("{:?}",cfg); return; }
    netflix_prize::fit::<TxModel>(cfg,"rtg","fulltrain","qual",job_name,
        true,true,true,"preds_independent_qual",false,false);
}
