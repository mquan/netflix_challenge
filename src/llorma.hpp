// Included inside netflix.cpp's anonymous namespace after joint.hpp.
//
// Local low-rank ensemble ("L1", FAMILY_AUDIT.md), after LLORMA (Lee, Kim,
// Lebanon, Singer, ICML 2013): instead of one global factorization, fit many
// low-rank models, each weighted around an anchor (user, movie) pair, and
// combine them with the same kernel weights at prediction time.
//
//   distance  d_U(u, u_t) = arccos(cos(p_u, p_ut)) on a frozen joint model's
//             effective user vectors; d_I likewise on its item factors
//   kernel    Epanechnikov K_h(d) = max(0, 1 - (d/h)^2)
//   weight    w_t(u, i) = K_h(d_U(u, u_t)) * K_h(d_I(i, i_t))
//   local fit weighted SGD on residuals r - b(u, i, t) of the joint model's
//             *baseline* terms only (its biases and time effects, not its
//             factors), so the local models own the user-item interaction
//   predict   b(u, i, t) + sum_t w_t f_t(u, i) / sum_t w_t   (b if no weight)
//
// Anchors are training rows drawn uniformly with --seed. Each anchor's model
// is trained independently with its own RNG (seed + anchor index); anchor t
// runs on thread t % threads and threads are reduced in order, so a run is
// reproducible for a fixed --threads.
namespace llorma {

inline float epanechnikov(float d, float h) { const float x = d/h; return x < 1 ? 1-x*x : 0; }

void normalize_rows(std::vector<float> &v, uint32_t rank) {
  for (size_t r = 0; r*rank < v.size(); ++r) {
    float *x = &v[r*rank]; double s = 0;
    for (uint32_t k = 0; k < rank; ++k) s += double(x[k])*x[k];
    const float inv = s > 0 ? float(1/std::sqrt(s)) : 0;
    for (uint32_t k = 0; k < rank; ++k) x[k] *= inv;
  }
}

// Kernel weights of every row of `unit` (unit-normalised vectors) to row a.
// With coverage > 0 the bandwidth is set per anchor so that the kernel's
// support holds that fraction of rows (the coverage-quantile distance, nudged up).
float kernel_row(const std::vector<float> &unit, uint32_t rank, size_t a, float h, double coverage,
                 std::vector<float> &out) {
  const size_t n = unit.size()/rank;
  out.resize(n);
  const float *pa = &unit[a*rank];
  for (size_t r = 0; r < n; ++r) {
    const float *x = &unit[r*rank]; float c = 0;
    for (uint32_t k = 0; k < rank; ++k) c += x[k]*pa[k];
    out[r] = std::acos(std::min(1.F, std::max(-1.F, c)));
  }
  if (coverage > 0) {
    std::vector<float> sorted(out);
    const size_t q = std::min(n-1, size_t(coverage*double(n)));
    std::nth_element(sorted.begin(), sorted.begin()+q, sorted.end());
    h = std::max(sorted[q]*1.0001F, 1e-6F);
  }
  for (auto &v : out) v = epanechnikov(v, h);
  return h;
}

// Project rows of v (n x rank) onto their top-k principal directions (centred),
// by subspace iteration on the rank x rank covariance. Deterministic.
std::vector<float> pca_project(const std::vector<float> &v, uint32_t rank, uint32_t k) {
  const size_t n = v.size()/rank;
  std::vector<double> mean(rank, 0), cov(size_t(rank)*rank, 0);
  for (size_t r = 0; r < n; ++r) for (uint32_t a = 0; a < rank; ++a) mean[a] += v[r*rank+a];
  for (auto &m : mean) m /= double(std::max<size_t>(n, 1));
  std::vector<double> x(rank);
  for (size_t r = 0; r < n; ++r) {
    for (uint32_t a = 0; a < rank; ++a) x[a] = v[r*rank+a]-mean[a];
    for (uint32_t a = 0; a < rank; ++a) { const double xa = x[a]; double *c = &cov[size_t(a)*rank];
      for (uint32_t b = a; b < rank; ++b) c[b] += xa*x[b]; }
  }
  for (uint32_t a = 0; a < rank; ++a) for (uint32_t b = 0; b < a; ++b) cov[size_t(a)*rank+b] = cov[size_t(b)*rank+a];
  std::vector<double> q(size_t(rank)*k), z(size_t(rank)*k);
  std::mt19937_64 rng(17); std::normal_distribution<double> nd(0, 1);
  for (auto &e : q) e = nd(rng);
  for (int it = 0; it < 200; ++it) {
    for (uint32_t a = 0; a < rank; ++a) for (uint32_t j = 0; j < k; ++j) {
      double s = 0; for (uint32_t b = 0; b < rank; ++b) s += cov[size_t(a)*rank+b]*q[size_t(b)*k+j];
      z[size_t(a)*k+j] = s; }
    for (uint32_t j = 0; j < k; ++j) {  // modified Gram-Schmidt on columns
      for (uint32_t i = 0; i < j; ++i) { double dot = 0;
        for (uint32_t a = 0; a < rank; ++a) dot += z[size_t(a)*k+j]*z[size_t(a)*k+i];
        for (uint32_t a = 0; a < rank; ++a) z[size_t(a)*k+j] -= dot*z[size_t(a)*k+i]; }
      double nn = 0; for (uint32_t a = 0; a < rank; ++a) nn += z[size_t(a)*k+j]*z[size_t(a)*k+j];
      nn = std::sqrt(nn) + 1e-300;
      for (uint32_t a = 0; a < rank; ++a) z[size_t(a)*k+j] /= nn;
    }
    q.swap(z);
  }
  std::vector<float> out(n*k);
  for (size_t r = 0; r < n; ++r) for (uint32_t j = 0; j < k; ++j) {
    double s = 0; for (uint32_t a = 0; a < rank; ++a) s += (v[r*rank+a]-mean[a])*q[size_t(a)*k+j];
    out[r*k+j] = float(s); }
  return out;
}

void command(const std::vector<std::string> &args) {
  const Dataset d = load_dataset(value_after(args, "--data"));
  if (d.probe.empty() || d.train_items.empty()) fail("empty train or probe");
  const auto t = build_temporal_index(d);
  const auto f = joint_features(d, t);
  const uint64_t fingerprint = bound_fingerprint(args, d);
  const JointModel g = load_joint(value_after(args, "--joint-model"), d, t, fingerprint);
  const std::string pred_path = value_after(args, "--predictions");
  const uint32_t anchors = uint_after(args, "--anchors", 100);
  const uint32_t rank = uint_after(args, "--rank", 20);
  const uint32_t epochs = uint_after(args, "--iterations", 15);
  const uint32_t threads = std::max(1U, uint_after(args, "--threads", 1));
  const uint32_t seed = uint_after(args, "--seed", 2026);
  const float h = float(double_after(args, "--bandwidth", .8));
  const float lr0 = float(double_after(args, "--learning-rate", .01));
  const float reg = float(double_after(args, "--regularization", .02));
  const float decay = float(double_after(args, "--decay", .95));
  const float init_std = float(double_after(args, "--init-std", 0.1/std::sqrt(double(std::max(1U, uint_after(args, "--rank", 20))))));
  if (!anchors || !rank || rank > 512 || !epochs || !(h > 0) || !(lr0 > 0) || reg < 0 || !(decay > 0) || decay > 1 || !(init_std > 0))
    fail("invalid llorma settings");
  const uint32_t users = d.header.users, items = d.header.items, gr = g.rank;
  // Frozen geometry from the joint model.
  // Geometry: the joint model's effective user vectors and item factors, or
  // low-dimensional coordinates supplied as float32 row-major files
  // (--user-geometry/--item-geometry, --geometry-dims columns), unit-normalised.
  std::vector<float> ue, ie;
  uint32_t gd = gr;
  const std::string ug = optional_value_after(args, "--user-geometry"), ig = optional_value_after(args, "--item-geometry");
  if (ug.empty() != ig.empty()) fail("pass both --user-geometry and --item-geometry, or neither");
  if (!ug.empty()) {
    gd = uint_after(args, "--geometry-dims", 0);
    if (!gd) fail("--geometry-dims required with geometry files");
    auto read = [&](const std::string &path, size_t rows, std::vector<float> &out) {
      std::ifstream in(path, std::ios::binary | std::ios::ate);
      if (!in || uint64_t(in.tellg()) != rows*gd*sizeof(float)) fail("geometry file size mismatch: "+path);
      in.seekg(0); read_vector(in, out, rows*gd);
    };
    read(ug, users, ue); read(ig, items, ie);
  } else {
    build_effective_users(d, g.factors, gr, ue);
    ie = g.factors.item_factors;
    const uint32_t pca = uint_after(args, "--geometry-pca", 0);
    if (pca) {
      if (pca > gr) fail("--geometry-pca exceeds the joint rank");
      ue = pca_project(ue, gr, pca); ie = pca_project(ie, gr, pca); gd = pca;
    }
  }
  normalize_rows(ue, gd); normalize_rows(ie, gd);
  const double coverage = double_after(args, "--coverage", 0);
  if (coverage < 0 || coverage >= 1) fail("--coverage must be in [0, 1)");
  // Baseline-only predictions for training and probe rows.
  std::vector<float> base_train(d.header.train_count), base_probe(d.probe.size());
  std::vector<uint32_t> row_user(d.header.train_count);
  for (uint32_t u = 0; u < users; ++u)
    for (uint64_t p = d.train_offsets[u]; p < d.train_offsets[u+1]; ++p) {
      row_user[p] = u;
      base_train[p] = joint_base(d, f, g, u, d.train_items[p]-1, d.train_days[p], f.train_session[p]);
    }
  for (size_t n = 0; n < d.probe.size(); ++n) {
    const auto &r = d.probe[n];
    base_probe[n] = joint_base(d, f, g, r.user, r.item-1, r.day, f.probe_session[n]);
  }
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<uint64_t> pick(0, d.header.train_count-1);
  std::vector<uint64_t> anchor_rows(anchors);
  for (auto &a : anchor_rows) a = pick(rng);
  std::cout << "llorma anchors=" << anchors << " rank=" << rank << " bandwidth=" << h << " epochs=" << epochs
            << " lr=" << lr0 << " reg=" << reg << " init_std=" << init_std << " decay=" << decay << " threads=" << threads
            << " joint_rank=" << gr << " geometry_dims=" << gd << " coverage=" << coverage
            << " fingerprint=" << fingerprint << std::endl;
  struct Acc { std::vector<double> num, den; };
  std::vector<Acc> acc(threads);
  for (auto &a : acc) { a.num.assign(d.probe.size(), 0); a.den.assign(d.probe.size(), 0); }
  std::mutex log_mutex;
  auto fit_anchor = [&](uint32_t a, Acc &out) {
    const auto start = std::chrono::steady_clock::now();
    const uint64_t row = anchor_rows[a];
    std::vector<float> ku, ki;
    const float hu = kernel_row(ue, gd, row_user[row], h, coverage, ku);
    const float hi = kernel_row(ie, gd, d.train_items[row]-1, h, coverage, ki);
    std::vector<uint64_t> rows;
    for (uint64_t p = 0; p < d.header.train_count; ++p)
      if (ku[row_user[p]] > 0 && ki[d.train_items[p]-1] > 0) rows.push_back(p);
    // Rescale this anchor's training weights to mean 1 over its rows: the
    // weighted optimum is unchanged (up to the regularisation scale) and the
    // SGD step no longer shrinks with the kernel product.
    double wmean = 0;
    for (uint64_t p : rows) wmean += double(ku[row_user[p]])*ki[d.train_items[p]-1];
    const float wscale = rows.empty() ? 1.F : float(rows.size()/wmean);
    std::mt19937_64 r(seed*1000003ULL+a);
    std::normal_distribution<float> init(0, init_std);
    std::vector<float> P(size_t(users)*rank), Q(size_t(items)*rank);
    for (auto &x : P) x = init(r);
    for (auto &x : Q) x = init(r);
    double sse = 0, wsum = 0;
    for (uint32_t e = 0; e < epochs; ++e) {
      const float lr = lr0*std::pow(decay, float(e));
      std::shuffle(rows.begin(), rows.end(), r);
      sse = 0; wsum = 0;
      for (uint64_t p : rows) {
        const uint32_t u = row_user[p], i = d.train_items[p]-1;
        const float w = ku[u]*ki[i]*wscale;
        float *pu = &P[size_t(u)*rank], *qi = &Q[size_t(i)*rank];
        float pred = 0;
        for (uint32_t k = 0; k < rank; ++k) pred += pu[k]*qi[k];
        const float err = float(d.train_ratings[p])-base_train[p]-pred;
        sse += double(w)*err*err; wsum += w;
        const float we = w*err;
        for (uint32_t k = 0; k < rank; ++k) {
          const float op = pu[k];
          pu[k] += lr*(we*qi[k]-reg*op);
          qi[k] += lr*(we*op-reg*qi[k]);
        }
      }
    }
    uint64_t covered = 0;
    for (size_t n = 0; n < d.probe.size(); ++n) {
      const auto &q = d.probe[n];
      const float w = ku[q.user]*ki[q.item-1];
      if (w <= 0) continue;
      const float *pu = &P[size_t(q.user)*rank], *qi = &Q[size_t(q.item-1)*rank];
      float pred = 0;
      for (uint32_t k = 0; k < rank; ++k) pred += pu[k]*qi[k];
      out.num[n] += double(w)*pred; out.den[n] += w; ++covered;
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    std::lock_guard<std::mutex> lock(log_mutex);
    std::cout << std::setprecision(6) << "anchor=" << a << " h_user=" << hu << " h_item=" << hi << " rows=" << rows.size()
              << " weighted_train_rmse=" << (wsum > 0 ? std::sqrt(sse/wsum) : 0)
              << " probe_covered=" << covered << " seconds=" << seconds << std::endl;
  };
  std::vector<std::thread> pool;
  for (uint32_t th = 0; th < threads; ++th)
    pool.emplace_back([&, th]() { for (uint32_t a = th; a < anchors; a += threads) fit_anchor(a, acc[th]); });
  for (auto &x : pool) x.join();
  std::vector<float> pred(d.probe.size()), support(d.probe.size());
  double sse = 0, base_sse = 0; uint64_t none = 0;
  for (size_t n = 0; n < d.probe.size(); ++n) {
    double num = 0, den = 0;
    for (const auto &a : acc) { num += a.num[n]; den += a.den[n]; }
    pred[n] = base_probe[n] + float(den > 0 ? num/den : 0);
    support[n] = float(den); none += den <= 0;
    if (!std::isfinite(pred[n])) fail("non-finite llorma prediction");
    const double e = d.probe[n].rating-clipped(pred[n]), eb = d.probe[n].rating-clipped(base_probe[n]);
    sse += e*e; base_sse += eb*eb;
  }
  { std::ofstream out(pred_path, std::ios::binary); write_vector(out, pred); if (!out) fail("cannot write predictions"); }
  const std::string support_path = optional_value_after(args, "--support");
  if (!support_path.empty()) { std::ofstream out(support_path, std::ios::binary); write_vector(out, support); }
  std::cout << std::setprecision(10) << "final probe_rmse=" << std::sqrt(sse/d.probe.size())
            << " baseline_probe_rmse=" << std::sqrt(base_sse/d.probe.size())
            << " uncovered_probe_rows=" << none << '\n';
}

}  // namespace llorma
