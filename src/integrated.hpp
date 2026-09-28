// Integrated model (Koren 2008) on top of the joint temporal SVD++: the joint
// prediction plus learned explicit/implicit item-item weights over the G1 graph,
// all fitted by the same SGD error. Neighbor residuals r_uj - b_uj use the fixed
// static baseline of static_residuals, so the neighborhood term stays linear.
// The training loop mirrors command_joint, which is left unchanged so existing
// joint runs stay bit-for-bit reproducible. Neighbor weights are saved next to
// the joint checkpoint as MODEL.nbr.

struct IntegratedNeighbors {
  const NeighborGraph &g;
  const std::vector<float> &residual;
  std::vector<float> w, c;
  std::vector<uint32_t> rated, known;
  std::vector<float> known_residual;
  std::vector<uint32_t> explicit_edges, implicit_edges;
  float explicit_norm = 0, implicit_norm = 0;

  IntegratedNeighbors(const Dataset &d, const NeighborGraph &graph, const std::vector<float> &r)
      : g(graph), residual(r), w(graph.neighbor.size(), 0), c(graph.neighbor.size(), 0),
        rated(d.header.items, UINT32_MAX), known(d.header.items, UINT32_MAX),
        known_residual(d.header.items, 0) {}

  void reset() {
    std::fill(rated.begin(), rated.end(), UINT32_MAX);
    std::fill(known.begin(), known.end(), UINT32_MAX);
  }
  void fill_user(const Dataset &d, uint32_t u) {
    for (uint64_t p = d.implicit_offsets[u]; p < d.implicit_offsets[u+1]; ++p) known[d.implicit_items[p]-1] = u;
    for (uint64_t p = d.train_offsets[u]; p < d.train_offsets[u+1]; ++p) {
      const uint32_t i = d.train_items[p]-1; rated[i] = u; known_residual[i] = residual[p];
    }
  }
  float correction(uint32_t u, uint32_t i) {
    explicit_edges.clear(); implicit_edges.clear();
    float a = 0, b = 0;
    for (uint32_t j = i*g.k; j < (i+1)*g.k; ++j) {
      if (g.similarity[j] == 0) continue;
      const uint32_t other = g.neighbor[j];
      if (rated[other] == u) { explicit_edges.push_back(j); a += w[j]*known_residual[other]; }
      if (known[other] == u) { implicit_edges.push_back(j); b += c[j]; }
    }
    explicit_norm = 1/std::sqrt(float(std::max<size_t>(1, explicit_edges.size())));
    implicit_norm = 1/std::sqrt(float(std::max<size_t>(1, implicit_edges.size())));
    return a*explicit_norm + b*implicit_norm;
  }
  void update(float err, float lr, float reg) {
    for (uint32_t j : explicit_edges)
      w[j] += lr*(err*explicit_norm*known_residual[g.neighbor[j]]-reg*w[j]);
    for (uint32_t j : implicit_edges) c[j] += lr*(err*implicit_norm-reg*c[j]);
  }
};

void save_integrated_neighbors(const fs::path &path, const IntegratedNeighbors &n, uint64_t graph_hash) {
  std::ofstream out(path.string()+".tmp", std::ios::binary);
  write_pod(out, uint64_t(0x4E46494E54474E31ULL)); write_pod(out, graph_hash);
  write_vector(out, n.w); write_vector(out, n.c);
  out.close(); if (!out) fail("integrated neighbor write failed");
  fs::rename(path.string()+".tmp", path);
}

void load_integrated_neighbors(const fs::path &path, IntegratedNeighbors &n, uint64_t graph_hash) {
  std::ifstream in(path, std::ios::binary);
  uint64_t magic = 0, hash = 0;
  read_pod(in, magic); read_pod(in, hash);
  if (magic != 0x4E46494E54474E31ULL || hash != graph_hash) fail("integrated neighbor file mismatch");
  read_vector(in, n.w, n.w.size()); read_vector(in, n.c, n.c.size());
  if (in.peek() != std::char_traits<char>::eof()) fail("trailing integrated neighbor data");
}

double eval_integrated(const Dataset &d, const JointFeatures &f, const JointModel &m,
                       IntegratedNeighbors &n, const std::vector<uint64_t> &po,
                       const std::vector<uint32_t> &probe_order, std::vector<float> &pred) {
  eval_joint(d, f, m, pred);
  n.reset();
  double sse = 0;
  for (uint32_t u = 0; u < d.header.users; ++u) {
    if (po[u] == po[u+1]) continue;
    n.fill_user(d, u);
    for (uint64_t q = po[u]; q < po[u+1]; ++q) {
      const uint32_t p = probe_order[q];
      pred[p] += n.correction(u, d.probe[p].item-1);
      const double err = d.probe[p].rating-clipped(pred[p]); sse += err*err;
    }
  }
  return std::sqrt(sse/d.probe.size());
}

void command_integrated(const std::vector<std::string> &args, bool eval_only) {
  const fs::path model_path = value_after(args, "--model");
  const fs::path nbr_path = model_path.string()+".nbr";
  const Dataset d = load_dataset(value_after(args, "--data"));
  const auto t = build_temporal_index(d);
  const auto f = joint_features(d, t);
  const uint64_t fingerprint = joint_fingerprint(d);
  const auto g = load_graph(value_after(args, "--graph"), d.header.items, fingerprint);
  const uint64_t graph_hash = graph_fingerprint(g);
  std::vector<float> residual; static_residuals(d, residual);
  IntegratedNeighbors n(d, g, residual);
  std::vector<uint64_t> po(d.header.users+1, 0);
  for (const auto &r : d.probe) ++po[r.user+1];
  std::partial_sum(po.begin(), po.end(), po.begin());
  auto cursor = po; std::vector<uint32_t> probe_order(d.probe.size());
  for (size_t p = 0; p < d.probe.size(); ++p) probe_order[cursor[d.probe[p].user]++] = uint32_t(p);
  std::vector<float> predictions;
  const std::string output = optional_value_after(args, "--predictions");
  auto export_predictions = [&]() {
    if (!output.empty()) { std::ofstream out(output, std::ios::binary); write_vector(out, predictions); }
  };
  if (eval_only) {
    const auto m = load_joint(model_path, d, t, fingerprint);
    load_integrated_neighbors(nbr_path, n, graph_hash);
    const double score = eval_integrated(d, f, m, n, po, probe_order, predictions);
    export_predictions();
    std::cout << std::setprecision(10) << "probe_rmse=" << score << " epoch=" << m.epoch << '\n';
    return;
  }
  const uint32_t rank = uint_after(args, "--factors", 64);
  const uint32_t epochs = uint_after(args, "--iterations", 30);
  const uint32_t patience = uint_after(args, "--patience", 4);
  const uint32_t seed = uint_after(args, "--seed", 2026);
  const float lr0 = float(double_after(args, "--learning-rate", .007));
  const float reg = float(double_after(args, "--regularization", .015));
  const float decay = float(double_after(args, "--decay", .9));
  const float base_rate = float(double_after(args, "--base-rate", 1));
  const float drift_rate = float(double_after(args, "--drift-rate", .00001));
  const float nlr0 = float(double_after(args, "--neighbor-rate", .005));
  const float nreg = float(double_after(args, "--neighbor-regularization", .002));
  if (!rank || rank > 2048 || !epochs || lr0 <= 0 || reg < 0 || lr0*reg >= 1
      || decay <= 0 || decay > 1 || base_rate <= 0 || drift_rate < 0 || nlr0 <= 0 || nreg < 0)
    fail("invalid integrated hyperparameters");
  const std::string init = optional_value_after(args, "--init-model");
  JointModel m = init.empty() ? new_joint(d, t, rank, seed) : load_joint(init, d, t, fingerprint);
  if (m.rank != rank || m.day_rank) fail("integrated requires matching rank and no day factors");
  std::mt19937 rng(seed);
  std::vector<uint32_t> users(d.header.users); std::iota(users.begin(), users.end(), 0);
  std::vector<uint64_t> order;
  std::vector<float> h(rank), h0(rank);
  double best = std::numeric_limits<double>::infinity();
  uint32_t stale = 0;
  std::cout << "integrated factors=" << rank << " seed=" << seed << " lr=" << lr0 << " regularization=" << reg
            << " decay=" << decay << " neighbor_rate=" << nlr0 << " neighbor_reg=" << nreg
            << " k=" << g.k << " init_model=" << init << std::endl;
  for (uint32_t epoch = 1; epoch <= epochs; ++epoch) {
    const auto start = std::chrono::steady_clock::now();
    const float lr = lr0*std::pow(decay, float(epoch-1)), nlr = nlr0*std::pow(decay, float(epoch-1));
    const float c = 1-lr*reg;
    std::shuffle(users.begin(), users.end(), rng);
    n.reset();
    double sse = 0;
    for (uint32_t u : users) {
      const uint64_t ib = d.implicit_offsets[u], ie = d.implicit_offsets[u+1];
      const uint64_t begin = d.train_offsets[u], end = d.train_offsets[u+1];
      if (begin == end) continue;
      n.fill_user(d, u);
      const float norm = 1/std::sqrt(float(ie-ib));
      std::fill(h.begin(), h.end(), 0);
      for (uint64_t j = ib; j < ie; ++j) {
        const float *y = &m.factors.implicit_factors[size_t(d.implicit_items[j]-1)*rank];
        for (uint32_t k = 0; k < rank; ++k) h[k] += y[k];
      }
      for (float &x : h) x *= norm;
      h0 = h;
      order.resize(end-begin); std::iota(order.begin(), order.end(), begin);
      std::shuffle(order.begin(), order.end(), rng);
      float *p = &m.factors.user_factors[size_t(u)*rank];
      float *a = &m.drift[size_t(u)*rank];
      auto &b = m.base;
      for (uint64_t pos : order) {
        const uint32_t i = d.train_items[pos]-1, s = f.train_session[pos];
        const auto day = d.train_days[pos]; const float dev = f.dev[s];
        float *q = &m.factors.item_factors[size_t(i)*rank];
        const uint32_t bin = i*kTimeBins+time_bin(d, day), freq = i*kFrequencyBins+f.freq[s];
        const float item = b.item_bias[i]+b.item_bin_bias[bin], scale = b.user_scale[u]+b.day_scale[s];
        float pred = float(d.header.mean)+b.user_bias[u]+b.user_alpha[u]*dev+b.day_bias[s]
                     +item*scale+b.item_frequency_bias[freq];
        for (uint32_t k = 0; k < rank; ++k) pred += q[k]*(p[k]+a[k]*dev+h[k]);
        pred += n.correction(u, i);
        const float err = float(d.train_ratings[pos])-pred;
        sse += double(err)*err;
        n.update(err, nlr, nreg);
        b.user_bias[u] += base_rate*.00267F*(err-.0255F*b.user_bias[u]);
        b.user_alpha[u] += base_rate*.00000311F*(err*dev-3.95F*b.user_alpha[u]);
        b.day_bias[s] += base_rate*.00257F*(err-.00231F*b.day_bias[s]);
        b.item_bias[i] += base_rate*.000488F*(err*scale-.0255F*b.item_bias[i]);
        b.item_bin_bias[bin] += base_rate*.000115F*(err*scale-.0929F*b.item_bin_bias[bin]);
        b.user_scale[u] += base_rate*.00564F*(err*item-.0476F*(b.user_scale[u]-1));
        b.day_scale[s] += base_rate*.00103F*(err*item-.019F*b.day_scale[s]);
        b.item_frequency_bias[freq] += base_rate*.00236F*(err-1.1e-8F*b.item_frequency_bias[freq]);
        for (uint32_t k = 0; k < rank; ++k) {
          const float oldq = q[k], oldp = p[k];
          q[k] += lr*(err*(oldp+a[k]*dev+h[k])-reg*oldq);
          p[k] += lr*(err*oldq-reg*oldp);
          a[k] += drift_rate*(err*oldq*dev-50*a[k]);
          h[k] = c*h[k]+lr*err*oldq;
        }
      }
      const float shrink = std::pow(c, float(end-begin));
      for (uint32_t k = 0; k < rank; ++k) h[k] = norm*(h[k]-shrink*h0[k]);
      for (uint64_t j = ib; j < ie; ++j) {
        float *y = &m.factors.implicit_factors[size_t(d.implicit_items[j]-1)*rank];
        for (uint32_t k = 0; k < rank; ++k) y[k] = shrink*y[k]+h[k];
      }
    }
    m.epoch = epoch;
    const double score = eval_integrated(d, f, m, n, po, probe_order, predictions);
    if (!std::isfinite(sse) || !std::isfinite(score)) fail("non-finite integrated run");
    if (score < best) {
      best = score; stale = 0;
      save_joint(model_path, m, fingerprint); save_integrated_neighbors(nbr_path, n, graph_hash);
      export_predictions();
    } else {
      ++stale;
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    std::cout << std::setprecision(10) << "epoch=" << epoch << " train_rmse=" << std::sqrt(sse/d.header.train_count)
              << " probe_rmse=" << score << " best=" << best << " seconds=" << seconds << std::endl;
    if (patience && stale >= patience) {
      std::cout << "early_stop patience=" << patience << std::endl;
      break;
    }
  }
}
