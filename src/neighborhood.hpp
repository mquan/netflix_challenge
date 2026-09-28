// Independently learned item-item weights, with no latent-factor restriction.
// Approximate graph discovery uses capped histories; weight training uses every
// non-probe training rating. Neither stage reads probe rating values.
struct NeighborGraph {
  uint32_t items = 0, k = 0;
  std::vector<uint16_t> neighbor;
  std::vector<float> similarity;
};

uint64_t graph_fingerprint(const NeighborGraph &g) {
  uint64_t h = 14695981039346656037ULL;
  hash_vector(h, g.neighbor); hash_vector(h, g.similarity);
  return h;
}

void static_residuals(const Dataset &d, std::vector<float> &residual) {
  std::vector<double> sums(d.header.items, 0);
  std::vector<uint64_t> counts(d.header.items, 0);
  for (size_t p = 0; p < d.train_items.size(); ++p) {
    uint32_t i = d.train_items[p]-1;
    sums[i] += d.train_ratings[p]-d.header.mean; ++counts[i];
  }
  std::vector<float> bias(d.header.items);
  for (uint32_t i = 0; i < d.header.items; ++i) bias[i] = float(sums[i]/(25+counts[i]));
  residual.resize(d.header.train_count);
  for (uint32_t u = 0; u < d.header.users; ++u) {
    auto begin = d.train_offsets[u], end = d.train_offsets[u+1];
    double sum = 0;
    for (uint64_t p = begin; p < end; ++p)
      sum += d.train_ratings[p]-d.header.mean-bias[d.train_items[p]-1];
    float bu = float(sum/(10+end-begin));
    for (uint64_t p = begin; p < end; ++p)
      residual[p] = float(d.train_ratings[p]-d.header.mean-bias[d.train_items[p]-1]-bu);
  }
}

NeighborGraph load_graph(const fs::path &path, uint32_t items, uint64_t fingerprint) {
  std::ifstream in(path, std::ios::binary);
  uint64_t magic = 0, hash = 0; NeighborGraph g;
  read_pod(in, magic); read_pod(in, hash); read_pod(in, g.items); read_pod(in, g.k);
  if (magic != 0x4E46475241504831ULL || hash != fingerprint || items != g.items || !g.k || g.k >= items)
    fail("neighborhood graph mismatch");
  read_vector(in, g.neighbor, size_t(items)*g.k);
  read_vector(in, g.similarity, size_t(items)*g.k);
  for (size_t p = 0; p < g.neighbor.size(); ++p) {
    if (g.neighbor[p] >= items || g.neighbor[p] == p/g.k) fail("invalid neighbor index");
    if (!std::isfinite(g.similarity[p]) || g.similarity[p] < 0) fail("invalid graph similarity");
  }
  return g;
}

void command_graph(const std::vector<std::string> &args) {
  const auto d = load_dataset(value_after(args, "--data"));
  const auto output = value_after(args, "--graph");
  uint32_t k = uint_after(args, "--neighbors", 64), cap = uint_after(args, "--history-cap", 256);
  if (!k || k >= d.header.items || cap < 2) fail("invalid graph size");
  const uint32_t n = d.header.items;
  std::vector<float> residual; static_residuals(d, residual);
  std::vector<float> gram(size_t(n)*n, 0);
  std::vector<uint32_t> count(size_t(n)*n, 0);
  std::vector<double> squares(n, 0);
  std::mt19937 rng(2026);
  std::vector<uint64_t> indices;
  uint64_t pairs = 0;
  for (uint32_t u = 0; u < d.header.users; ++u) {
    auto begin = d.train_offsets[u], end = d.train_offsets[u+1];
    indices.resize(end-begin); std::iota(indices.begin(), indices.end(), begin);
    if (indices.size() > cap) {
      std::shuffle(indices.begin(), indices.end(), rng); indices.resize(cap);
    }
    std::sort(indices.begin(), indices.end(), [&](uint64_t a, uint64_t b) { return d.train_items[a] < d.train_items[b]; });
    for (size_t a = 0; a < indices.size(); ++a) {
      auto p = indices[a]; uint32_t i = d.train_items[p]-1;
      float x = residual[p]; squares[i] += double(x)*x;
      for (size_t b = a+1; b < indices.size(); ++b) {
        auto q = indices[b]; size_t slot = size_t(i)*n+d.train_items[q]-1;
        gram[slot] += x*residual[q]; ++count[slot];
      }
    }
    pairs += indices.size()*(indices.size()-1)/2;
    if ((u+1)%50000 == 0) std::cout << "graph_users=" << u+1 << " pairs=" << pairs << std::endl;
  }
  NeighborGraph g; g.items = n; g.k = k;
  g.neighbor.resize(size_t(n)*k); g.similarity.resize(size_t(n)*k);
  std::vector<std::pair<float,uint16_t>> candidates;
  for (uint32_t i = 0; i < n; ++i) {
    candidates.clear();
    for (uint32_t j = 0; j < n; ++j) {
      if (i == j) continue;
      size_t slot = size_t(std::min(i,j))*n+std::max(i,j);
      float score = 0;
      if (count[slot] >= 10 && squares[i] > 0 && squares[j] > 0)
        score = float(gram[slot]/std::sqrt(squares[i]*squares[j]) * count[slot]/(count[slot]+100.0));
      candidates.emplace_back(std::max(0.0F, score), uint16_t(j));
    }
    std::partial_sort(candidates.begin(), candidates.begin()+k, candidates.end(), std::greater<std::pair<float,uint16_t>>());
    for (uint32_t a = 0; a < k; ++a) {
      g.neighbor[size_t(i)*k+a] = candidates[a].second;
      g.similarity[size_t(i)*k+a] = candidates[a].first;
    }
  }
  std::ofstream out(output, std::ios::binary);
  write_pod(out, uint64_t(0x4E46475241504831ULL)); write_pod(out, joint_fingerprint(d));
  write_pod(out, n); write_pod(out, k); write_vector(out, g.neighbor); write_vector(out, g.similarity);
  std::cout << "graph_complete items=" << n << " neighbors=" << k << " pairs=" << pairs << std::endl;
}

void command_neighborhood(const std::vector<std::string> &args, bool eval_only) {
  const auto d = load_dataset(value_after(args, "--data"));
  const auto t = build_temporal_index(d);
  const auto f = joint_features(d,t);
  TimeBaselineHeader header{};
  JointModel base;
  base.base = load_time_baseline(value_after(args, "--time-model"), d, t, header);
  const auto hash = joint_fingerprint(d);
  const auto g = load_graph(value_after(args, "--graph"), d.header.items, hash);
  const auto model_path = value_after(args, "--model");
  const auto output = value_after(args, "--predictions");
  const uint32_t epochs = eval_only ? 1 : uint_after(args, "--iterations", 12);
  const float lr0 = float(double_after(args, "--learning-rate", .005));
  const float reg = float(double_after(args, "--regularization", .002));
  if (!epochs || lr0 <= 0 || reg < 0) fail("invalid neighborhood hyperparameters");
  std::vector<float> residual; static_residuals(d, residual);
  std::vector<float> baseline(d.header.train_count), probe_base(d.probe.size());
  for (uint32_t u = 0; u < d.header.users; ++u)
    for (uint64_t p = d.train_offsets[u]; p < d.train_offsets[u+1]; ++p)
      baseline[p] = joint_base(d,f,base,u,d.train_items[p]-1,d.train_days[p],f.train_session[p]);
  for (size_t p = 0; p < d.probe.size(); ++p) {
    const auto &r = d.probe[p]; probe_base[p] = joint_base(d,f,base,r.user,r.item-1,r.day,f.probe_session[p]);
  }
  std::vector<uint64_t> po(d.header.users+1, 0);
  for (const auto &r : d.probe) ++po[r.user+1];
  std::partial_sum(po.begin(), po.end(), po.begin());
  auto cursor = po; std::vector<uint32_t> probe_order(d.probe.size());
  for (size_t p = 0; p < d.probe.size(); ++p) probe_order[cursor[d.probe[p].user]++] = uint32_t(p);
  std::vector<float> w(g.neighbor.size(), 0), c(w.size(), 0), known_residual(d.header.items);
  uint32_t saved_epoch = 0;
  if (eval_only) {
    std::ifstream in(model_path, std::ios::binary);
    uint64_t magic = 0, saved_hash = 0, graph_hash = 0;
    uint32_t items = 0, k = 0;
    read_pod(in,magic); read_pod(in,saved_hash); read_pod(in,graph_hash);
    read_pod(in,saved_epoch); read_pod(in,items); read_pod(in,k);
    if (magic != 0x4E464E42484F4F31ULL || saved_hash != hash ||
        graph_hash != graph_fingerprint(g) || items != g.items || k != g.k)
      fail("neighborhood checkpoint mismatch");
    read_vector(in,w,g.neighbor.size()); read_vector(in,c,g.neighbor.size());
  }
  std::vector<uint32_t> rated(d.header.items, UINT32_MAX), known(d.header.items, UINT32_MAX);
  std::vector<uint32_t> explicit_edges, implicit_edges;
  auto fill_user = [&](uint32_t u) {
    for (uint64_t p = d.implicit_offsets[u]; p < d.implicit_offsets[u+1]; ++p) known[d.implicit_items[p]-1] = u;
    for (uint64_t p = d.train_offsets[u]; p < d.train_offsets[u+1]; ++p) {
      uint32_t i = d.train_items[p]-1; rated[i] = u; known_residual[i] = residual[p];
    }
  };
  auto correction = [&](uint32_t u, uint32_t i) {
    explicit_edges.clear(); implicit_edges.clear();
    float a = 0, b = 0;
    for (uint32_t j = i*g.k; j < (i+1)*g.k; ++j) {
      if (g.similarity[j] == 0) continue;
      uint32_t other = g.neighbor[j];
      if (rated[other] == u) { explicit_edges.push_back(j); a += w[j]*known_residual[other]; }
      if (known[other] == u) { implicit_edges.push_back(j); b += c[j]; }
    }
    return a/std::sqrt(float(std::max<size_t>(1, explicit_edges.size())))
         + b/std::sqrt(float(std::max<size_t>(1, implicit_edges.size())));
  };
  std::vector<uint32_t> users(d.header.users); std::iota(users.begin(), users.end(), 0);
  std::mt19937 rng(2026); double best = 1e100; uint32_t stale = 0;
  std::vector<float> pred(d.probe.size());
  for (uint32_t epoch = 1; epoch <= epochs; ++epoch) {
    const auto start = std::chrono::steady_clock::now();
    std::fill(rated.begin(), rated.end(), UINT32_MAX); std::fill(known.begin(), known.end(), UINT32_MAX);
    std::shuffle(users.begin(), users.end(), rng); double train_sse = 0;
    float lr = lr0*std::pow(.95F, float(epoch-1));
    if (!eval_only) for (uint32_t u : users) {
      fill_user(u);
      for (uint64_t p = d.train_offsets[u]; p < d.train_offsets[u+1]; ++p) {
        float value = baseline[p]+correction(u, d.train_items[p]-1);
        float err = float(d.train_ratings[p])-value; train_sse += double(err)*err;
        float a = err/std::sqrt(float(std::max<size_t>(1, explicit_edges.size())));
        float b = err/std::sqrt(float(std::max<size_t>(1, implicit_edges.size())));
        for (uint32_t j : explicit_edges) w[j] += lr*(a*known_residual[g.neighbor[j]]-reg*w[j]);
        for (uint32_t j : implicit_edges) c[j] += lr*(b-reg*c[j]);
      }
    }
    std::fill(rated.begin(), rated.end(), UINT32_MAX); std::fill(known.begin(), known.end(), UINT32_MAX);
    double sse = 0;
    for (uint32_t u = 0; u < d.header.users; ++u) {
      fill_user(u);
      for (uint64_t q = po[u]; q < po[u+1]; ++q) {
        uint32_t p = probe_order[q]; pred[p] = probe_base[p]+correction(u,d.probe[p].item-1);
        if (!std::isfinite(pred[p])) fail("non-finite neighborhood prediction");
        double err = d.probe[p].rating-clipped(pred[p]); sse += err*err;
      }
    }
    double score = std::sqrt(sse/d.probe.size());
    if (eval_only) {
      std::ofstream pout(output, std::ios::binary); write_vector(pout,pred);
      std::cout << std::setprecision(10) << "epoch=" << saved_epoch << " probe_rmse=" << score << std::endl;
      return;
    }
    if (score < best) {
      best = score; stale = 0;
      std::ofstream out(model_path+".tmp", std::ios::binary);
      write_pod(out, uint64_t(0x4E464E42484F4F31ULL)); write_pod(out, hash);
      write_pod(out, graph_fingerprint(g)); write_pod(out, epoch);
      write_pod(out, g.items); write_pod(out,g.k); write_vector(out,w); write_vector(out,c);
      out.close(); if (!out) fail("neighborhood model write failed");
      fs::rename(model_path+".tmp", model_path);
      std::ofstream pout(output, std::ios::binary); write_vector(pout,pred);
    } else ++stale;
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    std::cout << std::setprecision(10) << "epoch=" << epoch << " train_rmse=" << std::sqrt(train_sse/d.header.train_count)
              << " probe_rmse=" << score << " best=" << best << " seconds=" << seconds << std::endl;
    if (stale >= 4) break;
  }
}

// Training-free item kNN over the temporal-baseline residuals: a probe rating is
// the baseline plus a shrunk, similarity^power-weighted mean of the user's
// residuals on rated graph neighbors. Also writes the neighbor support
// (sum of weights) as a blending feature. Probe ratings are never read.
void command_knn(const std::vector<std::string> &args) {
  const auto d = load_dataset(value_after(args, "--data"));
  const auto t = build_temporal_index(d);
  const auto f = joint_features(d, t);
  // With --joint-model, neighbors correct the full factor model's residuals
  // instead of the temporal baseline's.
  const auto joint_path = optional_value_after(args, "--joint-model");
  // --base-train/--base-probe: float32 predictions of any model for every
  // training row (training order) and probe row, replacing the joint model.
  const auto base_train_path = optional_value_after(args, "--base-train");
  const auto base_probe_path = optional_value_after(args, "--base-probe");
  const bool external = !base_train_path.empty();
  if (external != !base_probe_path.empty()) fail("--base-train and --base-probe go together");
  std::vector<float> base_train, base_probe;
  if (external) {
    std::ifstream tin(base_train_path, std::ios::binary), pin(base_probe_path, std::ios::binary);
    if (!tin || !pin) fail("cannot read base prediction files");
    read_vector(tin, base_train, d.header.train_count); read_vector(pin, base_probe, d.probe.size());
    if (tin.peek() != std::char_traits<char>::eof() || pin.peek() != std::char_traits<char>::eof())
      fail("base prediction file size mismatch");
    for (float x : base_train) if (!std::isfinite(x)) fail("non-finite base prediction");
    for (float x : base_probe) if (!std::isfinite(x)) fail("non-finite base prediction");
  }
  JointModel base;
  if (external) {
    // no model needed
  } else if (joint_path.empty()) {
    TimeBaselineHeader header{};
    base.base = load_time_baseline(value_after(args, "--time-model"), d, t, header);
  } else {
    base = load_joint(joint_path, d, t, joint_fingerprint(d));
  }
  std::vector<float> effective;
  if (base.rank) build_effective_users(d, base.factors, base.rank, effective);
  auto predict = [&](uint32_t u, uint32_t i, uint16_t day, uint32_t s) {
    float x = joint_base(d, f, base, u, i, day, s);
    const float *p = base.rank ? &effective[size_t(u)*base.rank] : nullptr;
    for (uint32_t k = 0; k < base.rank; ++k)
      x += base.factors.item_factors[size_t(i)*base.rank+k]*(p[k]+base.drift[size_t(u)*base.rank+k]*f.dev[s]);
    for (uint32_t k = 0; k < base.day_rank; ++k)
      x += base.factors.item_factors[size_t(i)*base.rank+k]*base.day_factors[size_t(s)*base.day_rank+k];
    return x;
  };
  const auto g = load_graph(value_after(args, "--graph"), d.header.items, joint_fingerprint(d));
  const auto output = value_after(args, "--predictions");
  const auto support_path = optional_value_after(args, "--support");
  const uint32_t k = std::min(g.k, uint_after(args, "--neighbors", g.k));
  const float power = float(double_after(args, "--power", 2));
  const float shrink = float(double_after(args, "--shrink", .05));
  // Optional BellKor-style temporal weighting: a neighbor rated d days away
  // from the probe date is down-weighted by exp(-time_decay*d). 0 disables it.
  const float time_decay = float(double_after(args, "--time-decay", 0));
  if (time_decay < 0) fail("invalid --time-decay");
  std::vector<uint64_t> po(d.header.users+1, 0);
  for (const auto &r : d.probe) ++po[r.user+1];
  std::partial_sum(po.begin(), po.end(), po.begin());
  auto cursor = po; std::vector<uint32_t> probe_order(d.probe.size());
  for (size_t p = 0; p < d.probe.size(); ++p) probe_order[cursor[d.probe[p].user]++] = uint32_t(p);
  std::vector<uint32_t> rated(d.header.items, UINT32_MAX);
  std::vector<float> known_residual(d.header.items), known_day(d.header.items), pred(d.probe.size()), support(d.probe.size());
  double sse = 0;
  for (uint32_t u = 0; u < d.header.users; ++u) {
    if (po[u] == po[u+1]) continue;
    for (uint64_t p = d.train_offsets[u]; p < d.train_offsets[u+1]; ++p) {
      const uint32_t i = d.train_items[p]-1; rated[i] = u; known_day[i] = float(d.train_days[p]);
      known_residual[i] = d.train_ratings[p]-(external ? base_train[p] : predict(u, i, d.train_days[p], f.train_session[p]));
    }
    for (uint64_t q = po[u]; q < po[u+1]; ++q) {
      const uint32_t p = probe_order[q];
      const auto &r = d.probe[p];
      const uint32_t i = r.item-1;
      float num = 0, den = 0;
      for (uint32_t j = i*g.k; j < i*g.k+k; ++j) {
        const uint32_t other = g.neighbor[j];
        if (rated[other] != u || g.similarity[j] <= 0) continue;
        float s = std::pow(g.similarity[j], power);
        if (time_decay > 0) s *= std::exp(-time_decay*std::fabs(float(r.day)-known_day[other]));
        num += s*known_residual[other]; den += s;
      }
      pred[p] = (external ? base_probe[p] : predict(r.user, i, r.day, f.probe_session[p])) + num/(den+shrink);
      support[p] = den;
      const double err = r.rating-clipped(pred[p]); sse += err*err;
    }
  }
  std::ofstream out(output, std::ios::binary); write_vector(out, pred);
  if (!support_path.empty()) { std::ofstream s(support_path, std::ios::binary); write_vector(s, support); }
  std::cout << std::setprecision(10) << "knn base=" << (external ? base_probe_path : joint_path.empty() ? "time" : joint_path) << " neighbors=" << k << " power=" << power
            << " shrink=" << shrink << " time_decay=" << time_decay << " probe_rmse=" << std::sqrt(sse/d.probe.size()) << std::endl;
}
