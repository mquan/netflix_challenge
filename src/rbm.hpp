// Conditional restricted Boltzmann machine (Salakhutdinov, Mnih & Hinton 2007).
// One softmax visible unit per rated movie; hidden units are also conditioned
// on every known (train, probe, qualifying) movie. Probe ratings are never read
// during training; the probe is used only for epoch selection, as elsewhere.
// Users are processed in minibatches split across threads. Each user's samples
// come from a generator seeded by (seed, epoch, user), and per-thread gradients
// are reduced in a fixed order, so results do not depend on thread timing.

struct RbmModel {
  uint32_t hidden = 0, epoch = 0;
  std::vector<float> w, vbias, hbias, d; // w: item*5*hidden, d: item*hidden
  // Optional conditional visible biases (BellKor 2009, sec. VI): rating-specific
  // user bias, user-day bias and movie-frequency bias. Empty when not used.
  std::vector<float> ubias, dbias, fbias; // user*5, session*5, item*kFrequencyBins*5
  bool conditional() const { return !ubias.empty(); }
};

// Sum of the conditional biases for one (user, session, item) into five slots.
inline void rbm_extra_bias(const RbmModel &m, uint32_t u, uint32_t s, uint32_t i,
                           uint8_t freq, float *extra) {
  const float *ub = &m.ubias[size_t(u)*5], *db = &m.dbias[size_t(s)*5];
  const float *fb = &m.fbias[(size_t(i)*kFrequencyBins+freq)*5];
  for (int k = 0; k < 5; ++k) extra[k] = ub[k]+db[k]+fb[k];
}

inline float sigmoid(float x) { return 1/(1+std::exp(-x)); }

template<class F> void parallel_for(uint32_t threads, size_t n, F body) {
  std::vector<std::thread> pool;
  for (uint32_t t = 0; t < threads; ++t)
    pool.emplace_back([&, t]() { for (size_t i = t; i < n; i += threads) body(t, i); });
  for (auto &x : pool) x.join();
}

// Hidden activation probabilities from observed ratings (items/ratings) and known items.
void rbm_hidden(const Dataset &d, const RbmModel &m, uint32_t u,
                const uint8_t *ratings, std::vector<float> &out) {
  const uint32_t F = m.hidden;
  out.assign(m.hbias.begin(), m.hbias.end());
  for (uint64_t p = d.train_offsets[u]; p < d.train_offsets[u+1]; ++p) {
    const float *row = &m.w[(size_t(d.train_items[p]-1)*5+ratings[p-d.train_offsets[u]]-1)*F];
    for (uint32_t j = 0; j < F; ++j) out[j] += row[j];
  }
  const uint64_t ib = d.implicit_offsets[u], ie = d.implicit_offsets[u+1];
  const float norm = ie == ib ? 0 : 1/std::sqrt(float(ie-ib));
  for (uint64_t p = ib; p < ie; ++p) {
    const float *row = &m.d[size_t(d.implicit_items[p]-1)*F];
    for (uint32_t j = 0; j < F; ++j) out[j] += norm*row[j];
  }
  for (float &x : out) x = sigmoid(x);
}

// Softmax over the five ratings of item i, given hidden activations h.
void rbm_visible(const RbmModel &m, uint32_t i, const float *h, float *prob,
                 const float *extra = nullptr) {
  const uint32_t F = m.hidden;
  float best = -1e30F;
  for (int k = 0; k < 5; ++k) {
    const float *row = &m.w[(size_t(i)*5+k)*F];
    float x = m.vbias[i*5+k]+(extra ? extra[k] : 0);
    for (uint32_t j = 0; j < F; ++j) x += h[j]*row[j];
    prob[k] = x; best = std::max(best, x);
  }
  float sum = 0;
  for (int k = 0; k < 5; ++k) { prob[k] = std::exp(prob[k]-best); sum += prob[k]; }
  for (int k = 0; k < 5; ++k) prob[k] /= sum;
}

double eval_rbm(const Dataset &d, const RbmModel &m, uint32_t threads,
                const std::vector<uint64_t> &po, const std::vector<uint32_t> &probe_order,
                std::vector<float> &pred, const JointFeatures *f = nullptr) {
  if (m.conditional() && !f) fail("conditional rbm needs temporal features");
  pred.resize(d.probe.size());
  std::vector<double> sse(threads, 0);
  std::vector<std::vector<float>> hidden(threads);
  parallel_for(threads, d.header.users, [&](uint32_t t, size_t u) {
    if (po[u] == po[u+1]) return;
    rbm_hidden(d, m, uint32_t(u), &d.train_ratings[d.train_offsets[u]], hidden[t]);
    for (uint64_t q = po[u]; q < po[u+1]; ++q) {
      const auto p = probe_order[q]; float prob[5], extra[5];
      const uint32_t i = d.probe[p].item-1;
      if (m.conditional()) {
        const uint32_t s = f->probe_session[p];
        rbm_extra_bias(m, uint32_t(u), s, i, f->freq[s], extra);
      }
      rbm_visible(m, i, hidden[t].data(), prob, m.conditional() ? extra : nullptr);
      float e = 0; for (int k = 0; k < 5; ++k) e += (k+1)*prob[k];
      pred[p] = e; const double err = d.probe[p].rating-clipped(e); sse[t] += err*err;
    }
  });
  return std::sqrt(std::accumulate(sse.begin(), sse.end(), 0.0)/d.probe.size());
}

void save_rbm(const fs::path &path, const RbmModel &m, uint64_t fingerprint) {
  std::ofstream out(path.string()+".tmp", std::ios::binary);
  if (!out) fail("cannot write rbm model");
  write_pod(out, uint64_t(m.conditional() ? 0x4E4650524253324DULL : 0x4E4650524253314DULL));
  write_pod(out, fingerprint);
  write_pod(out, m.hidden); write_pod(out, m.epoch);
  write_vector(out, m.w); write_vector(out, m.vbias); write_vector(out, m.hbias); write_vector(out, m.d);
  if (m.conditional()) {
    for (const auto *a : {&m.ubias, &m.dbias, &m.fbias}) {
      write_pod(out, uint64_t(a->size())); write_vector(out, *a);
    }
  }
  out.close(); if (!out) fail("rbm model write failed");
  fs::rename(path.string()+".tmp", path);
}

RbmModel load_rbm(const fs::path &path, uint32_t items, uint64_t fingerprint) {
  std::ifstream in(path, std::ios::binary);
  uint64_t magic = 0, hash = 0; RbmModel m;
  read_pod(in, magic); read_pod(in, hash); read_pod(in, m.hidden); read_pod(in, m.epoch);
  const bool conditional = magic == 0x4E4650524253324DULL;
  if ((magic != 0x4E4650524253314DULL && !conditional) || hash != fingerprint || !m.hidden || m.hidden > 4096)
    fail("rbm model format or dataset fingerprint mismatch");
  const size_t F = m.hidden;
  read_vector(in, m.w, items*5*F); read_vector(in, m.vbias, items*5);
  read_vector(in, m.hbias, F); read_vector(in, m.d, items*F);
  if (conditional) {
    for (auto *a : {&m.ubias, &m.dbias, &m.fbias}) {
      uint64_t n = 0; read_pod(in, n);
      if (!n || n > (uint64_t(1) << 34)) fail("invalid rbm conditional bias size");
      read_vector(in, *a, n);
    }
    if (m.fbias.size() != size_t(items)*kFrequencyBins*5) fail("rbm frequency bias size mismatch");
  }
  if (in.peek() != std::char_traits<char>::eof()) fail("trailing rbm model data");
  return m;
}

void command_rbm(const std::vector<std::string> &args, bool eval_only) {
  const auto d = load_dataset(value_after(args, "--data"));
  const fs::path model_path = value_after(args, "--model");
  const std::string output = optional_value_after(args, "--predictions");
  const uint64_t fingerprint = bound_fingerprint(args, d);
  const uint32_t threads = std::max(1U, uint_after(args, "--threads", 8));
  const uint32_t items = d.header.items;
  const bool want_conditional = std::find(args.begin(), args.end(), "--conditional-biases") != args.end();
  // Temporal sessions and frequency bins are shared with the joint model.
  const auto t = build_temporal_index(d);
  const auto f = joint_features(d, t);
  const size_t sessions = t.session_days.size();
  std::vector<uint64_t> po(d.header.users+1, 0);
  for (const auto &r : d.probe) ++po[r.user+1];
  std::partial_sum(po.begin(), po.end(), po.begin());
  auto cursor = po; std::vector<uint32_t> probe_order(d.probe.size());
  for (size_t p = 0; p < d.probe.size(); ++p) probe_order[cursor[d.probe[p].user]++] = uint32_t(p);
  std::vector<float> pred;
  auto export_predictions = [&]() {
    if (!output.empty()) { std::ofstream out(output, std::ios::binary); write_vector(out, pred); }
  };
  if (eval_only) {
    const auto m = load_rbm(model_path, items, fingerprint);
    if (m.conditional() && (m.ubias.size() != size_t(d.header.users)*5 || m.dbias.size() != sessions*5))
      fail("rbm conditional bias dimensions mismatch");
    const double score = eval_rbm(d, m, threads, po, probe_order, pred, &f);
    export_predictions();
    // --train-predictions writes the expected rating of every training row in
    // training order, for residual kNN over this model (in-sample: the hidden
    // units saw the row's own rating, as in the published RBM+kNN chains).
    const std::string train_out = optional_value_after(args, "--train-predictions");
    if (!train_out.empty()) {
      std::vector<float> train_pred(d.header.train_count);
      std::vector<std::vector<float>> hidden(threads);
      parallel_for(threads, d.header.users, [&](uint32_t t, size_t u) {
        const uint64_t b = d.train_offsets[u], e = d.train_offsets[u+1];
        if (b == e) return;
        rbm_hidden(d, m, uint32_t(u), &d.train_ratings[b], hidden[t]);
        for (uint64_t p = b; p < e; ++p) {
          float prob[5], extra[5]; const uint32_t i = d.train_items[p]-1;
          if (m.conditional()) {
            const uint32_t s = f.train_session[p];
            rbm_extra_bias(m, uint32_t(u), s, i, f.freq[s], extra);
          }
          rbm_visible(m, i, hidden[t].data(), prob, m.conditional() ? extra : nullptr);
          float ex = 0; for (int k = 0; k < 5; ++k) ex += (k+1)*prob[k];
          train_pred[p] = ex;
        }
      });
      std::ofstream out(train_out, std::ios::binary); write_vector(out, train_pred);
    }
    std::cout << std::setprecision(10) << "probe_rmse=" << score << " epoch=" << m.epoch << '\n';
    return;
  }
  const uint32_t F = uint_after(args, "--hidden", 100);
  const uint32_t epochs = uint_after(args, "--iterations", 60);
  const uint32_t patience = uint_after(args, "--patience", 6);
  // Save every epoch (ending at --iterations) instead of the best probe epoch.
  const bool fixed_epoch = std::find(args.begin(), args.end(), "--fixed-epoch") != args.end();
  const uint32_t batch = uint_after(args, "--batch", 1000);
  const uint32_t seed = uint_after(args, "--seed", 2026);
  // Rates are per summed user gradient, i.e. BellKor-era "per 100 users" rates / 100.
  const float lr_w = float(double_after(args, "--learning-rate", .0015))/100;
  const float lr_vb = float(double_after(args, "--visible-rate", .0012))/100;
  const float lr_hb = float(double_after(args, "--hidden-rate", .1))/100;
  const float lr_d = float(double_after(args, "--conditional-rate", .001))/100;
  const float decay_w = float(double_after(args, "--weight-cost", .0001));
  const float momentum = float(double_after(args, "--momentum", .9));
  const float anneal = float(double_after(args, "--decay", .96));
  const uint32_t cd_max = uint_after(args, "--cd-max", 5);
  const uint32_t cd_every = uint_after(args, "--cd-every", 15);
  // Conditional bias SGD (per summed user gradient, no momentum). User and day
  // biases are user-private and updated in place by the owning thread.
  const float lr_ub = float(double_after(args, "--user-bias-rate", .001));
  const float lr_db = float(double_after(args, "--day-bias-rate", .001));
  const float lr_fb = float(double_after(args, "--frequency-bias-rate", .0012))/100;
  const float cost_ub = float(double_after(args, "--user-bias-cost", .01));
  const float cost_db = float(double_after(args, "--day-bias-cost", .05));
  if (!F || F > 4096 || !epochs || !batch || cd_max < 1 || !cd_every) fail("invalid rbm settings");
  if (lr_ub < 0 || lr_db < 0 || lr_fb < 0 || cost_ub < 0 || cost_db < 0) fail("invalid rbm bias rates");
  const std::string init = optional_value_after(args, "--init-model");
  const uint32_t start_epoch = uint_after(args, "--start-epoch", 1);
  if (!start_epoch || start_epoch > epochs) fail("invalid --start-epoch");
  RbmModel m; m.hidden = F;
  m.w.resize(size_t(items)*5*F); m.vbias.assign(size_t(items)*5, 0);
  m.hbias.assign(F, 0); m.d.assign(size_t(items)*F, 0);
  if (!init.empty()) {
    // Warm start: momentum velocities restart at zero; the epoch-indexed
    // CD length and annealing continue from --start-epoch.
    m = load_rbm(init, items, fingerprint);
    if (m.hidden != F) fail("initial rbm hidden size differs from --hidden");
    if (m.conditional() && !want_conditional) fail("initial model has conditional biases; pass --conditional-biases");
  } else {
    std::mt19937 rng(seed); std::normal_distribution<float> normal(0, .01F);
    for (float &x : m.w) x = normal(rng);
    std::vector<double> counts(size_t(items)*5, 1);
    for (size_t p = 0; p < d.train_items.size(); ++p)
      ++counts[size_t(d.train_items[p]-1)*5+d.train_ratings[p]-1];
    for (uint32_t i = 0; i < items; ++i) {
      double total = 0; for (int k = 0; k < 5; ++k) total += counts[i*5+k];
      for (int k = 0; k < 5; ++k) m.vbias[i*5+k] = float(std::log(counts[i*5+k]/total));
    }
  }
  if (want_conditional && !m.conditional()) {
    // Zero biases keep a warm-started model's predictions unchanged at first.
    m.ubias.assign(size_t(d.header.users)*5, 0); m.dbias.assign(sessions*5, 0);
    m.fbias.assign(size_t(items)*kFrequencyBins*5, 0);
  }
  if (m.conditional() && (m.ubias.size() != size_t(d.header.users)*5 || m.dbias.size() != sessions*5))
    fail("rbm conditional bias dimensions mismatch");
  const bool conditional = m.conditional();
  std::vector<float> vel_w(m.w.size(), 0), vel_vb(m.vbias.size(), 0), vel_hb(F, 0), vel_d(m.d.size(), 0);
  std::vector<float> vel_fb(m.fbias.size(), 0);
  std::vector<std::vector<float>> g_fb(threads, std::vector<float>(m.fbias.size(), 0));
  // Per-thread gradient buffers, touched item rows tracked for sparse reduction.
  std::vector<std::vector<float>> g_w(threads, std::vector<float>(m.w.size(), 0));
  std::vector<std::vector<float>> g_vb(threads, std::vector<float>(m.vbias.size(), 0));
  std::vector<std::vector<float>> g_d(threads, std::vector<float>(m.d.size(), 0));
  std::vector<std::vector<float>> g_hb(threads, std::vector<float>(F, 0));
  std::vector<std::vector<uint8_t>> touched(threads, std::vector<uint8_t>(items, 0));
  std::vector<std::vector<uint8_t>> touched_d(threads, std::vector<uint8_t>(items, 0));
  std::vector<uint32_t> users(d.header.users); std::iota(users.begin(), users.end(), 0);
  std::mt19937 order_rng(seed+1);
  double best = std::numeric_limits<double>::infinity(); uint32_t stale = 0;
  std::cout << "rbm hidden=" << F << " batch=" << batch << " threads=" << threads
            << " seed=" << seed << " conditional_biases=" << conditional
            << " fingerprint=" << fingerprint << std::endl;
  for (uint32_t epoch = start_epoch; epoch <= epochs; ++epoch) {
    const auto start = std::chrono::steady_clock::now();
    const uint32_t cd = std::min(cd_max, 1+2*((epoch-1)/cd_every));
    const float scale = std::pow(anneal, float(epoch-1));
    const float mom = epoch <= 5 ? .5F : momentum;
    std::shuffle(users.begin(), users.end(), order_rng);
    std::vector<double> sse(threads, 0);
    for (size_t first = 0; first < users.size(); first += batch) {
      const size_t last = std::min(users.size(), first+batch);
      // Each thread owns a contiguous slice so reduction order is fixed.
      std::vector<std::thread> pool;
      for (uint32_t t = 0; t < threads; ++t) pool.emplace_back([&, t]() {
        std::vector<float> ph, h(F), nh, prob(5), extra;
        std::vector<uint8_t> sampled;
        auto &gw = g_w[t]; auto &gvb = g_vb[t]; auto &gd = g_d[t]; auto &ghb = g_hb[t];
        const size_t lo = first+(last-first)*t/threads, hi = first+(last-first)*(t+1)/threads;
        for (size_t idx = lo; idx < hi; ++idx) {
          const uint32_t u = users[idx];
          const uint64_t b = d.train_offsets[u], e = d.train_offsets[u+1];
          if (b == e) continue;
          std::mt19937 rng(uint32_t(seed*2654435761U) ^ (epoch*40503U) ^ (u*2246822519U));
          std::uniform_real_distribution<float> uni(0, 1);
          rbm_hidden(d, m, u, &d.train_ratings[b], ph);
          sampled.assign(d.train_ratings.begin()+b, d.train_ratings.begin()+e);
          if (conditional) {
            extra.resize((e-b)*5);
            for (uint64_t p = b; p < e; ++p) {
              const uint32_t s = f.train_session[p];
              rbm_extra_bias(m, u, s, d.train_items[p]-1, f.freq[s], &extra[(p-b)*5]);
            }
          }
          nh = ph;
          for (uint32_t step = 0; step < cd; ++step) {
            for (uint32_t j = 0; j < F; ++j) h[j] = uni(rng) < nh[j] ? 1.F : 0.F;
            for (uint64_t p = b; p < e; ++p) {
              const uint32_t i = d.train_items[p]-1;
              rbm_visible(m, i, h.data(), prob.data(), conditional ? &extra[(p-b)*5] : nullptr);
              if (step == 0) {
                float mean = 0; for (int k = 0; k < 5; ++k) mean += (k+1)*prob[k];
                const double err = d.train_ratings[p]-mean; sse[t] += err*err;
              }
              float x = uni(rng), acc = 0; int k = 0;
              for (; k < 4; ++k) { acc += prob[k]; if (x < acc) break; }
              sampled[p-b] = uint8_t(k+1);
            }
            rbm_hidden(d, m, u, sampled.data(), nh);
          }
          for (uint64_t p = b; p < e; ++p) {
            const uint32_t i = d.train_items[p]-1; touched[t][i] = 1;
            const uint32_t r = d.train_ratings[p]-1, s = sampled[p-b]-1;
            float *pos = &gw[(size_t(i)*5+r)*F], *neg = &gw[(size_t(i)*5+s)*F];
            for (uint32_t j = 0; j < F; ++j) { pos[j] += ph[j]; neg[j] -= nh[j]; }
            gvb[i*5+r] += 1; gvb[i*5+s] -= 1;
            if (conditional) {
              const uint32_t sess = f.train_session[p];
              float *ub = &m.ubias[size_t(u)*5], *db = &m.dbias[size_t(sess)*5];
              float *gf = &g_fb[t][(size_t(i)*kFrequencyBins+f.freq[sess])*5];
              ub[r] += scale*lr_ub; ub[s] -= scale*lr_ub;
              db[r] += scale*lr_db; db[s] -= scale*lr_db;
              gf[r] += 1; gf[s] -= 1;
            }
          }
          if (conditional) {
            // Weight decay once per user visit; day biases shrink per session visit.
            float *ub = &m.ubias[size_t(u)*5];
            for (int k = 0; k < 5; ++k) ub[k] -= scale*lr_ub*cost_ub*float(e-b)*ub[k];
            for (uint64_t p = b; p < e; ++p) {
              float *db = &m.dbias[size_t(f.train_session[p])*5];
              for (int k = 0; k < 5; ++k) db[k] -= scale*lr_db*cost_db*db[k];
            }
          }
          const uint64_t ib = d.implicit_offsets[u], ie = d.implicit_offsets[u+1];
          const float norm = 1/std::sqrt(float(ie-ib));
          for (uint64_t p = ib; p < ie; ++p) {
            const uint32_t i = d.implicit_items[p]-1; touched_d[t][i] = 1;
            float *row = &gd[size_t(i)*F];
            for (uint32_t j = 0; j < F; ++j) row[j] += norm*(ph[j]-nh[j]);
          }
          for (uint32_t j = 0; j < F; ++j) ghb[j] += ph[j]-nh[j];
        }
      });
      for (auto &x : pool) x.join();
      // Parallel over items: reduce thread gradients in thread order and apply momentum updates.
      parallel_for(threads, items, [&](uint32_t, size_t i) {
        const size_t wb = i*5*F, we = wb+5*F;
        // Velocity update: v = mom*v + lr*(g - cost*w).
        for (size_t x = wb; x < we; ++x) vel_w[x] *= mom;
        for (int k = 0; k < 5; ++k) vel_vb[i*5+k] *= mom;
        for (uint32_t t = 0; t < threads; ++t) if (touched[t][i]) {
          for (size_t x = wb; x < we; ++x) { vel_w[x] += scale*lr_w*g_w[t][x]; g_w[t][x] = 0; }
          for (int k = 0; k < 5; ++k) { vel_vb[i*5+k] += scale*lr_vb*g_vb[t][i*5+k]; g_vb[t][i*5+k] = 0; }
          touched[t][i] = 0;
        }
        for (size_t x = wb; x < we; ++x) { vel_w[x] -= scale*lr_w*decay_w*m.w[x]; m.w[x] += vel_w[x]; }
        for (int k = 0; k < 5; ++k) m.vbias[i*5+k] += vel_vb[i*5+k];
        if (conditional) {
          const size_t fb = i*kFrequencyBins*5, fe = fb+kFrequencyBins*5;
          for (size_t x = fb; x < fe; ++x) vel_fb[x] *= mom;
          for (uint32_t t = 0; t < threads; ++t)
            for (size_t x = fb; x < fe; ++x) { vel_fb[x] += scale*lr_fb*g_fb[t][x]; g_fb[t][x] = 0; }
          for (size_t x = fb; x < fe; ++x) m.fbias[x] += vel_fb[x];
        }
        const size_t db = i*F, de = db+F;
        for (size_t x = db; x < de; ++x) vel_d[x] *= mom;
        for (uint32_t t = 0; t < threads; ++t) if (touched_d[t][i]) {
          for (size_t x = db; x < de; ++x) { vel_d[x] += scale*lr_d*g_d[t][x]; g_d[t][x] = 0; }
          touched_d[t][i] = 0;
        }
        for (size_t x = db; x < de; ++x) { vel_d[x] -= scale*lr_d*decay_w*m.d[x]; m.d[x] += vel_d[x]; }
      });
      for (uint32_t j = 0; j < F; ++j) {
        vel_hb[j] *= mom;
        for (uint32_t t = 0; t < threads; ++t) { vel_hb[j] += scale*lr_hb*g_hb[t][j]; g_hb[t][j] = 0; }
        m.hbias[j] += vel_hb[j];
      }
    }
    m.epoch = epoch;
    const double score = eval_rbm(d, m, threads, po, probe_order, pred, &f);
    if (!std::isfinite(score)) fail("non-finite rbm run");
    if (fixed_epoch || score < best) {
      best = std::min(best, score); stale = 0; save_rbm(model_path, m, fingerprint); export_predictions();
    } else {
      ++stale;
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    std::cout << std::setprecision(10) << "epoch=" << epoch << " cd=" << cd
              << " train_rmse=" << std::sqrt(std::accumulate(sse.begin(), sse.end(), 0.0)/d.header.train_count)
              << " probe_rmse=" << score << " best=" << best << " seconds=" << seconds << std::endl;
    if (!fixed_epoch && patience && stale >= patience) {
      std::cout << "early_stop patience=" << patience << std::endl;
      break;
    }
  }
}
