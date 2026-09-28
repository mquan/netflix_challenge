// Included inside netflix.cpp's anonymous namespace to share dataset utilities.
// Jointly fitted temporal baseline + SVD++ + temporal user factors.
// Unlike the legacy residual models, every component sees the same SGD error.
struct JointModel {
  TimeBaselineModel base;
  SvdppModel factors;
  std::vector<float> drift;
  uint32_t rank = 0, epoch = 0;
  uint32_t day_rank = 0;
  std::vector<float> day_factors;
  std::vector<float> frequency_factors;
};

struct JointFeatures {
  std::vector<uint32_t> train_session, probe_session;
  std::vector<float> dev;
  std::vector<uint8_t> freq;
};

JointFeatures joint_features(const Dataset &d, const TemporalIndex &t) {
  JointFeatures f;
  if (t.session_days.size() >= UINT32_MAX) fail("too many user-day sessions");
  f.train_session.resize(d.header.train_count);
  f.dev.resize(t.session_days.size());
  f.freq.resize(t.session_days.size());
  for (uint32_t u = 0; u < d.header.users; ++u) {
    for (uint64_t s = t.session_offsets[u]; s < t.session_offsets[u+1]; ++s) {
      f.dev[s] = time_deviation(t.session_days[s], t.mean_day[u]);
      f.freq[s] = static_cast<uint8_t>(frequency_bin(t.session_counts[s]));
    }
    uint64_t s = t.session_offsets[u];
    for (uint64_t p = d.train_offsets[u]; p < d.train_offsets[u+1]; ++p) {
      while (s < t.session_offsets[u+1] && t.session_days[s] < d.train_days[p]) ++s;
      if (s == t.session_offsets[u+1] || t.session_days[s] != d.train_days[p])
        fail("training session missing");
      f.train_session[p] = static_cast<uint32_t>(s);
    }
  }
  for (const auto &r : d.probe)
    f.probe_session.push_back(static_cast<uint32_t>(find_session(t, r.user, r.day)));
  return f;
}

// Dataset fingerprint binds the model to row ordering and all numeric inputs.
// Non-cryptographic: integrity against accidental dataset mismatch, not attacks.
template<class T> void hash_vector(uint64_t &hash, const std::vector<T> &v) {
  const auto *bytes = reinterpret_cast<const unsigned char *>(v.data());
  for (size_t i = 0; i < v.size()*sizeof(T); ++i) {
    hash ^= bytes[i]; hash *= 1099511628211ULL;
  }
}
uint64_t joint_fingerprint(const Dataset &d) {
  uint64_t h = 14695981039346656037ULL;
  hash_vector(h, d.train_offsets); hash_vector(h, d.train_items);
  hash_vector(h, d.train_days); hash_vector(h, d.train_ratings);
  hash_vector(h, d.implicit_offsets); hash_vector(h, d.implicit_items);
  hash_vector(h, d.implicit_days);
  // Hash fields individually: ProbeRec includes compiler padding.
  for (const auto &r : d.probe) {
    h ^= r.user; h *= 1099511628211ULL;
    h ^= r.item; h *= 1099511628211ULL;
    h ^= r.day; h *= 1099511628211ULL;
    h ^= r.rating; h *= 1099511628211ULL;
  }
  return h;
}

JointModel new_joint(const Dataset &d, const TemporalIndex &t, uint32_t rank,
                     uint32_t seed) {
  JointModel m; m.rank = rank;
  auto &b = m.base;
  b.user_bias.assign(d.header.users, 0);
  b.user_alpha.assign(d.header.users, 0);
  b.user_scale.assign(d.header.users, 1);
  b.item_bias.assign(d.header.items, 0);
  b.item_bin_bias.assign(size_t(d.header.items)*kTimeBins, 0);
  b.item_frequency_bias.assign(size_t(d.header.items)*kFrequencyBins, 0);
  b.day_bias.assign(t.session_days.size(), 0);
  b.day_scale.assign(t.session_days.size(), 0);
  auto &v = m.factors;
  v.user_factors.resize(size_t(d.header.users)*rank);
  v.item_factors.resize(size_t(d.header.items)*rank);
  v.implicit_factors.assign(size_t(d.header.items)*rank, 0);
  v.user_bias.assign(d.header.users, 0); // unused compatibility vectors
  v.item_bias.assign(d.header.items, 0);
  m.drift.assign(size_t(d.header.users)*rank, 0);
  std::mt19937 rng(seed);
  std::normal_distribution<float> normal(0, 0.1F/std::sqrt(float(rank)));
  for (float &x : v.user_factors) x = normal(rng);
  for (float &x : v.item_factors) x = normal(rng);
  return m;
}

void save_joint(const fs::path &path, const JointModel &m, uint64_t fingerprint) {
  if (!path.parent_path().empty()) fs::create_directories(path.parent_path());
  std::ofstream out(path.string()+".tmp", std::ios::binary);
  if (!out) fail("cannot write joint model");
  write_pod(out, uint64_t(0x4E46504A4F494E33ULL));
  write_pod(out, fingerprint); write_pod(out, m.rank); write_pod(out, m.epoch);
  const auto &b = m.base; const auto &v = m.factors;
  for (const auto *a : {&b.user_bias, &b.user_alpha, &b.user_scale, &b.item_bias,
       &b.item_bin_bias, &b.item_frequency_bias, &b.day_bias, &b.day_scale,
       &v.user_factors, &v.item_factors, &v.implicit_factors, &m.drift}) {
    write_pod(out, uint64_t(a->size())); write_vector(out, *a);
  }
  write_pod(out, m.day_rank);
  write_pod(out, uint64_t(m.day_factors.size())); write_vector(out, m.day_factors);
  write_pod(out, uint64_t(m.frequency_factors.size())); write_vector(out, m.frequency_factors);
  out.close(); if (!out) fail("joint model write failed");
  fs::rename(path.string()+".tmp", path);
}

JointModel load_joint(const fs::path &path, const Dataset &d,
                       const TemporalIndex &t, uint64_t fingerprint) {
  std::ifstream in(path, std::ios::binary);
  uint64_t magic = 0, hash = 0; uint32_t rank = 0, epoch = 0;
  read_pod(in, magic); read_pod(in, hash); read_pod(in, rank); read_pod(in, epoch);
  if ((magic != 0x4E46504A4F494E31ULL && magic != 0x4E46504A4F494E32ULL && magic != 0x4E46504A4F494E33ULL)
      || hash != fingerprint || rank == 0 || rank > 2048)
    fail("joint model format or dataset fingerprint mismatch");
  JointModel m = new_joint(d, t, rank, 0); m.epoch = epoch;
  auto &b = m.base; auto &v = m.factors;
  for (auto *a : {&b.user_bias, &b.user_alpha, &b.user_scale, &b.item_bias,
       &b.item_bin_bias, &b.item_frequency_bias, &b.day_bias, &b.day_scale,
       &v.user_factors, &v.item_factors, &v.implicit_factors, &m.drift}) {
    uint64_t n = 0; read_pod(in, n);
    if (n != a->size()) fail("joint model vector size mismatch");
    read_vector(in, *a, n);
  }
  if (magic != 0x4E46504A4F494E31ULL) {
    read_pod(in, m.day_rank);
    uint64_t size = 0; read_pod(in, size);
    if (m.day_rank > m.rank || size != t.session_days.size()*m.day_rank)
      fail("invalid day-factor dimensions");
    read_vector(in, m.day_factors, size);
  }
  if (magic == 0x4E46504A4F494E33ULL) {
    uint64_t size = 0; read_pod(in, size);
    if (size && size != size_t(d.header.items)*kFrequencyBins*m.rank) fail("invalid frequency-factor dimensions");
    read_vector(in,m.frequency_factors,size);
  }
  if (in.peek() != std::char_traits<char>::eof()) fail("trailing joint model data");
  return m;
}

float joint_base(const Dataset &d, const JointFeatures &f, const JointModel &m,
                 uint32_t u, uint32_t i, uint16_t day, uint32_t s) {
  const auto &b = m.base;
  return float(d.header.mean) + b.user_bias[u] + b.user_alpha[u]*f.dev[s]
      + b.day_bias[s] + (b.item_bias[i]+b.item_bin_bias[i*kTimeBins+time_bin(d, day)])
      *(b.user_scale[u]+b.day_scale[s]) + b.item_frequency_bias[i*kFrequencyBins+f.freq[s]];
}

double eval_joint(const Dataset &d, const JointFeatures &f, const JointModel &m,
                  std::vector<float> &predictions) {
  std::vector<float> effective;
  build_effective_users(d, m.factors, m.rank, effective);
  predictions.resize(d.probe.size());
  double sse = 0;
  for (size_t n = 0; n < d.probe.size(); ++n) {
    const auto &r = d.probe[n]; auto s = f.probe_session[n]; uint32_t i = r.item-1;
    const float *p = &effective[size_t(r.user)*m.rank];
    const float *a = &m.drift[size_t(r.user)*m.rank];
    const float *q = &m.factors.item_factors[size_t(i)*m.rank];
    const float *fq = m.frequency_factors.empty() ? nullptr :
        &m.frequency_factors[(size_t(i)*kFrequencyBins+f.freq[s])*m.rank];
    double pred = joint_base(d, f, m, r.user, i, r.day, s);
    for (uint32_t k = 0; k < m.rank; ++k) pred += (q[k]+(fq ? fq[k] : 0))*(p[k]+a[k]*f.dev[s]);
    for (uint32_t k = 0; k < m.day_rank; ++k)
      pred += (q[k]+(fq ? fq[k] : 0))*m.day_factors[size_t(s)*m.day_rank+k];
    if (!std::isfinite(pred)) fail("non-finite joint prediction");
    predictions[n] = float(pred); // raw for blending, clip only when scoring
    const double err = r.rating-clipped(predictions[n]); sse += err*err;
  }
  return std::sqrt(sse/d.probe.size());
}

// The within-user implicit sum is updated after every rating. With c=1-lr*reg,
// h' = c*h + lr*err*q. After m ratings: y_j' = c^m*y_j + norm*(h'-c^m*h).
// This exactly recovers naive SVD++ SGD (up to floating point), O(n*f) per user
// instead of O(n^2*f). The legacy frozen-sum batch update is not equivalent.
void check_implicit_update() {
  const double lr = .013, reg = .07, norm = 1/std::sqrt(3.0), c = 1-lr*reg;
  std::vector<double> initial{.4, -.3, .2}, explicit_y = initial;
  double h0 = norm*std::accumulate(initial.begin(), initial.end(), 0.0), h = h0;
  for (double eq : {.6, -.9, .2, 1.1}) {
    for (double &y : explicit_y) y = c*y+lr*norm*eq;
    h = c*h+lr*eq;
    const double reference = norm*std::accumulate(explicit_y.begin(), explicit_y.end(), 0.0);
    if (std::abs(h-reference) > 1e-12) fail("implicit sum recurrence mismatch");
  }
  const double decay = std::pow(c, 4);
  for (size_t j = 0; j < initial.size(); ++j)
    if (std::abs(decay*initial[j]+norm*(h-decay*h0)-explicit_y[j]) > 1e-12)
      fail("implicit deferred update mismatch");
  std::cout << "implicit update matches explicit per-rating SGD\n";
}

void command_joint(const std::vector<std::string> &args, bool eval_only) {
  const fs::path model_path = value_after(args, "--model");
  const Dataset d = load_dataset(value_after(args, "--data"));
  if (d.probe.empty() || d.train_items.empty()) fail("empty train or probe");
  const auto t = build_temporal_index(d);
  const auto f = joint_features(d, t);
  const uint64_t fingerprint = joint_fingerprint(d);
  std::vector<float> predictions;
  auto export_predictions = [&]() {
    const std::string path = optional_value_after(args, "--predictions");
    if (!path.empty()) {
      std::ofstream out(path, std::ios::binary);
      write_vector(out, predictions);
    }
  };
  if (eval_only) {
    const auto m = load_joint(model_path, d, t, fingerprint);
    double score = eval_joint(d, f, m, predictions);
    export_predictions();
    std::cout << std::setprecision(10) << "probe_rmse=" << score << " epoch=" << m.epoch << '\n';
    return;
  }
  const uint32_t rank = uint_after(args, "--factors", 64);
  const uint32_t epochs = uint_after(args, "--iterations", 30);
  const uint32_t patience = uint_after(args, "--patience", 6);
  const bool fixed_epoch = std::find(args.begin(), args.end(), "--fixed-epoch") != args.end();
  const uint32_t seed = uint_after(args, "--seed", 2026);
  const float lr0 = float(double_after(args, "--learning-rate", .008));
  const float reg = float(double_after(args, "--regularization", .0015));
  const float decay = float(double_after(args, "--decay", .95));
  const float base_rate = float(double_after(args, "--base-rate", 1));
  const float drift_rate = float(double_after(args, "--drift-rate", .00001));
  const uint32_t day_rank = uint_after(args, "--day-factors", 0);
  const float day_rate = float(double_after(args, "--day-rate", .004));
  const float day_reg = float(double_after(args, "--day-regularization", .01));
  const bool frequency_factors = uint_after(args, "--frequency-factors", 0) != 0;
  const float frequency_rate = float(double_after(args, "--frequency-rate", .00002));
  const float frequency_reg = float(double_after(args, "--frequency-regularization", .02));
  if (!rank || rank > 2048 || !epochs || lr0 <= 0 || reg < 0 || lr0*reg >= 1
      || decay <= 0 || decay > 1 || base_rate <= 0 || drift_rate < 0)
    fail("invalid joint hyperparameters");
  if (day_rank > rank || day_rate <= 0 || day_reg < 0) fail("invalid day-factor settings");
  if (!std::isfinite(frequency_rate) || !std::isfinite(frequency_reg) || frequency_rate <= 0 || frequency_reg < 0)
    fail("invalid frequency-factor settings");
  const std::string init = optional_value_after(args, "--init-model");
  JointModel m = init.empty() ? new_joint(d, t, rank, seed) : load_joint(init, d, t, fingerprint);
  if (m.rank != rank) fail("initial model rank differs from --factors");
  if (m.day_rank && m.day_rank != day_rank) fail("initial day-factor rank differs");
  if (day_rank && !m.day_rank) {
    m.day_rank = day_rank;
    m.day_factors.assign(t.session_days.size()*day_rank, 0);
  }
  if (!m.frequency_factors.empty() && !frequency_factors) fail("initial model has frequency factors; enable --frequency-factors 1");
  if (frequency_factors && m.frequency_factors.empty())
    m.frequency_factors.assign(size_t(d.header.items)*kFrequencyBins*rank, 0);
  std::mt19937 rng(seed);
  std::vector<uint32_t> users(d.header.users); std::iota(users.begin(), users.end(), 0);
  std::vector<uint64_t> order;
  std::vector<float> h(rank), h0(rank);
  double best = std::numeric_limits<double>::infinity();
  uint32_t stale = 0;
  std::cout << "joint factors=" << rank << " seed=" << seed << " regularization=" << reg
            << " decay=" << decay << " fingerprint=" << fingerprint << std::endl;
  std::cout << "day_factors=" << day_rank << " frequency_factors=" << frequency_factors
            << " init_model=" << init << std::endl;
  // --start-epoch continues a warm start on the original learning-rate schedule.
  const uint32_t start_epoch = uint_after(args, "--start-epoch", 1);
  if (!start_epoch || start_epoch > epochs) fail("invalid --start-epoch");
  for (uint32_t epoch = start_epoch; epoch <= epochs; ++epoch) {
    const auto start = std::chrono::steady_clock::now();
    float lr = lr0*std::pow(decay, float(epoch-1));
    float c = 1-lr*reg;
    std::shuffle(users.begin(), users.end(), rng);
    double sse = 0;
    for (uint32_t u : users) {
      uint64_t ib = d.implicit_offsets[u], ie = d.implicit_offsets[u+1];
      uint64_t begin = d.train_offsets[u], end = d.train_offsets[u+1];
      if (begin == end) continue;
      float norm = 1/std::sqrt(float(ie-ib));
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
        uint32_t i = d.train_items[pos]-1, s = f.train_session[pos];
        auto day = d.train_days[pos]; float dev = f.dev[s];
        float *q = &m.factors.item_factors[size_t(i)*rank];
        uint32_t bin = i*kTimeBins+time_bin(d, day), freq = i*kFrequencyBins+f.freq[s];
        float *fq = frequency_factors ? &m.frequency_factors[size_t(freq)*rank] : nullptr;
        float item = b.item_bias[i]+b.item_bin_bias[bin], scale = b.user_scale[u]+b.day_scale[s];
        float pred = float(d.header.mean)+b.user_bias[u]+b.user_alpha[u]*dev+b.day_bias[s]
                     +item*scale+b.item_frequency_bias[freq];
        for (uint32_t k = 0; k < rank; ++k) pred += (q[k]+(fq ? fq[k] : 0))*(p[k]+a[k]*dev+h[k]);
        float *daily = day_rank ? &m.day_factors[size_t(s)*day_rank] : nullptr;
        for (uint32_t k = 0; k < day_rank; ++k) pred += (q[k]+(fq ? fq[k] : 0))*daily[k];
        float err = float(d.train_ratings[pos])-pred;
        sse += double(err)*err;
        b.user_bias[u] += base_rate*.00267F*(err-.0255F*b.user_bias[u]);
        b.user_alpha[u] += base_rate*.00000311F*(err*dev-3.95F*b.user_alpha[u]);
        b.day_bias[s] += base_rate*.00257F*(err-.00231F*b.day_bias[s]);
        b.item_bias[i] += base_rate*.000488F*(err*scale-.0255F*b.item_bias[i]);
        b.item_bin_bias[bin] += base_rate*.000115F*(err*scale-.0929F*b.item_bin_bias[bin]);
        b.user_scale[u] += base_rate*.00564F*(err*item-.0476F*(b.user_scale[u]-1));
        b.day_scale[s] += base_rate*.00103F*(err*item-.019F*b.day_scale[s]);
        b.item_frequency_bias[freq] += base_rate*.00236F*(err-1.1e-8F*b.item_frequency_bias[freq]);
        for (uint32_t k = 0; k < rank; ++k) {
          float oldq = q[k], oldp = p[k];
          float effective_q = oldq+(fq ? fq[k] : 0);
          const float dayvalue = k < day_rank ? daily[k] : 0;
          q[k] += lr*(err*(oldp+a[k]*dev+h[k]+dayvalue)-reg*oldq);
          if (fq) fq[k] += frequency_rate*(err*(oldp+a[k]*dev+h[k]+dayvalue)-frequency_reg*fq[k]);
          p[k] += lr*(err*effective_q-reg*oldp);
          a[k] += drift_rate*(err*effective_q*dev-50*a[k]);
          h[k] = c*h[k]+lr*err*effective_q;
          if (k < day_rank) daily[k] += day_rate*(err*effective_q-day_reg*dayvalue);
        }
      }
      float shrink = std::pow(c, float(end-begin));
      for (uint32_t k = 0; k < rank; ++k) h[k] = norm*(h[k]-shrink*h0[k]);
      for (uint64_t j = ib; j < ie; ++j) {
        float *y = &m.factors.implicit_factors[size_t(d.implicit_items[j]-1)*rank];
        for (uint32_t k = 0; k < rank; ++k) y[k] = shrink*y[k]+h[k];
      }
    }
    m.epoch = epoch;
    double score = eval_joint(d, f, m, predictions);
    if (!std::isfinite(sse) || !std::isfinite(score)) fail("non-finite training run");
    if (fixed_epoch || score < best) {
      best = std::min(best, score); stale = 0; save_joint(model_path, m, fingerprint); export_predictions();
    } else {
      ++stale;
    }
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    std::cout << std::setprecision(10) << "epoch=" << epoch << " train_rmse=" << std::sqrt(sse/d.header.train_count)
              << " probe_rmse=" << score << " best=" << best << " seconds=" << seconds << std::endl;
    if (!fixed_epoch && patience && stale >= patience) {
      std::cout << "early_stop patience=" << patience << std::endl;
      break;
    }
  }
}
