// Binary attribute representation (Deyo & Elser, arXiv:2304.14209).
// Dense divide-and-concur RRR with float32 replicas and train-only statistics.
// Missing pairs retain unconstrained replicas, as in the paper's dense method.
// --projection paper is identity inside the tolerance band. Upstream code
// instead projects to its center there; --projection upstream reproduces that.
// Extract bits by rounding concurred values, as in upstream storebits().
struct BarModel {
  uint32_t rank = 0, epoch = 0;
  std::vector<double> item_mean, user_bias, weights;
  std::vector<uint32_t> bits;
};

// Nearest bilinear projection: w'=(w+t*x)/(1-t²), x'=(x+t*w)/(1-t²).
// Solve the monotone scalar constraint with safeguarded Newton iterations.
double bar_multiplier(double dot, double norm, double target) {
  double lo = -1+1e-12, hi = 1-1e-12, t = 0;
  for (int step = 0; step < 60; ++step) {
    const double den = 1-t*t;
    const double value = (dot*(1+t*t)+norm*t)/(den*den);
    if (std::abs(value-target) < 1e-9*(1+std::abs(target))) return t;
    if (value < target) lo = t; else hi = t;
    const double derivative = (2*dot*t*(3+t*t)+norm*(1+3*t*t))/(den*den*den);
    const double next = t-(value-target)/derivative;
    t = std::isfinite(next) && next > lo && next < hi ? next : .5*(lo+hi);
  }
  return t;
}

void bar_project(const float *w, const float *x, float *aw, float *ax,
                 uint32_t rank, bool observed, double residual, bool interval_identity = true) {
  if (!observed) {
    std::copy(w, w+rank, aw); std::copy(x, x+rank, ax); return;
  }
  double dot = 0, norm = 0;
  for (uint32_t k = 0; k < rank; ++k) {
    dot += double(w[k])*x[k]; norm += double(w[k])*w[k]+double(x[k])*x[k];
  }
  const bool inside = std::abs(dot-residual) <= .5;
  if (inside && interval_identity) {
    std::copy(w, w+rank, aw); std::copy(x, x+rank, ax); return;
  }
  const double target = inside ? residual : (dot < residual ? residual-.5 : residual+.5);
  if (norm < 1e-24) {
    std::fill(aw, aw+rank, 0); std::fill(ax, ax+rank, 0);
    aw[0] = float(std::sqrt(std::abs(target)));
    ax[0] = std::copysign(aw[0], float(target)); return;
  }
  const double t = bar_multiplier(dot, norm, target), den = 1-t*t;
  for (uint32_t k = 0; k < rank; ++k) {
    aw[k] = float((w[k]+t*x[k])/den); ax[k] = float((x[k]+t*w[k])/den);
  }
}

BarModel bar_baseline(const Dataset &d, uint32_t rank) {
  BarModel m; m.rank = rank;
  m.item_mean.assign(d.header.items, 0); m.user_bias.assign(d.header.users, 0);
  m.bits.assign(d.header.users, 0); m.weights.assign(size_t(d.header.items)*rank, 0);
  std::vector<uint64_t> counts(d.header.items, 0);
  for (size_t p = 0; p < d.train_items.size(); ++p) {
    const auto i = d.train_items[p]-1;
    m.item_mean[i] += d.train_ratings[p]; ++counts[i];
  }
  for (size_t i = 0; i < counts.size(); ++i)
    m.item_mean[i] = counts[i] ? m.item_mean[i]/counts[i] : d.header.mean;
  for (uint32_t u = 0; u < d.header.users; ++u) {
    const auto begin = d.train_offsets[u], end = d.train_offsets[u+1];
    for (auto p = begin; p < end; ++p)
      m.user_bias[u] += d.train_ratings[p]-m.item_mean[d.train_items[p]-1];
    if (end > begin) m.user_bias[u] /= double(end-begin);
  }
  return m;
}

double bar_prediction(const BarModel &m, uint32_t u, uint32_t i) {
  double pred = m.item_mean[i]+m.user_bias[u];
  for (uint32_t mask = m.bits[u]; mask; mask &= mask-1)
    pred += m.weights[size_t(i)*m.rank+__builtin_ctz(mask)];
  return pred;
}

double eval_bar(const Dataset &d, const BarModel &m, const std::string &output) {
  double sse = 0; std::vector<float> pred(d.probe.size());
  for (size_t n = 0; n < d.probe.size(); ++n) {
    const auto &r = d.probe[n]; pred[n] = float(bar_prediction(m, r.user, r.item-1));
    if (!std::isfinite(pred[n])) fail("non-finite BAR prediction");
    const double error = r.rating-clipped(pred[n]); sse += error*error;
  }
  if (!output.empty()) {
    std::ofstream out(output, std::ios::binary); write_vector(out, pred);
  }
  return std::sqrt(sse/pred.size());
}

void save_bar(const fs::path &path, const BarModel &m, uint64_t fingerprint) {
  if (!path.parent_path().empty()) fs::create_directories(path.parent_path());
  std::ofstream out(path.string()+".tmp", std::ios::binary);
  write_pod(out, uint64_t(0x4e46504241523031ULL)); write_pod(out, fingerprint);
  write_pod(out, m.rank); write_pod(out, m.epoch);
  write_vector(out, m.item_mean); write_vector(out, m.user_bias);
  write_vector(out, m.bits); write_vector(out, m.weights);
  out.close(); if (!out) fail("BAR model write failed");
  fs::rename(path.string()+".tmp", path);
}

BarModel load_bar(const fs::path &path, const Dataset &d, uint64_t fingerprint) {
  std::ifstream in(path, std::ios::binary); uint64_t magic = 0, hash = 0; BarModel m;
  read_pod(in, magic); read_pod(in, hash); read_pod(in, m.rank); read_pod(in, m.epoch);
  if (magic != 0x4e46504241523031ULL || hash != fingerprint || !m.rank || m.rank > 16)
    fail("BAR format or dataset fingerprint mismatch");
  read_vector(in, m.item_mean, d.header.items); read_vector(in, m.user_bias, d.header.users);
  read_vector(in, m.bits, d.header.users); read_vector(in, m.weights, size_t(d.header.items)*m.rank);
  if (in.peek() != std::char_traits<char>::eof()) fail("trailing BAR model data");
  return m;
}

// Aggregate binary patterns before forming per-movie normal equations.
// This avoids an O(ratings * rank²) pass over the 99M training records.
void fit_bar_weights(const Dataset &d, BarModel &m, double ridge) {
  const uint32_t patterns = 1U << m.rank, F = m.rank, items = d.header.items;
  // Process items in blocks so a 16-bit model does not allocate 14 GiB here.
  const uint32_t block = std::min(items, 8000000U/patterns);
  for (uint32_t first = 0; first < items; first += block) {
    const uint32_t count = std::min(block, items-first);
    std::vector<uint32_t> counts(size_t(count)*patterns, 0);
    std::vector<double> sums(counts.size(), 0);
    for (uint32_t u = 0; u < d.header.users; ++u) {
      for (auto p = d.train_offsets[u]; p < d.train_offsets[u+1]; ++p) {
        const uint32_t i = d.train_items[p]-1;
        if (i < first || i >= first+count) continue;
        const size_t g = size_t(i-first)*patterns+m.bits[u];
        ++counts[g]; sums[g] += d.train_ratings[p]-m.item_mean[i]-m.user_bias[u];
      }
    }
    for (uint32_t i = first; i < first+count; ++i) {
      std::vector<double> gram(size_t(F)*F, 0), rhs(F, 0);
      for (uint32_t k = 0; k < F; ++k) gram[k*F+k] = ridge;
      for (uint32_t mask = 1; mask < patterns; ++mask) {
        const size_t g = size_t(i-first)*patterns+mask;
        if (!counts[g]) continue;
        for (uint32_t a = mask; a; a &= a-1) {
          const auto k = uint32_t(__builtin_ctz(a)); rhs[k] += sums[g];
          for (uint32_t b = mask; b; b &= b-1)
            gram[k*F+uint32_t(__builtin_ctz(b))] += counts[g];
        }
      }
      const auto weights = solve_blend(gram, rhs, 0, 0);
      std::copy(weights.begin(), weights.end(), m.weights.begin()+size_t(i)*F);
    }
  }
}

void command_bar(const std::vector<std::string> &args, bool eval_only) {
  const auto d = load_dataset(value_after(args, "--data"));
  if (d.probe.empty() || d.train_items.empty()) fail("empty train or probe");
  const auto model = value_after(args, "--model");
  const auto output = optional_value_after(args, "--predictions");
  const auto fingerprint = joint_fingerprint(d);
  if (eval_only) {
    const auto m = load_bar(model, d, fingerprint);
    const auto score = eval_bar(d, m, output);
    std::cout << std::setprecision(10) << "probe_rmse=" << score << " count=" << d.probe.size() << '\n';
    return;
  }
  const uint32_t F = uint_after(args, "--attributes", 8);
  const uint32_t C = uint_after(args, "--movies", 59), U = d.header.users;
  const uint32_t iterations = uint_after(args, "--iterations", 200);
  const uint32_t every = uint_after(args, "--eval-every", 10);
  const uint32_t threads = uint_after(args, "--threads", 4), seed = uint_after(args, "--seed", 2026);
  const double beta = double_after(args, "--beta", .5), ridge = double_after(args, "--ridge", .001);
  const double max_gib = double_after(args, "--max-replica-gib", 8);
  const auto projection = value_after(args, "--projection", "paper");
  if (projection != "paper" && projection != "upstream") fail("invalid BAR projection");
  if (!F || F > 16 || !C || C > d.header.items || !iterations || !every || !threads || threads > 64
      || beta <= 0 || beta >= 2 || ridge <= 0 || max_gib <= 0) fail("invalid BAR settings");
  const size_t slots = size_t(U)*C*F;
  const double gib = double(slots)*4*sizeof(float)/(1ULL<<30);
  if (gib > max_gib) fail("BAR replicas exceed --max-replica-gib");
  auto m = bar_baseline(d, F);
  std::vector<double> movie_sse(d.header.items, 0);
  for (size_t p = 0; p < d.train_items.size(); ++p) {
    const auto i = d.train_items[p]-1; const auto e = d.train_ratings[p]-m.item_mean[i];
    movie_sse[i] += e*e;
  }
  std::vector<uint32_t> selected(d.header.items); std::iota(selected.begin(), selected.end(), 0);
  std::sort(selected.begin(), selected.end(), [&](auto a, auto b) {
    return movie_sse[a] == movie_sse[b] ? a < b : movie_sse[a] > movie_sse[b];
  });
  selected.resize(C); std::vector<int32_t> positions(d.header.items, -1);
  for (uint32_t c = 0; c < C; ++c) positions[selected[c]] = int32_t(c);
  std::vector<uint8_t> observed(size_t(U)*C, 0);
  uint64_t nobs = 0;
  for (uint32_t u = 0; u < U; ++u)
    for (auto p = d.train_offsets[u]; p < d.train_offsets[u+1]; ++p) {
      const auto c = positions[d.train_items[p]-1];
      if (c >= 0) { observed[size_t(u)*C+size_t(c)] = d.train_ratings[p]; ++nobs; }
    }
  if (!nobs) fail("BAR subset has no observations");
  std::cout << std::setprecision(10) << "bar attributes=" << F << " movies=" << C << " users=" << U
            << " selected_ratings=" << nobs << " replica_gib=" << gib << " seed=" << seed
            << " beta=" << beta << " ridge=" << ridge << " threads=" << threads
            << " selection=training_subset_rmse projection=" << projection << std::endl;
  std::vector<float> w(slots), x(slots), aw(slots), ax(slots), y(size_t(U)*F);
  std::vector<double> wb(size_t(C)*F), yb(size_t(U)*F);
  std::vector<std::vector<double>> partial(threads, std::vector<double>(wb.size()));
  std::mt19937 rng(seed); std::uniform_real_distribution<float> uniform(0, 1);
  auto initialize = [&]() {
    for (size_t p = 0; p < slots; ++p) { w[p] = 4*(uniform(rng)-.5F)/F; x[p] = .5F*uniform(rng); }
    for (float &value : y) value = .5F*uniform(rng);
  };
  initialize(); double best = std::numeric_limits<double>::infinity();
  const auto started = std::chrono::steady_clock::now();
  for (uint32_t epoch = 1; epoch <= iterations; ++epoch) {
    for (auto &sums : partial) std::fill(sums.begin(), sums.end(), 0);
    parallel_for(threads, threads, [&](uint32_t thread, size_t) {
      for (size_t u = size_t(U)*thread/threads; u < size_t(U)*(thread+1)/threads; ++u) {
        double *ub = &yb[u*F];
        for (uint32_t k = 0; k < F; ++k) ub[k] = 2*(y[u*F+k] >= .5F)-y[u*F+k];
        for (uint32_t c = 0; c < C; ++c) {
          const auto rating = observed[u*C+c]; const size_t offset = (u*C+c)*F;
          bar_project(&w[offset], &x[offset], &aw[offset], &ax[offset], F, rating != 0,
                      rating-m.item_mean[selected[c]]-m.user_bias[u], projection == "paper");
          for (uint32_t k = 0; k < F; ++k) {
            partial[thread][c*F+k] += 2*double(aw[offset+k])-w[offset+k];
            ub[k] += 2*double(ax[offset+k])-x[offset+k];
          }
        }
        for (uint32_t k = 0; k < F; ++k) ub[k] /= C+1;
      }
    });
    std::fill(wb.begin(), wb.end(), 0);
    for (const auto &sums : partial)
      for (size_t p = 0; p < wb.size(); ++p) wb[p] += sums[p]/U;
    parallel_for(threads, threads, [&](uint32_t thread, size_t) {
      for (size_t u = size_t(U)*thread/threads; u < size_t(U)*(thread+1)/threads; ++u) {
        for (uint32_t c = 0; c < C; ++c) {
          const size_t offset = (u*C+c)*F;
          for (uint32_t k = 0; k < F; ++k) {
            w[offset+k] += float(beta*(wb[c*F+k]-aw[offset+k]));
            x[offset+k] += float(beta*(yb[u*F+k]-ax[offset+k]));
          }
        }
        for (uint32_t k = 0; k < F; ++k)
          y[u*F+k] += float(beta*(yb[u*F+k]-(y[u*F+k] >= .5F)));
      }
    });
    if (epoch % every && epoch != iterations) continue;
    std::vector<uint32_t> bits(U, 0); double sse = 0;
    for (uint32_t u = 0; u < U; ++u) {
      for (uint32_t k = 0; k < F; ++k) if (yb[size_t(u)*F+k] >= .5) bits[u] |= 1U<<k;
      for (uint32_t c = 0; c < C; ++c) {
        const auto rating = observed[size_t(u)*C+c]; if (!rating) continue;
        double pred = m.item_mean[selected[c]]+m.user_bias[u];
        for (uint32_t k = 0; k < F; ++k) if (bits[u] & (1U<<k)) pred += wb[c*F+k];
        const double error = pred-rating; sse += error*error;
      }
    }
    const double rmse = std::sqrt(sse/nobs);
    if (!std::isfinite(rmse)) fail("BAR optimizer diverged");
    if (rmse < best) { best = rmse; m.bits = std::move(bits); m.epoch = epoch; }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
    std::cout << "epoch=" << epoch << " subset_train_rmse=" << rmse << " best=" << best
              << " seconds=" << seconds << std::endl;
    if (rmse > 1.1) { std::cout << "restart=" << epoch << std::endl; initialize(); }
  }
  // Release dense optimizer state before fitting weights on all training movies.
  std::vector<float>().swap(w); std::vector<float>().swap(x);
  std::vector<float>().swap(aw); std::vector<float>().swap(ax);
  std::cout << "fitting_all_movie_weights best_epoch=" << m.epoch << std::endl;
  fit_bar_weights(d, m, ridge);
  save_bar(model, m, fingerprint);
  double train_sse = 0;
  for (uint32_t u = 0; u < U; ++u)
    for (auto p = d.train_offsets[u]; p < d.train_offsets[u+1]; ++p) {
      const double error = d.train_ratings[p]-clipped(bar_prediction(m, u, d.train_items[p]-1));
      train_sse += error*error;
    }
  const double score = eval_bar(d, m, output);
  std::cout << "train_rmse=" << std::sqrt(train_sse/d.train_items.size()) << " probe_rmse=" << score
            << " count=" << d.probe.size() << " best_epoch=" << m.epoch << std::endl;
}
