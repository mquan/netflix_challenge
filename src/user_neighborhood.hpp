// Included inside netflix.cpp's anonymous namespace (after joint.hpp).
//
// User-user kNN on a base model's residuals (Bell & Koren 2007 user-oriented
// neighborhood; the BigChaos user kNN). Residual r_vj = rating - base(v, j) on
// every training row. For a target (u, i):
//
//   pred = base(u, i) + sum_v w_uv r_vi / (sum_v w_uv + shrink)
//
// over the K neighbors v who rated i with the largest weights. The weight is
// a shrunk residual correlation on the items u and v both rated,
//   rho = sum r_uj r_vj / sqrt(sum r_uj^2 sum r_vj^2),  rho *= n/(n+alpha),
//   w = max(0, rho)^power * exp(-time_decay*|day - day_vi|).
//
// Exact correlations against every rater of i are infeasible (7.3e10 query x
// rater pairs on the Netflix probe). Candidates are therefore screened by
// cosine similarity of the base joint model's effective user vectors
// (p_u + |N(u)|^-1/2 sum y_j), optionally projected onto their top principal
// directions (--screen-rank); the --candidates best raters of i are rescored
// with the exact residual correlation. The screen cosine also gives a second,
// factor-similarity column (--factor-predictions), as in Bell & Koren's
// factor-derived user similarities.
//
// The "probe" rows of --data are the targets: the original probe on the full
// dataset, the train-matched holdout on artifacts/validation/train-matched-*.
// Probe ratings are read only to print the score.

struct UserKnnCandidate { float score; uint32_t index; };

template <class F> void user_knn_parallel(uint32_t threads, size_t tasks, F &&fn) {
  std::atomic<size_t> next{0};
  auto work = [&](uint32_t worker) {
    for (size_t k; (k = next.fetch_add(1)) < tasks;) fn(worker, k);
  };
  std::vector<std::thread> pool;
  for (uint32_t w = 1; w < threads; ++w) pool.emplace_back(work, w);
  work(0);
  for (auto &th : pool) th.join();
}

// Min-heap of the best `capacity` screen scores; ties keep the earlier rater,
// so results do not depend on thread count.
inline void user_knn_push(UserKnnCandidate *heap, uint32_t &size, uint32_t capacity,
                          float score, uint32_t index) {
  if (size < capacity) {
    uint32_t c = size++;
    while (c) {
      uint32_t parent = (c-1)/2;
      if (heap[parent].score <= score) break;
      heap[c] = heap[parent]; c = parent;
    }
    heap[c] = {score, index};
  } else if (score > heap[0].score) {
    uint32_t c = 0;
    for (;;) {
      uint32_t l = 2*c+1, r = l+1, m = c;
      float best = score;
      if (l < size && heap[l].score < best) { m = l; best = heap[l].score; }
      if (r < size && heap[r].score < best) m = r;
      if (m == c) break;
      heap[c] = heap[m]; c = m;
    }
    heap[c] = {score, index};
  }
}

// Unit-length screen vectors: the effective user vectors normalized, then (if
// 0 < dim < rank) projected on the top `dim` eigenvectors of their uncentered
// second-moment matrix and renormalized.
std::vector<float> user_knn_screen_vectors(std::vector<float> &x, uint32_t users, uint32_t rank,
                                           uint32_t &dim, uint32_t threads) {
  auto normalize = [](float *v, uint32_t n) {
    double s = 0; for (uint32_t k = 0; k < n; ++k) s += double(v[k])*v[k];
    const float inv = s > 1e-20 ? float(1/std::sqrt(s)) : 0.F;
    for (uint32_t k = 0; k < n; ++k) v[k] *= inv;
  };
  for (uint32_t u = 0; u < users; ++u) normalize(&x[size_t(u)*rank], rank);
  if (!dim || dim >= rank) { dim = rank; return std::move(x); }
  const uint32_t blocks = (users+4095)/4096;
  std::vector<std::vector<double>> partial(threads, std::vector<double>(size_t(rank)*rank, 0));
  user_knn_parallel(threads, blocks, [&](uint32_t w, size_t b) {
    auto &mm = partial[w];
    for (uint32_t u = uint32_t(b*4096); u < std::min<uint64_t>(users, (b+1)*4096); ++u) {
      const float *v = &x[size_t(u)*rank];
      for (uint32_t a = 0; a < rank; ++a) {
        const double va = v[a]; double *row = &mm[size_t(a)*rank];
        for (uint32_t c = a; c < rank; ++c) row[c] += va*v[c];
      }
    }
  });
  std::vector<double> mm(size_t(rank)*rank, 0);
  for (const auto &p : partial) for (size_t k = 0; k < mm.size(); ++k) mm[k] += p[k];
  for (uint32_t a = 0; a < rank; ++a) for (uint32_t c = 0; c < a; ++c) mm[size_t(a)*rank+c] = mm[size_t(c)*rank+a];
  // Orthogonal (subspace) iteration for the top-dim eigenvectors.
  std::mt19937 rng(20261002); std::normal_distribution<double> normal(0, 1);
  std::vector<double> v(size_t(rank)*dim), next(v.size());
  for (auto &e : v) e = normal(rng);
  auto orthonormalize = [&](std::vector<double> &a) {
    for (uint32_t j = 0; j < dim; ++j) {
      for (uint32_t i = 0; i < j; ++i) {
        double dot = 0; for (uint32_t k = 0; k < rank; ++k) dot += a[size_t(k)*dim+i]*a[size_t(k)*dim+j];
        for (uint32_t k = 0; k < rank; ++k) a[size_t(k)*dim+j] -= dot*a[size_t(k)*dim+i];
      }
      double s = 0; for (uint32_t k = 0; k < rank; ++k) s += a[size_t(k)*dim+j]*a[size_t(k)*dim+j];
      s = 1/std::sqrt(std::max(s, 1e-300));
      for (uint32_t k = 0; k < rank; ++k) a[size_t(k)*dim+j] *= s;
    }
  };
  orthonormalize(v);
  for (int iter = 0; iter < 100; ++iter) {
    std::fill(next.begin(), next.end(), 0);
    for (uint32_t a = 0; a < rank; ++a)
      for (uint32_t c = 0; c < rank; ++c) {
        const double m = mm[size_t(a)*rank+c];
        for (uint32_t j = 0; j < dim; ++j) next[size_t(a)*dim+j] += m*v[size_t(c)*dim+j];
      }
    orthonormalize(next); v.swap(next);
  }
  double total = 0, kept = 0;
  for (uint32_t a = 0; a < rank; ++a) total += mm[size_t(a)*rank+a];
  for (uint32_t j = 0; j < dim; ++j) {
    double q = 0;
    for (uint32_t a = 0; a < rank; ++a) {
      double s = 0; for (uint32_t c = 0; c < rank; ++c) s += mm[size_t(a)*rank+c]*v[size_t(c)*dim+j];
      q += v[size_t(a)*dim+j]*s;
    }
    kept += q;
  }
  std::cout << "screen_projection rank=" << rank << " dim=" << dim
            << " energy_kept=" << std::setprecision(4) << kept/std::max(total, 1e-300) << std::endl;
  std::vector<float> z(size_t(users)*dim), vf(v.begin(), v.end());
  user_knn_parallel(threads, blocks, [&](uint32_t, size_t b) {
    for (uint32_t u = uint32_t(b*4096); u < std::min<uint64_t>(users, (b+1)*4096); ++u) {
      const float *xu = &x[size_t(u)*rank]; float *zu = &z[size_t(u)*dim];
      for (uint32_t a = 0; a < rank; ++a) {
        const float xa = xu[a]; const float *row = &vf[size_t(a)*dim];
        for (uint32_t j = 0; j < dim; ++j) zu[j] += xa*row[j];
      }
      normalize(zu, dim);
    }
  });
  return z;
}

void command_user_knn(const std::vector<std::string> &args) {
  using clock = std::chrono::steady_clock;
  const auto start = clock::now();
#ifdef __APPLE__
  setenv("VECLIB_MAXIMUM_THREADS", "1", 0);  // parallelism comes from --threads
#endif
  auto seconds_since = [](clock::time_point t0) {
    return std::chrono::duration<double>(clock::now()-t0).count();
  };
  const auto d = load_dataset(value_after(args, "--data"));
  const auto t = build_temporal_index(d);
  const auto f = joint_features(d, t);
  // The joint model always supplies the screen vectors; it is also the base
  // unless --base-train/--base-probe give another model's raw predictions for
  // every training row (training order) and probe row.
  const auto joint_path = value_after(args, "--joint-model");
  const JointModel m = load_joint(joint_path, d, t, joint_fingerprint(d));
  const auto output = value_after(args, "--predictions");
  const auto support_path = optional_value_after(args, "--support");
  const auto count_path = optional_value_after(args, "--support-count");
  const auto factor_path = optional_value_after(args, "--factor-predictions");
  const auto factor_support_path = optional_value_after(args, "--factor-support");
  const auto base_train_path = optional_value_after(args, "--base-train");
  const auto base_probe_path = optional_value_after(args, "--base-probe");
  const uint32_t threads = std::max(1U, uint_after(args, "--threads", 1));
  uint32_t dim = uint_after(args, "--screen-rank", 0);
  const uint32_t candidates = uint_after(args, "--candidates", 300);
  const uint32_t k = std::min(candidates, uint_after(args, "--neighbors", 100));
  const uint32_t max_raters = uint_after(args, "--max-raters", 0);
  const uint32_t min_common = uint_after(args, "--min-common", 2);
  const uint32_t query_modulo = std::max(1U, uint_after(args, "--query-modulo", 1));
  const float alpha = float(double_after(args, "--alpha", 50));
  const float power = float(double_after(args, "--power", 2));
  const float shrink = float(double_after(args, "--shrink", 2));
  const float factor_power = float(double_after(args, "--factor-power", 4));
  const float factor_shrink = float(double_after(args, "--factor-shrink", 64));
  const float time_decay = float(double_after(args, "--time-decay", 0));
  if (!candidates || !k || alpha < 0 || power <= 0 || shrink < 0 || factor_power <= 0 ||
      factor_shrink < 0 || time_decay < 0 || !std::isfinite(time_decay))
    fail("invalid user-knn parameters");
  const bool external = !base_train_path.empty();
  if (external != !base_probe_path.empty()) fail("--base-train and --base-probe go together");
  const uint32_t U = d.header.users, I = d.header.items;
  const uint64_t T = d.header.train_count;
  const size_t P = d.probe.size();

  // Phase 1: base predictions and training residuals.
  auto t0 = clock::now();
  std::vector<float> residual(T), base_probe(P);
  if (external) {
    std::ifstream tin(base_train_path, std::ios::binary), pin(base_probe_path, std::ios::binary);
    if (!tin || !pin) fail("cannot read base prediction files");
    read_vector(tin, residual, T); read_vector(pin, base_probe, P);
    if (tin.peek() != std::char_traits<char>::eof() || pin.peek() != std::char_traits<char>::eof())
      fail("base prediction file size mismatch");
    for (float x : residual) if (!std::isfinite(x)) fail("non-finite base prediction");
    for (float x : base_probe) if (!std::isfinite(x)) fail("non-finite base prediction");
    for (uint64_t p = 0; p < T; ++p) residual[p] = float(d.train_ratings[p])-residual[p];
  }
  std::vector<float> effective;
  build_effective_users(d, m.factors, m.rank, effective);
  if (!external) {
    auto predict = [&](uint32_t u, uint32_t i, uint16_t day, uint32_t s) {
      float x = joint_base(d, f, m, u, i, day, s);
      const float *p = &effective[size_t(u)*m.rank];
      const float *a = &m.drift[size_t(u)*m.rank];
      const float *q = &m.factors.item_factors[size_t(i)*m.rank];
      const float *fq = m.frequency_factors.empty() ? nullptr :
          &m.frequency_factors[(size_t(i)*kFrequencyBins+f.freq[s])*m.rank];
      for (uint32_t r = 0; r < m.rank; ++r) x += (q[r]+(fq ? fq[r] : 0))*(p[r]+a[r]*f.dev[s]);
      for (uint32_t r = 0; r < m.day_rank; ++r)
        x += (q[r]+(fq ? fq[r] : 0))*m.day_factors[size_t(s)*m.day_rank+r];
      return x;
    };
    user_knn_parallel(threads, (U+1023)/1024, [&](uint32_t, size_t b) {
      for (uint32_t u = uint32_t(b*1024); u < std::min<uint64_t>(U, (b+1)*1024); ++u)
        for (uint64_t p = d.train_offsets[u]; p < d.train_offsets[u+1]; ++p)
          residual[p] = float(d.train_ratings[p])-predict(u, d.train_items[p]-1, d.train_days[p], f.train_session[p]);
    });
    for (size_t p = 0; p < P; ++p) {
      const auto &r = d.probe[p];
      base_probe[p] = predict(r.user, r.item-1, r.day, f.probe_session[p]);
    }
  }
  double train_sse = 0;
  for (float x : residual) train_sse += double(x)*x;
  std::cout << "user_knn base=" << (external ? base_probe_path : joint_path)
            << " train_rmse=" << std::setprecision(6) << std::sqrt(train_sse/double(T))
            << " residual_seconds=" << seconds_since(t0) << std::endl;

  // Phase 2: screen vectors.
  t0 = clock::now();
  const auto z = user_knn_screen_vectors(effective, U, m.rank, dim, threads);
  effective.clear(); effective.shrink_to_fit();
  std::cout << "screen_seconds=" << seconds_since(t0) << std::endl;

  // Phase 3: item-major index of training rows and per-item query lists.
  std::vector<uint64_t> item_offsets(size_t(I)+1, 0);
  for (uint64_t p = 0; p < T; ++p) ++item_offsets[d.train_items[p]];
  std::partial_sum(item_offsets.begin(), item_offsets.end(), item_offsets.begin());
  std::vector<uint32_t> item_user(T), item_pos(T);
  {
    auto cursor = item_offsets;
    for (uint32_t u = 0; u < U; ++u)
      for (uint64_t p = d.train_offsets[u]; p < d.train_offsets[u+1]; ++p) {
        const uint64_t c = cursor[d.train_items[p]-1]++;
        item_user[c] = u; item_pos[c] = uint32_t(p);
      }
  }
  if (T >= UINT32_MAX) fail("too many training rows for 32-bit positions");
  std::vector<std::vector<uint32_t>> queries(I);
  size_t active = 0;
  for (size_t p = 0; p < P; ++p)
    if (d.probe[p].user % query_modulo == 0) { queries[d.probe[p].item-1].push_back(uint32_t(p)); ++active; }
  auto pool_size = [&](uint32_t i) {
    const uint64_t n = item_offsets[i+1]-item_offsets[i];
    return max_raters && n > max_raters ? uint64_t(max_raters) : n;
  };
  std::vector<uint32_t> order;
  double screen_pairs = 0;
  for (uint32_t i = 0; i < I; ++i)
    if (!queries[i].empty()) { order.push_back(i); screen_pairs += double(queries[i].size())*pool_size(i); }
  std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
    const double ca = double(queries[a].size())*pool_size(a), cb = double(queries[b].size())*pool_size(b);
    return ca != cb ? ca > cb : a < b;
  });
  std::cout << "user_knn queries=" << active << " items=" << order.size() << " screen_pairs="
            << std::setprecision(4) << screen_pairs << " screen_rank=" << dim << " candidates=" << candidates
            << " neighbors=" << k << " max_raters=" << max_raters << " alpha=" << alpha << " power=" << power
            << " shrink=" << shrink << " time_decay=" << time_decay << " threads=" << threads << std::endl;

  // Phase 4: screen, rescore and predict, one item per task.
  t0 = clock::now();
  std::vector<float> pred = base_probe, fpred = base_probe, support(P, 0), count(P, 0), fsupport(P, 0);
  // Screen tiles: kTile raters x kQB queries per matrix product.
  constexpr uint32_t kTile = 2048, kQB = 256;
  struct Scratch {
    std::vector<float> tile, qz, dot, dense;
    std::vector<UserKnnCandidate> heap;
    std::vector<uint32_t> heap_size;
    std::vector<std::pair<float,float>> scored, fscored;  // (weight, residual)
    double pearson_rows = 0, screen_seconds = 0, rescore_seconds = 0, gather_seconds = 0;
  };
  std::vector<Scratch> scratch(threads);
  for (auto &s : scratch) {
    s.tile.resize(size_t(kTile)*dim); s.dot.resize(size_t(kQB)*kTile); s.dense.assign(I, 0);
  }
  std::atomic<size_t> done{0};
  std::mutex log_mutex;
  user_knn_parallel(threads, order.size(), [&](uint32_t w, size_t task) {
    Scratch &s = scratch[w];
    const auto item_start = clock::now();
    const uint32_t i = order[task];
    const auto &qs = queries[i];
    const uint64_t begin = item_offsets[i], total = item_offsets[i+1]-begin;
    const uint64_t n = pool_size(i);
    // Deterministic subsample when capped: evenly spaced raters by user id.
    auto rater = [&](uint64_t c) { return begin + (n == total ? c : c*total/n); };
    const uint32_t mq = uint32_t(qs.size());
    const uint32_t cap = uint32_t(std::min<uint64_t>(candidates, n));
    s.qz.resize(size_t(mq)*dim);
    for (uint32_t a = 0; a < mq; ++a)
      std::copy_n(&z[size_t(d.probe[qs[a]].user)*dim], dim, &s.qz[size_t(a)*dim]);
    s.heap.resize(size_t(mq)*std::max(cap, 1U)); s.heap_size.assign(mq, 0);
    for (uint64_t t0r = 0; t0r < n; t0r += kTile) {
      const uint32_t tn = uint32_t(std::min<uint64_t>(kTile, n-t0r));
      const auto gather_start = clock::now();
      for (uint32_t r = 0; r < tn; ++r)
        std::copy_n(&z[size_t(item_user[rater(t0r+r)])*dim], dim, &s.tile[size_t(r)*dim]);
      s.gather_seconds += seconds_since(gather_start);
      for (uint32_t qa = 0; qa < mq; qa += kQB) {
        const uint32_t mb = std::min(kQB, mq-qa);
#ifdef __APPLE__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, int(mb), int(tn), int(dim), 1.F,
                    &s.qz[size_t(qa)*dim], int(dim), s.tile.data(), int(dim), 0.F, s.dot.data(), int(kTile));
#pragma clang diagnostic pop
#else
        for (uint32_t a = 0; a < mb; ++a) {
          const float *q = &s.qz[size_t(qa+a)*dim];
          for (uint32_t r = 0; r < tn; ++r) {
            const float *x = &s.tile[size_t(r)*dim]; float acc = 0;
            for (uint32_t c = 0; c < dim; ++c) acc += q[c]*x[c];
            s.dot[size_t(a)*kTile+r] = acc;
          }
        }
#endif
        for (uint32_t a = 0; a < mb; ++a) {
          UserKnnCandidate *heap = &s.heap[size_t(qa+a)*cap];
          uint32_t &size = s.heap_size[qa+a];
          const float *row = &s.dot[size_t(a)*kTile];
          uint32_t r = 0;
          for (; r < tn && size < cap; ++r) user_knn_push(heap, size, cap, row[r], uint32_t(t0r+r));
          float threshold = cap ? heap[0].score : 0;
          for (; r < tn; ++r)
            if (row[r] > threshold) {
              user_knn_push(heap, size, cap, row[r], uint32_t(t0r+r)); threshold = heap[0].score;
            }
        }
      }
    }
    const auto rescore_start = clock::now();
    s.screen_seconds += std::chrono::duration<double>(rescore_start-item_start).count();
    for (uint32_t a = 0; a < mq; ++a) {
      const uint32_t p = qs[a]; const auto &r = d.probe[p]; const uint32_t u = r.user;
      // Dense residuals of u; 0 marks "not rated" (an exact-zero residual is
      // stored as a tiny nonzero so it still counts as co-rated).
      for (uint64_t q = d.train_offsets[u]; q < d.train_offsets[u+1]; ++q)
        s.dense[d.train_items[q]-1] = residual[q] != 0 ? residual[q] : 1e-30F;
      s.scored.clear(); s.fscored.clear();
      const UserKnnCandidate *heap = &s.heap[size_t(a)*cap];
      for (uint32_t h = 0; h < s.heap_size[a]; ++h) {
        const uint64_t e = rater(heap[h].index);
        const uint32_t v = item_user[e], pos = item_pos[e];
        const float tw = time_decay > 0 ?
            std::exp(-time_decay*std::fabs(float(r.day)-float(d.train_days[pos]))) : 1.F;
        const float rv = residual[pos];
        if (heap[h].score > 0) s.fscored.emplace_back(std::pow(heap[h].score, factor_power)*tw, rv);
        float sxy = 0, sxx = 0, syy = 0, common_f = 0;
        const uint64_t vb = d.train_offsets[v], ve = d.train_offsets[v+1];
        const float *dense = s.dense.data();
        for (uint64_t q = vb; q < ve; ++q) {  // branch-free co-rated sums
          const float x = dense[d.train_items[q]-1], y = residual[q], rated = x != 0 ? 1.F : 0.F;
          sxy += x*y; sxx += x*x; syy += rated*y*y; common_f += rated;
        }
        s.pearson_rows += double(ve-vb);
        const uint32_t common = uint32_t(common_f);
        if (common < min_common || sxx <= 0 || syy <= 0) continue;
        const double rho = double(sxy)/std::sqrt(double(sxx)*double(syy))*common/(common+alpha);
        if (rho <= 0) continue;
        s.scored.emplace_back(float(std::pow(rho, double(power)))*tw, rv);
      }
      auto combine = [&](std::vector<std::pair<float,float>> &v, float sh, float &sum_w, uint32_t &used) {
        auto greater = [](const std::pair<float,float> &x, const std::pair<float,float> &y) {
          return x.first != y.first ? x.first > y.first : x.second > y.second;
        };
        if (v.size() > k) { std::nth_element(v.begin(), v.begin()+k, v.end(), greater); v.resize(k); }
        double num = 0, den = 0;
        for (const auto &e : v) { num += double(e.first)*e.second; den += e.first; }
        sum_w = float(den); used = uint32_t(v.size());
        return float(num/(den+sh));
      };
      float sw = 0, fsw = 0; uint32_t used = 0, fused = 0;
      pred[p] = base_probe[p]+combine(s.scored, shrink, sw, used);
      fpred[p] = base_probe[p]+combine(s.fscored, factor_shrink, fsw, fused);
      support[p] = sw; count[p] = float(used); fsupport[p] = fsw;
      if (!std::isfinite(pred[p]) || !std::isfinite(fpred[p])) fail("non-finite user-knn prediction");
      for (uint64_t q = d.train_offsets[u]; q < d.train_offsets[u+1]; ++q) s.dense[d.train_items[q]-1] = 0;
    }
    s.rescore_seconds += seconds_since(rescore_start);
    const size_t finished = ++done;
    if (finished % 2000 == 0 || finished == order.size()) {
      std::lock_guard<std::mutex> lock(log_mutex);
      std::cout << "items_done=" << finished << "/" << order.size()
                << " seconds=" << std::setprecision(5) << seconds_since(t0) << std::endl;
    }
  });
  const double knn_seconds = seconds_since(t0);
  double pearson_rows = 0, screen_cpu = 0, rescore_cpu = 0, gather_cpu = 0;
  for (const auto &s : scratch) {
    gather_cpu += s.gather_seconds; pearson_rows += s.pearson_rows; screen_cpu += s.screen_seconds; rescore_cpu += s.rescore_seconds;
  }
  double sse_base = 0, sse = 0, sse_f = 0, mean_count = 0;
  for (size_t p = 0; p < P; ++p) {
    if (d.probe[p].user % query_modulo) continue;
    const double y = d.probe[p].rating;
    sse_base += std::pow(y-clipped(base_probe[p]), 2);
    sse += std::pow(y-clipped(pred[p]), 2);
    sse_f += std::pow(y-clipped(fpred[p]), 2);
    mean_count += count[p];
  }
  auto save = [](const std::string &path, const std::vector<float> &values) {
    if (path.empty()) return;
    std::ofstream out(path, std::ios::binary); write_vector(out, values);
    out.close(); if (!out) fail("user-knn write failed: "+path);
  };
  save(output, pred); save(support_path, support); save(count_path, count); save(factor_path, fpred);
  save(factor_support_path, fsupport);
  const double nq = double(std::max<size_t>(active, 1));
  std::cout << std::setprecision(10) << "user_knn_result queries=" << active
            << " base_rmse=" << std::sqrt(sse_base/nq) << " probe_rmse=" << std::sqrt(sse/nq)
            << " factor_rmse=" << std::sqrt(sse_f/nq) << " mean_neighbors=" << mean_count/nq << std::endl;
  std::cout << std::setprecision(5) << "user_knn_timing knn_seconds=" << knn_seconds
            << " screen_pairs_per_second=" << screen_pairs/std::max(knn_seconds, 1e-9)
            << " screen_cpu_seconds=" << screen_cpu << " (gather " << gather_cpu << ")" << " rescore_cpu_seconds=" << rescore_cpu
            << " pearson_rows=" << pearson_rows
            << " total_seconds=" << seconds_since(start) << std::endl;
}
