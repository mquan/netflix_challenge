// Per-user ridge correction on frozen joint-model item factors.
// Fits only training ratings; probe labels are used for reporting only.
#define main netflix_original_main
#include "netflix.cpp"
#undef main

static std::vector<double> adaptive_solve(std::vector<double> a,
                                          std::vector<double> b,
                                          double penalty) {
  const size_t n = b.size();
  for (size_t i = 0; i < n; ++i) {
    a[i*n+i] += penalty;
    for (size_t j = 0; j <= i; ++j) {
      double v = a[i*n+j];
      for (size_t k = 0; k < j; ++k) v -= a[i*n+k]*a[j*n+k];
      if (i == j) {
        if (!(v > 0) || !std::isfinite(v)) fail("invalid adaptive ridge matrix");
        a[i*n+j] = std::sqrt(v);
      } else a[i*n+j] = v/a[j*n+j];
    }
  }
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < i; ++j) b[i] -= a[i*n+j]*b[j];
    b[i] /= a[i*n+i];
  }
  for (size_t i = n; i-- > 0;) {
    for (size_t j = i+1; j < n; ++j) b[i] -= a[j*n+i]*b[j];
    b[i] /= a[i*n+i];
  }
  return b;
}

int main(int argc, char **argv) {
  try {
    const std::vector<std::string> args(argv+1, argv+argc);
    if (args.size() == 1 && args[0] == "--self-test") {
      const auto w = adaptive_solve({1, .5, .5, 1}, {1, 0}, 1);
      if (std::abs(w[0]-8./15) > 1e-12 || std::abs(w[1]+2./15) > 1e-12)
        fail("adaptive ridge solve failed");
      std::cout << "adaptive ridge analytic check passed\n";
      return 0;
    }
    const auto start = std::chrono::steady_clock::now();
    const auto d = load_dataset(value_after(args, "--data"));
    const auto t = build_temporal_index(d);
    const auto f = joint_features(d, t);
    const auto m = load_joint(value_after(args, "--model"), d, t, joint_fingerprint(d));
    const auto prefix = value_after(args, "--output");
    const unsigned dim = uint_after(args, "--features", 24);
    const unsigned offset = uint_after(args, "--offset", 0);
    const unsigned modulo = uint_after(args, "--user-modulo", 1);
    if (!dim || dim > 64 || offset+dim > m.rank || !modulo)
      fail("invalid adaptive dimensions or user modulo");
    const double scale = std::sqrt(double(m.rank)/dim);
    const size_t width = dim+1;
    std::vector<float> effective, base;
    build_effective_users(d, m.factors, m.rank, effective);
    eval_joint(d, f, m, base);
    auto joint_prediction = [&](uint32_t u, uint32_t i, uint16_t day, uint32_t s) {
      double x = joint_base(d, f, m, u, i, day, s);
      for (uint32_t k = 0; k < m.rank; ++k)
        x += m.factors.item_factors[size_t(i)*m.rank+k]
          *(effective[size_t(u)*m.rank+k]+m.drift[size_t(u)*m.rank+k]*f.dev[s]);
      for (uint32_t k = 0; k < m.day_rank; ++k)
        x += m.factors.item_factors[size_t(i)*m.rank+k]
          *m.day_factors[size_t(s)*m.day_rank+k];
      return x;
    };
    auto feature = [&](uint32_t i, std::vector<double> &x) {
      x[0] = 1;
      for (uint32_t k = 0; k < dim; ++k)
        x[k+1] = scale*m.factors.item_factors[size_t(i)*m.rank+offset+k];
    };
    std::vector<std::vector<uint32_t>> queries(d.header.users);
    for (uint32_t p = 0; p < d.probe.size(); ++p)
      queries[d.probe[p].user].push_back(p);
    const std::vector<double> penalties{1, 10, 100};
    std::vector<std::vector<float>> output(penalties.size(), base);
    std::vector<float> mask(base.size(), 0);
    std::vector<double> sse(penalties.size()+1, 0);
    std::vector<double> x(width), gram(width*width), rhs(width);
    uint64_t count = 0;
    for (uint32_t u = 0; u < d.header.users; ++u) {
      if ((uint64_t(u)*2654435761ULL % modulo) != 0 || queries[u].empty()) continue;
      std::fill(gram.begin(), gram.end(), 0);
      std::fill(rhs.begin(), rhs.end(), 0);
      for (uint64_t h = d.train_offsets[u]; h < d.train_offsets[u+1]; ++h) {
        const uint32_t i = d.train_items[h]-1;
        const double residual = d.train_ratings[h]
          -joint_prediction(u, i, d.train_days[h], f.train_session[h]);
        feature(i, x);
        for (size_t j = 0; j < width; ++j) {
          rhs[j] += x[j]*residual;
          for (size_t k = 0; k <= j; ++k) gram[j*width+k] += x[j]*x[k];
        }
      }
      for (size_t j = 0; j < width; ++j)
        for (size_t k = j+1; k < width; ++k)
          gram[j*width+k] = gram[k*width+j];
      std::vector<std::vector<double>> weights;
      for (double penalty : penalties)
        weights.push_back(adaptive_solve(gram, rhs, penalty));
      for (const auto p : queries[u]) {
        const auto &r = d.probe[p];
        feature(r.item-1, x);
        mask[p] = 1; ++count;
        for (size_t z = 0; z < penalties.size(); ++z) {
          double correction = 0;
          for (size_t j = 0; j < width; ++j) correction += x[j]*weights[z][j];
          output[z][p] = float(base[p]+correction);
          const double e = r.rating-clipped(output[z][p]);
          sse[z] += e*e;
        }
        const double e = r.rating-clipped(base[p]);
        sse.back() += e*e;
      }
      if (u % (modulo*10000) == 0)
        std::cout << "user=" << u << " rows=" << count << std::endl;
    }
    if (!count) fail("no sampled probe rows");
    auto save = [&](const std::string &suffix, const std::vector<float> &v) {
      std::ofstream out(prefix+suffix, std::ios::binary);
      write_vector(out, v); out.close();
      if (!out) fail("adaptive output write failed");
    };
    for (size_t z = 0; z < penalties.size(); ++z) {
      save("."+std::to_string(z)+".f32", output[z]);
      std::cout << std::setprecision(10) << "variant=" << z
                << " penalty=" << penalties[z]
                << " sampled_rmse=" << std::sqrt(sse[z]/count) << std::endl;
    }
    save(".base.f32", base); save(".mask.f32", mask);
    std::cout << std::setprecision(10) << "base_sampled_rmse="
              << std::sqrt(sse.back()/count) << " count=" << count
              << " seconds=" << std::chrono::duration<double>(
                    std::chrono::steady_clock::now()-start).count() << std::endl;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n'; return 1;
  }
}
