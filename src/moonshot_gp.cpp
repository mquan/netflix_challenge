// Experimental local Gaussian-process residual inference. Separate binary;
// reuse the dataset/checkpoint readers without changing the production CLI.
#define main netflix_original_main
#include "netflix.cpp"
#undef main
#include <Accelerate/Accelerate.h>

// Cholesky, then triangular solves. K is symmetric row-major.
static std::vector<double> posterior(std::vector<double> a,
                                     std::vector<double> b, double noise) {
  const size_t n = b.size();
  for (size_t i = 0; i < n; ++i) {
    a[i*n+i] += noise;
    for (size_t j = 0; j <= i; ++j) {
      double v = a[i*n+j];
      for (size_t k = 0; k < j; ++k) v -= a[i*n+k]*a[j*n+k];
      if (i == j) {
        if (!(v > 0) || !std::isfinite(v)) fail("kernel is not positive definite");
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
    std::vector<std::string> args(argv+1, argv+argc);
    if (args.size() == 1 && args[0] == "--self-test") {
      auto x = posterior({1, .5, .5, 1}, {1, 0}, 1);
      if (std::abs(x[0]-8.0/15) > 1e-12 || std::abs(x[1]+2.0/15) > 1e-12)
        fail("posterior solve failed");
      std::cout << "posterior analytic check passed\n"; return 0;
    }
    const auto start = std::chrono::steady_clock::now();
    const auto d = load_dataset(value_after(args, "--data"));
    const auto t = build_temporal_index(d);
    const auto f = joint_features(d, t);
    const auto m = load_joint(value_after(args, "--model"), d, t, joint_fingerprint(d));
    const auto prefix = value_after(args, "--output");
    const unsigned cap = uint_after(args, "--neighbors", 32);
    const unsigned modulo = uint_after(args, "--user-modulo", 1);
    const double gamma = double_after(args, "--gamma", 4);
    const double time_scale = double_after(args, "--time-scale", 0);
    const double persistent = double_after(args, "--persistent", 0);
    const double implicit_mix = double_after(args, "--implicit-mix", 0);
    if (!cap || cap > 256 || !modulo || !std::isfinite(gamma) || !(gamma > 0)
        || !std::isfinite(time_scale) || time_scale < 0 || !std::isfinite(persistent)
        || persistent < 0 || persistent > 1 || !std::isfinite(implicit_mix)
        || implicit_mix < 0 || implicit_mix > 1)
      fail("invalid parameters");
    std::vector<float> effective, base;
    build_effective_users(d, m.factors, m.rank, effective);
    eval_joint(d, f, m, base);
    auto predict = [&](uint32_t u, uint32_t i, uint16_t day, uint32_t s) {
      double x = joint_base(d, f, m, u, i, day, s);
      for (uint32_t k = 0; k < m.rank; ++k)
        x += m.factors.item_factors[size_t(i)*m.rank+k]*(effective[size_t(u)*m.rank+k]
             + m.drift[size_t(u)*m.rank+k]*f.dev[s]);
      for (uint32_t k = 0; k < m.day_rank; ++k)
        x += m.factors.item_factors[size_t(i)*m.rank+k]*m.day_factors[size_t(s)*m.day_rank+k];
      return x;
    };
    const size_t items = d.header.items;
    auto make_kernel = [&](std::vector<float> embeddings) {
      for (size_t i = 0; i < items; ++i) {
        double norm = 0;
        for (size_t k = 0; k < m.rank; ++k) norm += std::pow(embeddings[i*m.rank+k], 2);
        norm = std::sqrt(std::max(norm, 1e-20));
        for (size_t k = 0; k < m.rank; ++k) embeddings[i*m.rank+k] /= norm;
      }
      std::vector<float> kernel(items*items);
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, int(items), int(items),
                  int(m.rank), 1, embeddings.data(), int(m.rank), embeddings.data(),
                  int(m.rank), 0, kernel.data(), int(items));
      for (auto &x : kernel) x = float(std::exp(gamma*(std::clamp(double(x), -1., 1.)-1)));
      for (size_t i = 0; i < items; ++i) kernel[i*items+i] = 1;
      return kernel;
    };
    auto kernel = make_kernel(implicit_mix == 1 ? m.factors.implicit_factors : m.factors.item_factors);
    if (implicit_mix > 0 && implicit_mix < 1) {
      const auto other = make_kernel(m.factors.implicit_factors);
      for (size_t i = 0; i < kernel.size(); ++i)
        kernel[i] = float((1-implicit_mix)*kernel[i]+implicit_mix*other[i]);
    }
    auto cov = [&](uint32_t i, uint16_t day, uint32_t j, uint16_t other_day) {
      return double(kernel[i*items+j])*(time_scale == 0 ? 1 :
             persistent + (1-persistent)*std::exp(-std::abs(int(day)-int(other_day))/time_scale));
    };
    std::cout << "kernel_ready items=" << items << " gamma=" << gamma
              << " time_scale=" << time_scale << " persistent=" << persistent
              << " implicit_mix=" << implicit_mix << " neighbors=" << cap
              << " user_modulo=" << modulo << std::endl;
    std::vector<std::vector<uint32_t>> queries(d.header.users);
    for (uint32_t p = 0; p < d.probe.size(); ++p) queries[d.probe[p].user].push_back(p);
    const std::vector<double> noises{.1, 1, 10};
    std::vector<std::vector<float>> pred(4, base);
    std::vector<float> mask(base.size(), 0);
    std::vector<float> support(base.size(), 0), strongest(base.size(), 0), effective_neighbors(base.size(), 0);
    std::vector<double> sse(5, 0);
    uint64_t count = 0;
    for (uint32_t u = 0; u < d.header.users; ++u) {
      // Deterministic user sampling, independent of labels and blend folds.
      if ((uint64_t(u)*2654435761ULL % modulo) != 0 || queries[u].empty()) continue;
      const uint64_t first = d.train_offsets[u], end = d.train_offsets[u+1];
      std::vector<double> residual(end-first);
      for (uint64_t p = first; p < end; ++p)
        residual[p-first] = d.train_ratings[p]-predict(u, d.train_items[p]-1,
                                                      d.train_days[p], f.train_session[p]);
      for (const auto p : queries[u]) {
        const auto &r = d.probe[p];
        std::vector<std::pair<double, uint64_t>> near;
        near.reserve(end-first);
        for (uint64_t h = first; h < end; ++h) {
          if (d.train_items[h] == r.item) fail("probe target present in training history");
          near.emplace_back(cov(r.item-1, r.day, d.train_items[h]-1, d.train_days[h]), h);
        }
        const size_t n = std::min(size_t(cap), near.size());
        std::partial_sort(near.begin(), near.begin()+n, near.end(), std::greater<>());
        near.resize(n);
        std::vector<double> a(n*n), y(n);
        double num = 0, den = .1;
        for (size_t i = 0; i < n; ++i) {
          const auto h = near[i].second;
          y[i] = residual[h-first]; num += near[i].first*y[i]; den += near[i].first;
          for (size_t j = 0; j < n; ++j) {
            const auto h2 = near[j].second;
            a[i*n+j] = cov(d.train_items[h]-1, d.train_days[h],
                          d.train_items[h2]-1, d.train_days[h2]);
          }
        }
        double mass = 0, mass_sq = 0;
        for (const auto &entry : near) { mass += entry.first; mass_sq += entry.first*entry.first; }
        support[p] = float(mass);
        strongest[p] = n ? float(near.front().first) : 0;
        effective_neighbors[p] = float(mass*mass/std::max(mass_sq, 1e-20));
        for (size_t z = 0; z < noises.size(); ++z) {
          const auto weights = posterior(a, y, noises[z]);
          double correction = 0;
          for (size_t j = 0; j < n; ++j) correction += near[j].first*weights[j];
          pred[z][p] = float(base[p]+correction);
        }
        pred[3][p] = float(base[p]+num/den);
        mask[p] = 1; ++count;
        for (size_t z = 0; z < 5; ++z) {
          double e = r.rating-clipped(z == 4 ? base[p] : pred[z][p]); sse[z] += e*e;
        }
      }
      if (u % (modulo*10000) == 0) std::cout << "user=" << u << " rows=" << count << std::endl;
    }
    if (!count) fail("no sampled probe rows");
    auto save = [&](const std::string &suffix, const std::vector<float> &v) {
      std::ofstream out(prefix+suffix, std::ios::binary); write_vector(out, v);
      out.close(); if (!out) fail("output write failed");
    };
    for (size_t z = 0; z < 4; ++z) {
      save("."+std::to_string(z)+".f32", pred[z]);
      std::cout << std::setprecision(10) << "variant=" << z << " sampled_rmse="
                << std::sqrt(sse[z]/count) << " count=" << count << std::endl;
    }
    save(".base.f32", base); save(".mask.f32", mask);
    save(".support.f32", support);
    save(".strongest.f32", strongest);
    save(".effective-neighbors.f32", effective_neighbors);
    std::cout << std::setprecision(10) << "base_sampled_rmse=" << std::sqrt(sse[4]/count)
              << " seconds=" << std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()
              << std::endl;
  } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
