// Sparse mixture of local residual factorizations, trained against a frozen mean.
#define main netflix_original_main
#include "netflix.cpp"
#undef main

struct Route { uint32_t group[2]{}; float weight[2]{1, 0}; };

static std::vector<Route> item_routes(const JointModel &m, uint32_t items,
                                     uint32_t groups, uint32_t seed) {
  std::vector<Route> routes(items);
  if (groups == 1) return routes;
  auto x = m.factors.item_factors;
  auto normalize = [&](float *v) {
    double sum = 0; for (uint32_t k = 0; k < m.rank; ++k) sum += v[k]*v[k];
    const double scale = 1/std::sqrt(std::max(sum, 1e-20));
    for (uint32_t k = 0; k < m.rank; ++k) v[k] *= scale;
  };
  for (uint32_t i = 0; i < items; ++i) normalize(&x[size_t(i)*m.rank]);
  std::mt19937 rng(seed);
  std::vector<float> centers(size_t(groups)*m.rank);
  for (uint32_t g = 0; g < groups; ++g) {
    const auto i = rng()%items;
    std::copy_n(&x[size_t(i)*m.rank], m.rank, &centers[size_t(g)*m.rank]);
  }
  auto score = [&](uint32_t i, uint32_t g) {
    double dot = 0;
    for (uint32_t k = 0; k < m.rank; ++k) dot += x[size_t(i)*m.rank+k]*centers[size_t(g)*m.rank+k];
    return dot;
  };
  for (unsigned iteration = 0; iteration < 12; ++iteration) {
    std::vector<float> sums(centers.size(), 0);
    std::vector<uint32_t> counts(groups, 0);
    for (uint32_t i = 0; i < items; ++i) {
      uint32_t best = 0; double value = -2;
      for (uint32_t g = 0; g < groups; ++g) {
        const auto s = score(i, g); if (s > value) { value = s; best = g; }
      }
      ++counts[best];
      for (uint32_t k = 0; k < m.rank; ++k) sums[size_t(best)*m.rank+k] += x[size_t(i)*m.rank+k];
    }
    for (uint32_t g = 0; g < groups; ++g) if (counts[g]) {
      normalize(&sums[size_t(g)*m.rank]);
      std::copy_n(&sums[size_t(g)*m.rank], m.rank, &centers[size_t(g)*m.rank]);
    }
  }
  for (uint32_t i = 0; i < items; ++i) {
    std::vector<std::pair<double,uint32_t>> scores;
    for (uint32_t g = 0; g < groups; ++g) scores.emplace_back(score(i,g),g);
    std::partial_sort(scores.begin(), scores.begin()+2, scores.end(), std::greater<>());
    const float w = float(1/(1+std::exp(8*(scores[1].first-scores[0].first))));
    routes[i] = {{scores[0].second,scores[1].second},{w,1-w}};
  }
  return routes;
}

int main(int argc, char **argv) {
  try {
    const std::vector<std::string> args(argv+1,argv+argc);
    const auto d = load_dataset(value_after(args,"--data"));
    const auto t = build_temporal_index(d);
    const auto f = joint_features(d,t);
    const auto m = load_joint(value_after(args,"--model"),d,t,joint_fingerprint(d));
    const auto output = value_after(args,"--output");
    const uint32_t groups = uint_after(args,"--groups",8), rank = uint_after(args,"--rank",8);
    const uint32_t epochs = uint_after(args,"--epochs",6), seed = uint_after(args,"--seed",20260928);
    double lr = double_after(args,"--learning-rate",.03);
    const double reg = double_after(args,"--regularization",.05);
    if (!groups || groups > d.header.items || groups > 64 || !rank || rank > 128 || !epochs
        || !std::isfinite(lr) || lr <= 0 || !std::isfinite(reg) || reg < 0) fail("invalid parameters");
    const unsigned active = groups == 1 ? 1 : 2;
    const auto routes = item_routes(m,d.header.items,groups,seed);
    std::vector<float> effective, base;
    build_effective_users(d,m.factors,m.rank,effective); eval_joint(d,f,m,base);
    std::vector<float> residual(d.header.train_count);
    for (uint32_t u = 0; u < d.header.users; ++u)
      for (uint64_t p = d.train_offsets[u]; p < d.train_offsets[u+1]; ++p) {
        const uint32_t i = d.train_items[p]-1, s = f.train_session[p];
        double x = joint_base(d,f,m,u,i,d.train_days[p],s);
        for (uint32_t k = 0; k < m.rank; ++k)
          x += m.factors.item_factors[size_t(i)*m.rank+k]*(effective[size_t(u)*m.rank+k]
                +m.drift[size_t(u)*m.rank+k]*f.dev[s]);
        for (uint32_t k = 0; k < m.day_rank; ++k)
          x += m.factors.item_factors[size_t(i)*m.rank+k]*m.day_factors[size_t(s)*m.day_rank+k];
        residual[p] = float(d.train_ratings[p]-x);
      }
    std::vector<float> uf(size_t(d.header.users)*groups*rank), itf(size_t(d.header.items)*active*rank);
    std::vector<float> ub(size_t(d.header.users)*groups,0), ib(size_t(d.header.items)*active,0);
    std::mt19937 rng(seed); std::normal_distribution<float> normal(0,.05F);
    for (auto &x : uf) x = normal(rng);
    for (auto &x : itf) x = normal(rng);
    std::vector<uint32_t> users(d.header.users);
    std::iota(users.begin(),users.end(),0);
    auto correction = [&](uint32_t u,uint32_t i) {
      double x = 0;
      for (unsigned a = 0; a < active; ++a) {
        const auto g = routes[i].group[a];
        double v = ub[size_t(u)*groups+g]+ib[size_t(i)*active+a];
        for (uint32_t k = 0; k < rank; ++k)
          v += uf[(size_t(u)*groups+g)*rank+k]*itf[(size_t(i)*active+a)*rank+k];
        x += routes[i].weight[a]*v;
      }
      return x;
    };
    std::cout << "local groups=" << groups << " rank=" << rank << " active=" << active
              << " regularization=" << reg << " seed=" << seed << std::endl;
    for (uint32_t epoch = 1; epoch <= epochs; ++epoch) {
      const auto start = std::chrono::steady_clock::now();
      std::shuffle(users.begin(),users.end(),rng);
      double train_sse = 0;
      std::vector<uint64_t> order;
      for (const auto u : users) {
        order.resize(d.train_offsets[u+1]-d.train_offsets[u]);
        std::iota(order.begin(),order.end(),d.train_offsets[u]);
        std::shuffle(order.begin(),order.end(),rng);
        for (const auto p : order) {
          const uint32_t i = d.train_items[p]-1;
          const double error = residual[p]-correction(u,i); train_sse += error*error;
          for (unsigned a = 0; a < active; ++a) {
            const auto g = routes[i].group[a]; const double step = lr*routes[i].weight[a];
            auto &bu = ub[size_t(u)*groups+g]; auto &bi = ib[size_t(i)*active+a];
            bu += float(step*(error-.2*bu)); bi += float(step*(error-.2*bi));
            float *pv = &uf[(size_t(u)*groups+g)*rank], *qv = &itf[(size_t(i)*active+a)*rank];
            for (uint32_t k = 0; k < rank; ++k) {
              const auto old = pv[k];
              pv[k] += float(step*(error*qv[k]-reg*pv[k]));
              qv[k] += float(step*(error*old-reg*qv[k]));
            }
          }
        }
      }
      std::vector<float> pred(base); double sse = 0;
      for (size_t p = 0; p < d.probe.size(); ++p) {
        const auto &r = d.probe[p]; pred[p] += float(correction(r.user,r.item-1));
        if (!std::isfinite(pred[p])) fail("non-finite prediction");
        const double e = r.rating-clipped(pred[p]); sse += e*e;
      }
      const auto path = output+".epoch"+std::to_string(epoch)+".f32";
      std::ofstream out(path,std::ios::binary); write_vector(out,pred); out.close();
      if (!out) fail("prediction write failed");
      std::cout << std::setprecision(10) << "epoch=" << epoch << " train_rmse="
                << std::sqrt(train_sse/d.header.train_count) << " probe_rmse=" << std::sqrt(sse/d.probe.size())
                << " seconds=" << std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count() << std::endl;
      lr *= .9;
    }
  } catch(const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
