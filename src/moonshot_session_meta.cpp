// Probe-row context from training ratings on the user's rating day.
// No probe rating enters feature construction.
#define main netflix_original_main
#include "netflix.cpp"
#undef main

int main(int argc, char **argv) {
  try {
    const std::vector<std::string> args(argv+1, argv+argc);
    const auto d = load_dataset(value_after(args, "--data"));
    const auto prefix = value_after(args, "--output");
    const auto model_path = optional_value_after(args, "--model");
    const bool residual_features = !model_path.empty();
    JointModel model;
    JointFeatures joint;
    std::vector<float> effective;
    if (residual_features) {
      const auto temporal = build_temporal_index(d);
      joint = joint_features(d, temporal);
      model = load_joint(model_path, d, temporal, joint_fingerprint(d));
      build_effective_users(d, model.factors, model.rank, effective);
    }
    auto joint_prediction = [&](uint32_t u, uint32_t i, uint16_t day, uint32_t s) {
      double pred = joint_base(d, joint, model, u, i, day, s);
      const float *q = &model.factors.item_factors[size_t(i)*model.rank];
      const float *fq = model.frequency_factors.empty() ? nullptr :
        &model.frequency_factors[(size_t(i)*kFrequencyBins+joint.freq[s])*model.rank];
      const float *p = &effective[size_t(u)*model.rank];
      const float *a = &model.drift[size_t(u)*model.rank];
      for (uint32_t k = 0; k < model.rank; ++k)
        pred += (q[k]+(fq ? fq[k] : 0))*(p[k]+a[k]*joint.dev[s]);
      for (uint32_t k = 0; k < model.day_rank; ++k)
        pred += (q[k]+(fq ? fq[k] : 0))*model.day_factors[size_t(s)*model.day_rank+k];
      return pred;
    };
    std::vector<std::vector<uint32_t>> queries(d.header.users);
    for (uint32_t p = 0; p < d.probe.size(); ++p)
      queries[d.probe[p].user].push_back(p);
    std::vector<std::vector<float>> out(residual_features ? 8 : 6,
                                         std::vector<float>(d.probe.size(), 0));
    uint64_t same_day_rows = 0, same_day_queries = 0;
    for (uint32_t u = 0; u < d.header.users; ++u) {
      if (queries[u].empty()) continue;
      const auto begin = d.train_offsets[u], end = d.train_offsets[u+1];
      double user_sum = 0, user_one = 0, user_five = 0;
      for (uint64_t h = begin; h < end; ++h) {
        const auto r = d.train_ratings[h];
        user_sum += r; user_one += r == 1; user_five += r == 5;
      }
      const double user_n = end-begin;
      const double user_mean = user_n ? user_sum/user_n : d.header.mean;
      const double user_frac_one = user_n ? user_one/user_n : 0;
      const double user_frac_five = user_n ? user_five/user_n : 0;
      for (const auto p : queries[u]) {
        const auto day = d.probe[p].day;
        const auto first = std::lower_bound(d.train_days.begin()+begin,
                                             d.train_days.begin()+end, day);
        const auto last = std::upper_bound(first, d.train_days.begin()+end, day);
        const auto n = last-first;
        double sum = 0, sum_sq = 0, one = 0, five = 0;
        double residual_sum = 0, residual_sq = 0;
        for (auto it = first; it != last; ++it) {
          const auto h = size_t(it-d.train_days.begin());
          const auto r = d.train_ratings[h];
          sum += r; sum_sq += double(r)*r; one += r == 1; five += r == 5;
          if (residual_features) {
            const double residual = r-joint_prediction(u, d.train_items[h]-1,
                                                       d.train_days[h], joint.train_session[h]);
            residual_sum += residual;
            residual_sq += residual*residual;
          }
        }
        const double avg = n ? sum/n : user_mean;
        const double var = n ? std::max(0., sum_sq/n-avg*avg) : 0;
        double nearest = 0;
        if (first == d.train_days.begin()+end)
          nearest = end > begin ? day-d.train_days[end-1] : 0;
        else if (first == d.train_days.begin()+begin)
          nearest = d.train_days[begin]-day;
        else
          nearest = std::min(double(d.train_days[size_t(first-d.train_days.begin())]-day),
                             double(day-d.train_days[size_t(first-d.train_days.begin())-1]));
        out[0][p] = float(std::log1p(double(n)));
        out[1][p] = float((sum+10*user_mean)/(n+10)-user_mean);
        out[2][p] = float((one+10*user_frac_one)/(n+10)-user_frac_one);
        out[3][p] = float((five+10*user_frac_five)/(n+10)-user_frac_five);
        out[4][p] = float(std::sqrt(var));
        out[5][p] = float(std::log1p(std::max(0., nearest)));
        if (residual_features) {
          out[6][p] = float(residual_sum/(n+10));
          out[7][p] = float(n ? std::sqrt(std::max(0., residual_sq/n
                                       -std::pow(residual_sum/n, 2))) : 0);
        }
        if (n) { ++same_day_queries; same_day_rows += n; }
      }
    }
    for (size_t j = 0; j < out.size(); ++j) {
      std::ofstream stream(prefix+"."+std::to_string(j)+".f32", std::ios::binary);
      write_vector(stream, out[j]); stream.close();
      if (!stream) fail("session feature output write failed");
    }
    std::cout << "probe_rows=" << d.probe.size()
              << " same_day_queries=" << same_day_queries
              << " same_day_training_rows=" << same_day_rows
              << " features=" << out.size() << std::endl;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n'; return 1;
  }
}
