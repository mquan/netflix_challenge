// Export frozen joint-model residuals and predictions for a separate persona fit.
#define main netflix_original_main
#include "netflix.cpp"
#undef main

int main(int argc, char **argv) {
  try {
    const std::vector<std::string> args(argv + 1, argv + argc);
    const Dataset d = load_dataset(value_after(args, "--data"));
    const TemporalIndex t = build_temporal_index(d);
    const JointFeatures f = joint_features(d, t);
    const JointModel m = load_joint(value_after(args, "--model"), d, t, joint_fingerprint(d));
    const fs::path prefix = value_after(args, "--output");
    if (!prefix.parent_path().empty()) fs::create_directories(prefix.parent_path());
    std::vector<float> effective;
    build_effective_users(d, m.factors, m.rank, effective);
    std::ofstream train(prefix.string() + ".train-residual.f32", std::ios::binary);
    if (!train) fail("cannot create training residual output");
    for (uint32_t u = 0; u < d.header.users; ++u) {
      const float *p = &effective[size_t(u) * m.rank];
      const float *a = &m.drift[size_t(u) * m.rank];
      for (uint64_t row = d.train_offsets[u]; row < d.train_offsets[u + 1]; ++row) {
        const uint32_t i = d.train_items[row] - 1;
        const uint32_t s = f.train_session[row];
        const float *q = &m.factors.item_factors[size_t(i) * m.rank];
        const float *fq = m.frequency_factors.empty() ? nullptr :
            &m.frequency_factors[(size_t(i) * kFrequencyBins + f.freq[s]) * m.rank];
        double prediction = joint_base(d, f, m, u, i, d.train_days[row], s);
        for (uint32_t k = 0; k < m.rank; ++k)
          prediction += (q[k] + (fq ? fq[k] : 0)) * (p[k] + a[k] * f.dev[s]);
        for (uint32_t k = 0; k < m.day_rank; ++k)
          prediction += (q[k] + (fq ? fq[k] : 0)) *
              m.day_factors[size_t(s) * m.day_rank + k];
        const float residual = float(d.train_ratings[row] - prediction);
        train.write(reinterpret_cast<const char *>(&residual), sizeof(residual));
      }
    }
    train.close();
    if (!train) fail("training residual write failed");
    std::vector<float> probe;
    const double rmse = eval_joint(d, f, m, probe);
    std::ofstream out(prefix.string() + ".probe.f32", std::ios::binary);
    write_vector(out, probe);
    out.close();
    if (!out) fail("probe prediction write failed");
    std::cout << "train_rows=" << d.header.train_count << " probe_rows=" << probe.size()
              << " rank=" << m.rank << " base_probe_rmse=" << rmse << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << std::endl;
    return 1;
  }
}
