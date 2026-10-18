// Ridge blending with five folds assigned by user, not by individual rating.
// Fold holdouts prevent direct fitting of blend weights on their scored rows.
// Base models still use probe for checkpoint selection: this is development CV.
uint32_t blend_fold(uint32_t user) {
  uint64_t x = uint64_t(user)+0x9e3779b97f4a7c15ULL;
  x = (x^(x>>30))*0xbf58476d1ce4e5b9ULL;
  x = (x^(x>>27))*0x94d049bb133111ebULL;
  return uint32_t((x^(x>>31))%5);
}

std::vector<double> solve_blend(std::vector<double> a, std::vector<double> b,
                              double ridge, uint64_t count) {
  const size_t n = b.size();
  // Feature zero is the intercept; remaining columns are prediction minus 3.6.
  for (size_t j = 1; j < n; ++j) a[j*n+j] += ridge*double(count);
  for (size_t i = 0; i < n; ++i) {
    size_t pivot = i;
    for (size_t j = i+1; j < n; ++j)
      if (std::abs(a[j*n+i]) > std::abs(a[pivot*n+i])) pivot = j;
    if (std::abs(a[pivot*n+i]) < 1e-12) fail("singular blend; increase ridge");
    for (size_t k = 0; k < n; ++k) std::swap(a[i*n+k], a[pivot*n+k]);
    std::swap(b[i], b[pivot]);
    const double diagonal = a[i*n+i];
    for (size_t k = i; k < n; ++k) a[i*n+k] /= diagonal;
    b[i] /= diagonal;
    for (size_t j = 0; j < n; ++j) {
      if (j == i) continue;
      const double factor = a[j*n+i];
      for (size_t k = i; k < n; ++k) a[j*n+k] -= factor*a[i*n+k];
      b[j] -= factor*b[i];
    }
  }
  return b;
}

void command_ensemble(const std::vector<std::string> &args) {
  const auto d = load_dataset(value_after(args, "--data"));
  const double ridge = double_after(args, "--ridge", .00001);
  if (ridge <= 0) fail("ridge must be positive");
  std::vector<std::string> paths;
  for (size_t i = 0; i+1 < args.size(); ++i)
    if (args[i] == "--prediction") paths.push_back(args[i+1]);
  if (paths.empty() || paths.size() > 32) fail("supply 1..32 prediction columns");
  const bool use_contexts = uint_after(args, "--contexts", 0) != 0;
  const bool quadratic = uint_after(args, "--quadratic", 0) != 0;
  const size_t rows = d.probe.size(), width = paths.size()+1;
  const size_t contexts = use_contexts ? 5 : 1;
  const size_t interaction_count = quadratic ? paths.size()*(paths.size()+1)/2 : 0;
  const size_t n = width*contexts+interaction_count;
  std::vector<float> context(rows*contexts, 1);
  if (use_contexts) {
    std::vector<uint64_t> item_count(d.header.items, 0);
    for (uint16_t i : d.train_items) ++item_count[i-1];
    std::vector<float> user_day(d.header.users, 0);
    for (uint32_t u = 0; u < d.header.users; ++u) {
      const auto first = d.implicit_offsets[u], last = d.implicit_offsets[u+1];
      double sum = 0;
      for (uint64_t i = first; i < last; ++i) sum += d.implicit_days[i];
      user_day[u] = last == first ? 0 : float(sum/(last-first));
    }
    for (size_t r = 0; r < rows; ++r) {
      const auto &row = d.probe[r];
      const auto u = row.user;
      const auto begin = d.implicit_days.begin()+d.implicit_offsets[u];
      const auto end = d.implicit_days.begin()+d.implicit_offsets[u+1];
      const auto same_day = std::equal_range(begin, end, row.day);
      context[r*contexts+1] = float(std::log1p(double(d.train_offsets[u+1]-d.train_offsets[u]))/6-1);
      context[r*contexts+2] = float(std::log1p(double(item_count[row.item-1]))/10-1);
      context[r*contexts+3] = (float(row.day)-user_day[u])/1000;
      context[r*contexts+4] = float(std::log1p(double(same_day.second-same_day.first))/4-.5);
    }
  }
  std::vector<std::vector<float>> preds(paths.size());
  std::cout << std::setprecision(10);
  for (size_t j = 0; j < paths.size(); ++j) {
    std::ifstream in(paths[j], std::ios::binary);
    read_vector(in, preds[j], rows);
    if (in.peek() != std::char_traits<char>::eof()) fail("prediction column has extra rows");
    double sse = 0;
    for (size_t r = 0; r < rows; ++r) {
      if (!std::isfinite(preds[j][r])) fail("non-finite prediction column");
      double e = d.probe[r].rating-clipped(preds[j][r]); sse += e*e;
    }
    std::cout << "column=" << paths[j] << " probe_rmse=" << std::sqrt(sse/rows) << '\n';
  }
  std::vector<std::vector<double>> gram(5, std::vector<double>(n*n)), rhs(5, std::vector<double>(n));
  std::vector<uint64_t> counts(5);
  std::vector<double> x(n);
  auto features = [&](size_t r) {
    for (size_t c = 0; c < contexts; ++c) {
      x[c*width] = context[r*contexts+c];
      for (size_t j = 1; j < width; ++j)
        x[c*width+j] = (preds[j-1][r]-3.6)*context[r*contexts+c];
    }
    if (quadratic) {
      size_t index = width*contexts;
      for (size_t a = 0; a < paths.size(); ++a)
        for (size_t b = a; b < paths.size(); ++b)
          x[index++] = (preds[a][r]-3.6)*(preds[b][r]-3.6);
    }
  };
  for (size_t r = 0; r < rows; ++r) {
    uint32_t fold = blend_fold(d.probe[r].user); ++counts[fold];
    features(r);
    for (size_t j = 0; j < n; ++j) {
      rhs[fold][j] += x[j]*(d.probe[r].rating-3.6);
      for (size_t k = 0; k < n; ++k) gram[fold][j*n+k] += x[j]*x[k];
    }
  }
  std::vector<std::vector<double>> weights;
  for (int held = 0; held <= 5; ++held) {
    std::vector<double> a(n*n), b(n); uint64_t count = 0;
    for (int fold = 0; fold < 5; ++fold) {
      if (fold == held) continue;
      count += counts[fold];
      for (size_t j = 0; j < n*n; ++j) a[j] += gram[fold][j];
      for (size_t j = 0; j < n; ++j) b[j] += rhs[fold][j];
    }
    weights.push_back(solve_blend(a, b, ridge, count));
  }
  double cv_sse = 0, fit_sse = 0;
  std::vector<double> fold_sse(5);
  std::vector<float> cv(rows), fitted(rows);
  for (size_t r = 0; r < rows; ++r) {
    const auto fold = blend_fold(d.probe[r].user);
    features(r);
    double cv_pred = 3.6, fit_pred = 3.6;
    for (size_t j = 0; j < n; ++j) {
      cv_pred += weights[fold][j]*x[j];
      fit_pred += weights[5][j]*x[j];
    }
    cv[r] = float(cv_pred); fitted[r] = float(fit_pred);
    double e = d.probe[r].rating-clipped(cv[r]);
    cv_sse += e*e; fold_sse[fold] += e*e;
    e = d.probe[r].rating-clipped(fitted[r]); fit_sse += e*e;
  }
  for (size_t fold = 0; fold < 5; ++fold)
    std::cout << "heldout_user_fold=" << fold << " count=" << counts[fold]
              << " rmse=" << std::sqrt(fold_sse[fold]/counts[fold]) << '\n';
  std::cout << "blend_cv_rmse=" << std::sqrt(cv_sse/rows)
            << " blend_fit_rmse=" << std::sqrt(fit_sse/rows) << " ridge=" << ridge
            << " centered_intercept=" << weights[5][0] << " contexts=" << contexts
            << " quadratic=" << quadratic << " features=" << n << '\n';
  for (size_t j = 1; j < width; ++j)
    std::cout << "weight=" << weights[5][j] << " column=" << paths[j-1] << '\n';
  const auto outpath = optional_value_after(args, "--predictions");
  if (!outpath.empty()) {
    std::ofstream out(outpath, std::ios::binary); write_vector(out, cv);
    std::ofstream fitout(outpath+".fit", std::ios::binary); write_vector(fitout, fitted);
  }
  const auto config = optional_value_after(args, "--weights");
  if (!config.empty()) {
    std::ofstream out(config);
    if (!out) fail("cannot save blend weights");
    out << std::setprecision(17) << "center=3.6\ncontexts=" << contexts << "\nquadratic=" << quadratic << '\n';
    for (size_t c = 0; c < contexts; ++c) {
      out << "context=" << c << " intercept=" << weights[5][c*width] << '\n';
      for (size_t j = 1; j < width; ++j)
        out << weights[5][c*width+j] << '\t' << paths[j-1] << '\n';
    }
    if (quadratic) {
      size_t index = width*contexts;
      for (size_t a = 0; a < paths.size(); ++a)
        for (size_t b = a; b < paths.size(); ++b)
          out << "interaction=" << weights[5][index++] << '\t' << paths[a] << '\t' << paths[b] << '\n';
    }
    if (!out) fail("failed writing blend weights");
  }
}
