// Included inside netflix.cpp's anonymous namespace after joint.hpp.
//
// Bayesian timeSVD++ "flipped" (Rendle, Zhang & Koren 2019, arXiv 1905.01395,
// appendix 5.1-5.2). The paper fits a factorization machine (FM) with libFM's
// MCMC solver over the feature groups u, i, t, iu (items the user rated) and
// ii (users who rated the item). This file implements that sampler directly:
//
//   y = w0 + sum_j w_j x_j + sum_f 1/2 [ (sum_j v_jf x_j)^2 - sum_j v_jf^2 x_j^2 ]
//
// over these groups (each has a linear weight and, unless disabled, a factor
// vector; every pair of active features interacts, as in any FM):
//
//   user      one-hot user id                      (U-block)
//   user_bag  items in N(u), weight |N(u)|^-p      (U-block, implicit incl.
//                                                   probe/qualifying pairs)
//   item      one-hot item id                      (I-block)
//   item_bag  users in N(i), weight |N(i)|^-p      (I-block, the "flipped" bag)
//   day       one-hot global rating day            (paper's t)
//   item_bin  one-hot item x 30 time bins          (timeSVD++ b_i,Bin(t))
//   session   one-hot user-day                     (timeSVD++ b_u,t)
//   drift     user id with value dev_u(t)/rms      (timeSVD++ alpha_u dev)
//
// Learning is element-wise Gibbs sampling (Freudenthaler, Schmidt-Thieme &
// Rendle 2011): every parameter has a Gaussian prior N(mu_g, 1/lambda_g), with
// one (mu, lambda) per group for linear weights and per group *and factor* for
// factors; (mu, lambda) get Normal-Gamma hyperpriors and the noise precision a
// Gamma prior (all hyper-hyperparameters 1, mu0 = 0, as libFM). Because the
// model is multilinear, each parameter's conditional is Gaussian given
// sum_n h_n^2 and sum_n h_n e_n. The bags use Rendle's (2013) relational block
// trick: rows of a user share the U-block, so per-block sufficient statistics
// replace per-row work and a bag parameter costs O(|blocks containing it|).
//
// Factor parameters are stored factor-major (v[f*n + j]) because the sampler
// visits one factor at a time. Residuals are kept in double and updated
// exactly after each group; --check-residuals recomputes them from scratch.
// Prediction is the posterior mean: probe predictions averaged over sweeps
// after --burn-in. Judge runs by mean_probe_rmse; a single Gibbs sample is
// ~0.03-0.04 worse at rank 16+ (posterior predictive variance ~ K sigma^2/n).
//
// Defaults (from ablations on the train-matched holdout, see bet-b notes in
// the commit log): only user/item start from N(0, --init-std); auxiliary
// groups start at zero (--aux-init-std 0). Random auxiliary init leaves weakly
// identified groups (sessions, item-bins, day vectors) with a small sampled
// prior precision that mixes away very slowly, an overfit mode with lower
// train but worse probe error. Day, item-bin and drift default to bias-only:
// their factors added cost without a reliable gain.
namespace bayes {

enum Group : uint32_t {
  kUser, kUserBag, kItem, kItemBag, kDay, kItemBin, kSession, kDrift, kItemFreq, kGroups
};
constexpr const char *kGroupNames[kGroups] = {
    "user", "user-bag", "item", "item-bag", "day", "item-bin", "session", "drift", "item-freq"};
constexpr Group kRowGroups[] = {kDay, kItemBin, kSession, kDrift, kItemFreq};
constexpr size_t kRowGroupCount = sizeof(kRowGroups)/sizeof(kRowGroups[0]);
constexpr uint64_t kMagic = 0x4E46504241595332ULL;  // NFPBAYS2: includes item-frequency group

struct Model {
  uint32_t rank = 0, sweep = 0, kept = 0;
  std::array<uint32_t, kGroups> mode{};  // 0 off, 1 linear, 2 linear+factors
  std::array<uint64_t, kGroups> size{};
  float bag_power = .5F, drift_scale = 1;
  double w0 = 0, alpha = 1;
  std::array<std::vector<float>, kGroups> w, v;
  std::array<double, kGroups> w_mu{}, w_lambda{};
  std::array<std::vector<double>, kGroups> v_mu, v_lambda;
  std::vector<double> pred_sum;  // probe predictions summed over kept sweeps
  std::string rng_state;
};

// U-block (one per user) and I-block (one per item) caches for the current
// factor. q = sum of in-block v x (or linear sum in the linear phase), s = sum
// of (v x)^2, s0 = row count, s1/s2 = sum of Qout and Qout^2 over the block's
// rows, se/sqe = sum e and sum Qout*e, x = bag weight. Qout is the rest of the
// row's factor sum, outside this block.
struct Block {
  double q = 0, s = 0, s1 = 0, s2 = 0, se = 0, sqe = 0;
  float s0 = 0, x = 0;
};

inline float draw(double theta, double a, double b, double alpha, double mu,
                  double lambda, std::mt19937_64 &rng) {
  const double precision = alpha*a + lambda;
  const double mean = (alpha*(b + theta*a) + mu*lambda)/precision;
  return float(mean + std::normal_distribution<double>(0, 1)(rng)/std::sqrt(precision));
}

// Accumulates wall time of a scope into a profile slot (--profile).
struct Clock {
  double &acc; std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
  explicit Clock(double &a) : acc(a) {}
  ~Clock() { acc += std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count(); }
};
enum Profile { kPass, kRowDraw, kBlockSum, kUserBagDraw, kItemBagDraw, kProbe, kProfiles };
constexpr const char *kProfileNames[kProfiles] = {"row_pass", "row_draw", "block_sum", "user_bag_draw", "item_bag_draw", "probe"};

template<class F> void parallel(uint32_t threads, F fn) {
  if (threads == 1) { fn(0U); return; }
  std::vector<std::thread> pool;
  for (uint32_t t = 0; t < threads; ++t) pool.emplace_back([&, t]() { fn(t); });
  for (auto &th : pool) th.join();
}

// Split [0, n) into `parts` ranges with roughly equal cumulative weight.
std::vector<uint32_t> balanced_cuts(const std::vector<uint64_t> &offsets, uint32_t parts) {
  const uint32_t n = uint32_t(offsets.size()-1);
  std::vector<uint32_t> cuts(parts+1, n); cuts[0] = 0;
  const uint64_t total = offsets[n]-offsets[0];
  uint32_t k = 0;
  for (uint32_t t = 1; t < parts; ++t) {
    const uint64_t goal = offsets[0]+total*t/parts;
    while (k < n && offsets[k] < goal) ++k;
    cuts[t] = k;
  }
  return cuts;
}

struct Sampler {
  const Dataset &d; const TemporalIndex &t; const JointFeatures &f;
  Model &m; uint32_t threads; std::mt19937_64 rng;
  uint32_t users = 0, items = 0;
  float drift_inv = 1;
  std::vector<uint8_t> row_bin, probe_bin;
  std::vector<uint64_t> item_offsets, user_weight;
  std::vector<uint32_t> item_users, user_cut, item_cut, bag_item_cut, bag_user_cut;
  std::vector<Block> ub, ib;
  std::vector<double> uq0, us0, iq0, is0, iq1, is1, uq1;
  bool pending = false; int pending_f = -1;  // deferred item-block residual update
  std::vector<double> E;            // train residuals y - yhat
  std::vector<float> QR;            // row-level factor sum for the current f
  std::array<std::vector<double>, kGroups> A, B;
  std::array<std::vector<float>, kGroups> delta;
  struct Local { std::array<std::vector<double>, kGroups> a, b; std::vector<double> item; double sse = 0; };
  std::vector<Local> local;
  std::vector<double> probe_pred;
  std::array<double, kProfiles> prof{};

  enum Kind { kNone, kRow, kUserBlock, kItemBlock };

  Sampler(const Dataset &d_, const TemporalIndex &t_, const JointFeatures &f_, Model &m_,
          uint32_t threads_) : d(d_), t(t_), f(f_), m(m_), threads(threads_) {
    users = d.header.users; items = d.header.items;
    row_bin.resize(d.header.train_count);
    for (uint64_t p = 0; p < d.header.train_count; ++p) row_bin[p] = uint8_t(time_bin(d, d.train_days[p]));
    for (const auto &r : d.probe) probe_bin.push_back(uint8_t(time_bin(d, r.day)));
    drift_inv = 1/m.drift_scale;
    // Inverted implicit index: users who rated (or have a probe/qualifying
    // pair for) each item, i.e. the flipped bag N(i).
    item_offsets.assign(size_t(items)+1, 0);
    for (uint64_t j = 0; j < d.header.implicit_count; ++j) ++item_offsets[d.implicit_items[j]];
    std::partial_sum(item_offsets.begin(), item_offsets.end(), item_offsets.begin());
    item_users.resize(d.header.implicit_count);
    {
      std::vector<uint64_t> fill(item_offsets.begin(), item_offsets.end()-1);
      for (uint32_t u = 0; u < users; ++u)
        for (uint64_t j = d.implicit_offsets[u]; j < d.implicit_offsets[u+1]; ++j)
          item_users[fill[d.implicit_items[j]-1]++] = u;
    }
    ub.resize(users); ib.resize(items);
    for (uint32_t u = 0; u < users; ++u) {
      const uint64_t n = d.implicit_offsets[u+1]-d.implicit_offsets[u];
      ub[u].s0 = float(d.train_offsets[u+1]-d.train_offsets[u]);
      ub[u].x = n ? float(std::pow(double(n), -double(m.bag_power))) : 0;
    }
    for (uint64_t p = 0; p < d.header.train_count; ++p) ib[d.train_items[p]-1].s0 += 1;
    for (uint32_t i = 0; i < items; ++i) {
      const uint64_t n = item_offsets[i+1]-item_offsets[i];
      ib[i].x = n ? float(std::pow(double(n), -double(m.bag_power))) : 0;
    }
    user_weight.assign(size_t(users)+1, 0);
    for (uint32_t u = 0; u < users; ++u)
      user_weight[u+1] = user_weight[u]+(d.train_offsets[u+1]-d.train_offsets[u])
                         +(d.implicit_offsets[u+1]-d.implicit_offsets[u])+1;
    user_cut = balanced_cuts(user_weight, threads);
    bag_user_cut = balanced_cuts(d.implicit_offsets, threads);
    bag_item_cut = balanced_cuts(item_offsets, threads);
    std::vector<uint64_t> unit(size_t(items)+1); std::iota(unit.begin(), unit.end(), 0);
    for (uint32_t i = 0; i < items; ++i) unit[i+1] = unit[i]+1+item_offsets[i+1]-item_offsets[i];
    item_cut = balanced_cuts(unit, threads);
    uq0.resize(users); us0.resize(users); iq0.resize(items); is0.resize(items);
    uq1.resize(users); iq1.resize(items); is1.resize(items);
    E.resize(d.header.train_count); QR.assign(d.header.train_count, 0);
    local.resize(threads);
    for (Group g : kRowGroups) if (m.mode[g]) {
      delta[g].assign(m.size[g], 0);
      if (g == kSession || g == kDrift) { A[g].resize(m.size[g]); B[g].resize(m.size[g]); }
      else {
        A[g].resize(m.size[g]); B[g].resize(m.size[g]);
        for (auto &l : local) { l.a[g].resize(m.size[g]); l.b[g].resize(m.size[g]); }
      }
    }
    for (auto &l : local) l.item.resize(size_t(items)*4);
    probe_pred.resize(d.probe.size());
  }

  bool factors(Group g) const { return m.mode[g] == 2; }
  float *V(Group g, uint32_t fac) { return &m.v[g][size_t(fac)*m.size[g]]; }

  // Index and value of a row-level feature for train row p of user u.
  void train_feature(Group g, uint32_t u, uint64_t p, uint64_t &j, float &x) const {
    x = 1;
    switch (g) {
      case kDay: j = std::min<uint64_t>(m.size[kDay]-1, d.train_days[p]-std::min(d.train_days[p], d.header.min_day)); break;
      case kItemBin: j = uint64_t(d.train_items[p]-1)*kTimeBins+row_bin[p]; break;
      case kSession: j = f.train_session[p]; break;
      case kItemFreq: j = uint64_t(d.train_items[p]-1)*kFrequencyBins+f.freq[f.train_session[p]]; break;
      default: j = u; x = f.dev[f.train_session[p]]*drift_inv; break;
    }
  }
  void probe_feature(Group g, size_t n, uint64_t &j, float &x) const {
    const auto &r = d.probe[n]; x = 1;
    switch (g) {
      case kDay: j = std::min<uint64_t>(m.size[kDay]-1, r.day-std::min(r.day, d.header.min_day)); break;
      case kItemBin: j = uint64_t(r.item-1)*kTimeBins+probe_bin[n]; break;
      case kSession: j = f.probe_session[n]; break;
      case kItemFreq: j = uint64_t(r.item-1)*kFrequencyBins+f.freq[f.probe_session[n]]; break;
      default: j = r.user; x = f.dev[f.probe_session[n]]*drift_inv; break;
    }
  }

  // One parallel pass over train rows (partitioned by user). It optionally
  // (1) subtracts a scalar shift from residuals (w0 updates), (2) applies the
  // pending change of the previously sampled group to residuals, (3) recomputes
  // QR for factor tf, and (4) accumulates sufficient statistics for the next
  // group to sample. af/tf < 0 means the linear phase.
  void pass(double shift, int af, Kind ak, Group ag, int tf, Kind tk, Group tg, bool compute_qr) {
    Clock clock(prof[kPass]);
    // Instantiate per (apply, target) kind so the row loop has no kind branches.
    auto go = [&](auto a, auto t) { pass_rows<decltype(a)::value, decltype(t)::value>(shift, af, ag, tf, tg, compute_qr); };
    auto with_target = [&](auto a) {
      switch (tk) {
        case kNone: go(a, std::integral_constant<int, kNone>()); break;
        case kRow: go(a, std::integral_constant<int, kRow>()); break;
        case kUserBlock: go(a, std::integral_constant<int, kUserBlock>()); break;
        default: go(a, std::integral_constant<int, kItemBlock>()); break;
      }
    };
    switch (ak) {
      case kNone: with_target(std::integral_constant<int, kNone>()); break;
      case kRow: with_target(std::integral_constant<int, kRow>()); break;
      case kUserBlock: with_target(std::integral_constant<int, kUserBlock>()); break;
      default: with_target(std::integral_constant<int, kItemBlock>()); break;
    }
  }
  template<int AK, int TK>
  void pass_rows(double shift, int af, Group ag, int tf, Group tg, bool compute_qr) {
    constexpr Kind ak = Kind(AK), tk = Kind(TK);
    const bool row_target_local = tk == kRow && tg != kSession && tg != kDrift;
    const bool use_qr = (ak != kNone && af >= 0) || tf >= 0;
    const bool write_qr = use_qr || compute_qr;
    parallel(threads, [&](uint32_t th) {
      auto &l = local[th];
      double *la = nullptr, *lb = nullptr;
      if (row_target_local) {
        la = l.a[tg].data(); lb = l.b[tg].data();
        std::fill(l.a[tg].begin(), l.a[tg].end(), 0); std::fill(l.b[tg].begin(), l.b[tg].end(), 0);
      } else if (tk == kRow) {
        // Session and drift members belong to one user: zero this thread's share.
        la = A[tg].data(); lb = B[tg].data();
        const uint64_t lo = tg == kDrift ? user_cut[th] : t.session_offsets[user_cut[th]];
        const uint64_t hi = tg == kDrift ? user_cut[th+1] : t.session_offsets[user_cut[th+1]];
        std::fill(la+lo, la+hi, 0); std::fill(lb+lo, lb+hi, 0);
      }
      if (tk == kItemBlock) std::fill(l.item.begin(), l.item.end(), 0);
      const float *av = (ak == kRow && af >= 0) ? V(ag, uint32_t(af)) : nullptr;
      const float *tv = (tk == kRow && tf >= 0) ? V(tg, uint32_t(tf)) : nullptr;
      const float *dl = ak == kRow ? delta[ag].data() : nullptr;
      std::array<const float *, kRowGroupCount> qv{};
      if (compute_qr)
        for (size_t k = 0; k < kRowGroupCount; ++k)
          qv[k] = factors(kRowGroups[k]) ? V(kRowGroups[k], uint32_t(tf)) : nullptr;
      for (uint32_t u = user_cut[th]; u < user_cut[th+1]; ++u) {
        double s1 = 0, s2 = 0, se = 0, sqe = 0;
        const Block &bu = ub[u];
        for (uint64_t p = d.train_offsets[u]; p < d.train_offsets[u+1]; ++p) {
          const uint32_t i = d.train_items[p]-1;
          const Block &bi = ib[i];
          double e = E[p]-shift;
          float qr = use_qr ? QR[p] : 0;
          uint64_t j; float x;
          if (ak == kRow) {
            train_feature(ag, u, p, j, x);
            const float change = dl[j];
            if (change != 0) {
              if (af < 0) e -= double(change)*x;
              else {
                const double old = double(av[j])-change;
                e -= double(change)*x*(qr+bu.q+bi.q-old*x);
                qr += change*x;
              }
            }
          } else if (ak == kUserBlock) {
            if (af < 0) e -= bu.q-uq0[u];
            else {
              const double qo = uq0[u], qout = double(qr)+bi.q;
              e -= .5*(bu.q*bu.q-qo*qo)+qout*(bu.q-qo)-.5*(bu.s-us0[u]);
            }
          } else if (ak == kItemBlock) {
            // Reads the saved end-of-phase caches (iq1, is1, uq1) because this
            // apply is deferred into the next phase, after blocks are recomputed.
            if (af < 0) e -= iq1[i]-iq0[i];
            else {
              const double qo = iq0[i], qn = iq1[i], qout = double(qr)+uq1[u];
              e -= .5*(qn*qn-qo*qo)+qout*(qn-qo)-.5*(is1[i]-is0[i]);
            }
          }
          if (compute_qr) {
            qr = 0;
            for (size_t k = 0; k < kRowGroupCount; ++k) if (qv[k]) {
              train_feature(kRowGroups[k], u, p, j, x); qr += qv[k][j]*x;
            }
          }
          if (tk == kRow) {
            train_feature(tg, u, p, j, x);
            const double h = tf < 0 ? x : x*(qr+bu.q+bi.q-double(tv[j])*x);
            la[j] += h*h; lb[j] += h*e;
          } else if (tk == kUserBlock) {
            const double qout = tf < 0 ? 0 : double(qr)+bi.q;
            s1 += qout; s2 += qout*qout; se += e; sqe += qout*e;
          } else if (tk == kItemBlock) {
            const double qout = tf < 0 ? 0 : double(qr)+bu.q;
            double *it = &l.item[size_t(i)*4];
            it[0] += qout; it[1] += qout*qout; it[2] += e; it[3] += qout*e;
          }
          E[p] = e;
          if (write_qr) QR[p] = qr;
        }
        if (tk == kUserBlock) { ub[u].s1 = s1; ub[u].s2 = s2; ub[u].se = se; ub[u].sqe = sqe; }
      }
    });
    if (row_target_local) {
      for (uint64_t j = 0; j < m.size[tg]; ++j) {
        double a = 0, b = 0;
        for (const auto &l : local) { a += l.a[tg][j]; b += l.b[tg][j]; }
        A[tg][j] = a; B[tg][j] = b;
      }
    }
    if (tk == kItemBlock) {
      for (uint32_t i = 0; i < items; ++i) {
        double s[4] = {0, 0, 0, 0};
        for (const auto &l : local) for (int k = 0; k < 4; ++k) s[k] += l.item[size_t(i)*4+k];
        ib[i].s1 = s[0]; ib[i].s2 = s[1]; ib[i].se = s[2]; ib[i].sqe = s[3];
      }
    }
  }

  // Row-level group g: members touch disjoint rows, so all are conditionally
  // independent and can be drawn together from the accumulated statistics.
  void sample_rows(Group g, int fac) {
    Clock clock(prof[kRowDraw]);
    float *theta = fac < 0 ? m.w[g].data() : V(g, uint32_t(fac));
    const double mu = fac < 0 ? m.w_mu[g] : m.v_mu[g][size_t(fac)];
    const double lambda = fac < 0 ? m.w_lambda[g] : m.v_lambda[g][size_t(fac)];
    for (uint64_t j = 0; j < m.size[g]; ++j) {
      const float old = theta[j], next = draw(old, A[g][j], B[g][j], m.alpha, mu, lambda, rng);
      theta[j] = next; delta[g][j] = next-old;
    }
  }

  // Recompute block caches (q, s) for factor fac, or linear sums for fac < 0.
  void compute_user_blocks(int fac) {
    Clock clock(prof[kBlockSum]);
    const bool bag = fac < 0 ? m.mode[kUserBag] >= 1 : factors(kUserBag);
    const float *self = fac < 0 ? m.w[kUser].data() : V(kUser, uint32_t(fac));
    const float *bagv = !bag ? nullptr : fac < 0 ? m.w[kUserBag].data() : V(kUserBag, uint32_t(fac));
    parallel(threads, [&](uint32_t th) {
      for (uint32_t u = user_cut[th]; u < user_cut[th+1]; ++u) {
        double q = self[u], s = fac < 0 ? 0 : double(self[u])*self[u];
        if (bagv) {
          double sum = 0, sq = 0;
          for (uint64_t j = d.implicit_offsets[u]; j < d.implicit_offsets[u+1]; ++j) {
            const double y = bagv[d.implicit_items[j]-1]; sum += y; sq += y*y;
          }
          const double x = ub[u].x; q += x*sum; s += x*x*sq;
        }
        ub[u].q = q; ub[u].s = s;
      }
    });
  }
  void compute_item_blocks(int fac) {
    Clock clock(prof[kBlockSum]);
    const bool bag = fac < 0 ? m.mode[kItemBag] >= 1 : factors(kItemBag);
    const float *self = fac < 0 ? m.w[kItem].data() : V(kItem, uint32_t(fac));
    const float *bagv = !bag ? nullptr : fac < 0 ? m.w[kItemBag].data() : V(kItemBag, uint32_t(fac));
    parallel(threads, [&](uint32_t th) {
      for (uint32_t i = item_cut[th]; i < item_cut[th+1]; ++i) {
        double q = self[i], s = fac < 0 ? 0 : double(self[i])*self[i];
        if (bagv) {
          double sum = 0, sq = 0;
          for (uint64_t j = item_offsets[i]; j < item_offsets[i+1]; ++j) {
            const double y = bagv[item_users[j]]; sum += y; sq += y*y;
          }
          const double x = ib[i].x; q += x*sum; s += x*x*sq;
        }
        ib[i].q = q; ib[i].s = s;
      }
    });
  }

  // Sufficient statistics of one in-block parameter (current value theta,
  // feature value x) contributed by a block, and the exact cache update after
  // it changes by `change`.
  static void block_stats(const Block &b, double theta, double x, bool linear, double &a, double &bb) {
    if (linear) { a += x*x*b.s0; bb += x*b.se; return; }
    const double qin = b.q-theta*x;
    a += x*x*(b.s0*qin*qin+2*qin*b.s1+b.s2);
    bb += x*(qin*b.se+b.sqe);
  }
  static void block_update(Block &b, double theta, double next, double x, bool linear) {
    const double change = next-theta;
    if (linear) { b.se -= change*x*b.s0; b.q += change*x; return; }
    const double qin = b.q-theta*x;
    b.se -= change*x*(qin*b.s0+b.s1);
    b.sqe -= change*x*(qin*b.s1+b.s2);
    b.q += change*x;
    b.s += x*x*(next*next-theta*theta);
  }

  // Sample one block family: the one-hot owner parameters (independent across
  // blocks) and then the bag parameters, each shared by many blocks. With one
  // thread the bag is an exact sequential Gibbs sweep; with more threads the
  // bag is split Hogwild-style (block statistics may race) and the block
  // caches are recomputed exactly afterwards, so residuals stay exact.
  template<class ForBlocks>
  void sample_block_family(std::vector<Block> &blk, Group self, Group bag, int fac,
                           const std::vector<uint32_t> &self_cut, const std::vector<uint32_t> &bag_cut,
                           uint64_t bag_count, ForBlocks for_blocks) {
    const bool linear = fac < 0;
    float *sv = linear ? m.w[self].data() : V(self, uint32_t(fac));
    const double smu = linear ? m.w_mu[self] : m.v_mu[self][size_t(fac)];
    const double slam = linear ? m.w_lambda[self] : m.v_lambda[self][size_t(fac)];
    std::vector<uint64_t> seeds(threads);
    for (auto &s : seeds) s = rng();
    parallel(threads, [&](uint32_t th) {
      std::mt19937_64 r(seeds[th]);
      for (uint32_t b = self_cut[th]; b < self_cut[th+1]; ++b) {
        double a = 0, bb = 0; const double old = sv[b];
        block_stats(blk[b], old, 1, linear, a, bb);
        const float next = draw(old, a, bb, m.alpha, smu, slam, r);
        block_update(blk[b], old, next, 1, linear); sv[b] = next;
      }
    });
    const bool bag_on = linear ? m.mode[bag] >= 1 : factors(bag);
    if (!bag_on) return;
    float *bv = linear ? m.w[bag].data() : V(bag, uint32_t(fac));
    const double bmu = linear ? m.w_mu[bag] : m.v_mu[bag][size_t(fac)];
    const double blam = linear ? m.w_lambda[bag] : m.v_lambda[bag][size_t(fac)];
    for (auto &s : seeds) s = rng();
    parallel(threads, [&](uint32_t th) {
      std::mt19937_64 r(seeds[th]);
      const uint64_t lo = threads == 1 ? 0 : bag_cut[th], hi = threads == 1 ? bag_count : bag_cut[th+1];
      for (uint64_t j = lo; j < hi; ++j) {
        double a = 0, bb = 0; const double old = bv[j];
        for_blocks(j, [&](uint32_t b) { block_stats(blk[b], old, blk[b].x, linear, a, bb); });
        const float next = draw(old, a, bb, m.alpha, bmu, blam, r);
        if (double(next) != old)
          for_blocks(j, [&](uint32_t b) { block_update(blk[b], old, next, blk[b].x, linear); });
        bv[j] = next;
      }
    });
  }
  void sample_user_blocks(int fac) {
    Clock clock(prof[kUserBagDraw]);
    sample_block_family(ub, kUser, kUserBag, fac, user_cut, bag_item_cut, items,
        [&](uint64_t j, auto fn) { for (uint64_t k = item_offsets[j]; k < item_offsets[j+1]; ++k) fn(item_users[k]); });
    if (threads > 1) compute_user_blocks(fac);
  }
  void sample_item_blocks(int fac) {
    Clock clock(prof[kItemBagDraw]);
    sample_block_family(ib, kItem, kItemBag, fac, item_cut, bag_user_cut, users,
        [&](uint64_t v, auto fn) {
          for (uint64_t k = d.implicit_offsets[v]; k < d.implicit_offsets[v+1]; ++k) fn(uint32_t(d.implicit_items[k]-1));
        });
    if (threads > 1) compute_item_blocks(fac);
  }

  // Normal-Gamma hyperprior draws (libFM: alpha0 = beta0 = gamma0 = 1, mu0 = 0).
  void draw_hyper(const float *values, uint64_t n, double &mu, double &lambda) {
    double ss = 0, sum = 0;
    for (uint64_t j = 0; j < n; ++j) { const double y = values[j]; sum += y; ss += (y-mu)*(y-mu); }
    const double shape = (1.0+double(n)+1)/2, rate = (1.0+mu*mu+ss)/2;
    lambda = std::gamma_distribution<double>(shape, 1/rate)(rng);
    const double mean = sum/(double(n)+1);
    mu = mean+std::normal_distribution<double>(0, 1)(rng)/std::sqrt((double(n)+1)*lambda);
  }

  double sum_squared_residuals() {
    parallel(threads, [&](uint32_t th) {
      double s = 0;
      for (uint64_t p = d.train_offsets[user_cut[th]]; p < d.train_offsets[user_cut[th+1]]; ++p) s += E[p]*E[p];
      local[th].sse = s;
    });
    double s = 0; for (const auto &l : local) s += l.sse;
    return s;
  }
  double sum_residuals() {
    parallel(threads, [&](uint32_t th) {
      double s = 0;
      for (uint64_t p = d.train_offsets[user_cut[th]]; p < d.train_offsets[user_cut[th+1]]; ++p) s += E[p];
      local[th].sse = s;
    });
    double s = 0; for (const auto &l : local) s += l.sse;
    return s;
  }

  // Accumulate the probe contribution of the current block caches. In the
  // linear phase this sets the prediction; afterwards it adds factor fac.
  void probe_contribution(int fac) {
    Clock clock(prof[kProbe]);
    std::array<const float *, kRowGroupCount> rv{};
    for (size_t k = 0; k < kRowGroupCount; ++k) {
      const Group g = kRowGroups[k];
      rv[k] = fac < 0 ? (m.mode[g] ? m.w[g].data() : nullptr) : (factors(g) ? V(g, uint32_t(fac)) : nullptr);
    }
    for (size_t n = 0; n < d.probe.size(); ++n) {
      const auto &r = d.probe[n];
      double qr = 0, sr = 0;
      for (size_t k = 0; k < kRowGroupCount; ++k) if (rv[k]) {
        uint64_t j; float x; probe_feature(kRowGroups[k], n, j, x);
        const double y = double(rv[k][j])*x; qr += y; sr += y*y;
      }
      const Block &bu = ub[r.user], &bi = ib[r.item-1];
      if (fac < 0) probe_pred[n] = m.w0+qr+bu.q+bi.q;
      else {
        const double q = qr+bu.q+bi.q;
        probe_pred[n] += .5*(q*q-(sr+bu.s+bi.s));
      }
    }
  }

  // Chain the group draws of one phase (linear or factor fac): each pass
  // applies the previous group's change and gathers statistics for the next.
  void phase(int fac, double shift) {
    const bool linear = fac < 0;
    compute_user_blocks(fac); compute_item_blocks(fac);
    // The previous phase's item-block update rides along with the first pass.
    Kind ak = pending ? kItemBlock : kNone; Group ag = kItem;
    int af = pending ? pending_f : fac;
    bool first = true;
    for (Group g : kRowGroups) {
      if (linear ? !m.mode[g] : !factors(g)) continue;
      pass(first ? shift : 0, af, ak, ag, fac, kRow, g, first && !linear);
      first = false;
      sample_rows(g, fac); ak = kRow; ag = g; af = fac;
    }
    pass(first ? shift : 0, af, ak, ag, fac, kUserBlock, kUser, first && !linear);
    for (uint32_t u = 0; u < users; ++u) { uq0[u] = ub[u].q; us0[u] = ub[u].s; }
    sample_user_blocks(fac);
    pass(0, fac, kUserBlock, kUser, fac, kItemBlock, kItem, false);
    for (uint32_t i = 0; i < items; ++i) { iq0[i] = ib[i].q; is0[i] = ib[i].s; }
    sample_item_blocks(fac);
    for (uint32_t i = 0; i < items; ++i) { iq1[i] = ib[i].q; is1[i] = ib[i].s; }
    for (uint32_t u = 0; u < users; ++u) uq1[u] = ub[u].q;
    pending = true; pending_f = fac;
    probe_contribution(fac);
  }
  void flush() {
    if (pending) pass(0, pending_f, kItemBlock, kItem, -1, kNone, kUser, false);
    pending = false;
  }

  // Exact residuals from the current parameters (resume and self-check).
  void recompute_residuals(std::vector<double> &out) {
    out.resize(d.header.train_count);
    for (int fac = -1; fac < int(m.rank); ++fac) {
      compute_user_blocks(fac); compute_item_blocks(fac);
      std::array<const float *, kRowGroupCount> rv{};
      for (size_t k = 0; k < kRowGroupCount; ++k) {
        const Group g = kRowGroups[k];
        rv[k] = fac < 0 ? (m.mode[g] ? m.w[g].data() : nullptr) : (factors(g) ? V(g, uint32_t(fac)) : nullptr);
      }
      parallel(threads, [&](uint32_t th) {
        for (uint32_t u = user_cut[th]; u < user_cut[th+1]; ++u)
          for (uint64_t p = d.train_offsets[u]; p < d.train_offsets[u+1]; ++p) {
            double qr = 0, sr = 0;
            for (size_t k = 0; k < kRowGroupCount; ++k) if (rv[k]) {
              uint64_t j; float x; train_feature(kRowGroups[k], u, p, j, x);
              const double y = double(rv[k][j])*x; qr += y; sr += y*y;
            }
            const Block &bu = ub[u], &bi = ib[d.train_items[p]-1];
            if (fac < 0) out[p] = double(d.train_ratings[p])-(m.w0+qr+bu.q+bi.q);
            else { const double q = qr+bu.q+bi.q; out[p] -= .5*(q*q-(sr+bu.s+bi.s)); }
          }
      });
    }
  }

  // One full Gibbs sweep. Returns the train RMSE after the sweep.
  double sweep() {
    const double n = double(d.header.train_count);
    const double sse = sum_squared_residuals();
    m.alpha = std::gamma_distribution<double>((1+n)/2, 2/(1+sse))(rng);
    for (uint32_t g = 0; g < kGroups; ++g) {
      if (!m.mode[g]) continue;
      draw_hyper(m.w[g].data(), m.size[g], m.w_mu[g], m.w_lambda[g]);
      if (m.mode[g] == 2)
        for (uint32_t k = 0; k < m.rank; ++k)
          draw_hyper(V(Group(g), k), m.size[g], m.v_mu[g][k], m.v_lambda[g][k]);
    }
    const double old = m.w0;
    m.w0 = draw(old, n, sum_residuals(), m.alpha, 0, 0, rng);
    phase(-1, m.w0-old);
    for (uint32_t k = 0; k < m.rank; ++k) phase(int(k), 0);
    flush();
    return std::sqrt(sum_squared_residuals()/n);
  }
};

void save(const fs::path &path, const Model &m, uint64_t fingerprint) {
  if (!path.parent_path().empty()) fs::create_directories(path.parent_path());
  std::ofstream out(path.string()+".tmp", std::ios::binary);
  if (!out) fail("cannot write bayes model");
  write_pod(out, kMagic); write_pod(out, fingerprint);
  write_pod(out, m.rank); write_pod(out, m.sweep); write_pod(out, m.kept);
  for (uint32_t g = 0; g < kGroups; ++g) { write_pod(out, m.mode[g]); write_pod(out, m.size[g]); }
  write_pod(out, m.bag_power); write_pod(out, m.drift_scale);
  write_pod(out, m.w0); write_pod(out, m.alpha);
  for (uint32_t g = 0; g < kGroups; ++g) {
    write_pod(out, m.w_mu[g]); write_pod(out, m.w_lambda[g]);
    for (const auto *a : {&m.w[g], &m.v[g]}) { write_pod(out, uint64_t(a->size())); write_vector(out, *a); }
    for (const auto *a : {&m.v_mu[g], &m.v_lambda[g]}) { write_pod(out, uint64_t(a->size())); write_vector(out, *a); }
  }
  write_pod(out, uint64_t(m.pred_sum.size())); write_vector(out, m.pred_sum);
  write_pod(out, uint64_t(m.rng_state.size())); out.write(m.rng_state.data(), std::streamsize(m.rng_state.size()));
  out.close(); if (!out) fail("bayes model write failed");
  fs::rename(path.string()+".tmp", path);
}

Model load(const fs::path &path, uint64_t fingerprint) {
  std::ifstream in(path, std::ios::binary);
  if (!in) fail("cannot open bayes model");
  uint64_t magic = 0, hash = 0; Model m;
  read_pod(in, magic); read_pod(in, hash);
  if (magic != kMagic || hash != fingerprint) fail("bayes model format or dataset fingerprint mismatch");
  read_pod(in, m.rank); read_pod(in, m.sweep); read_pod(in, m.kept);
  for (uint32_t g = 0; g < kGroups; ++g) { read_pod(in, m.mode[g]); read_pod(in, m.size[g]); }
  read_pod(in, m.bag_power); read_pod(in, m.drift_scale);
  read_pod(in, m.w0); read_pod(in, m.alpha);
  auto vec = [&](auto &a) { uint64_t n = 0; read_pod(in, n); if (n > (uint64_t(1) << 36)) fail("bad bayes vector"); read_vector(in, a, n); };
  for (uint32_t g = 0; g < kGroups; ++g) {
    read_pod(in, m.w_mu[g]); read_pod(in, m.w_lambda[g]);
    vec(m.w[g]); vec(m.v[g]); vec(m.v_mu[g]); vec(m.v_lambda[g]);
  }
  vec(m.pred_sum);
  uint64_t n = 0; read_pod(in, n); if (n > 1 << 20) fail("bad bayes rng state");
  m.rng_state.resize(n); in.read(m.rng_state.data(), std::streamsize(n));
  if (!in || in.peek() != std::char_traits<char>::eof()) fail("truncated or trailing bayes model data");
  return m;
}

void write_predictions(const std::string &path, const Model &m) {
  if (path.empty()) return;
  std::vector<float> out(m.pred_sum.size());
  for (size_t n = 0; n < out.size(); ++n) out[n] = float(m.pred_sum[n]/std::max(1U, m.kept));
  if (!fs::path(path).parent_path().empty()) fs::create_directories(fs::path(path).parent_path());
  std::ofstream file(path, std::ios::binary);
  write_vector(file, out);
  if (!file) fail("cannot write bayes predictions");
}

double probe_rmse(const Dataset &d, const std::vector<double> &pred, double scale) {
  double sse = 0;
  for (size_t n = 0; n < d.probe.size(); ++n) {
    const double err = d.probe[n].rating-clipped(pred[n]*scale); sse += err*err;
  }
  return std::sqrt(sse/std::max<size_t>(1, d.probe.size()));
}

void command(const std::vector<std::string> &args) {
  const fs::path model_path = value_after(args, "--model");
  const std::string pred_path = optional_value_after(args, "--predictions");
  const Dataset d = load_dataset(value_after(args, "--data"));
  if (d.train_items.empty()) fail("empty training set");
  const auto t = build_temporal_index(d);
  const auto feats = joint_features(d, t);
  const uint64_t fingerprint = joint_fingerprint(d);
  const uint32_t threads = std::max(1U, uint_after(args, "--threads", 1));
  const uint32_t total = uint_after(args, "--samples", 128);
  const uint32_t burn_in = uint_after(args, "--burn-in", 8);
  const uint32_t snapshot_every = uint_after(args, "--snapshot-every", 0);
  const uint32_t checkpoint_every = uint_after(args, "--checkpoint-every", 16);
  const bool check = std::find(args.begin(), args.end(), "--check-residuals") != args.end();
  const bool profile = std::find(args.begin(), args.end(), "--profile") != args.end();
  const std::string init = optional_value_after(args, "--init-model");
  Model m;
  if (!init.empty()) {
    m = load(init, fingerprint);
  } else {
    m.rank = uint_after(args, "--factors", 64);
    if (!m.rank || m.rank > 1024) fail("invalid --factors");
    m.mode[kUser] = m.mode[kItem] = 2;
    m.mode[kUserBag] = uint_after(args, "--user-bag", 2);
    m.mode[kItemBag] = uint_after(args, "--item-bag", 2);
    m.mode[kDay] = uint_after(args, "--day", 1);
    m.mode[kItemBin] = uint_after(args, "--item-bin", 1);
    m.mode[kSession] = uint_after(args, "--session", 1);
    m.mode[kDrift] = uint_after(args, "--drift", 1);
    m.mode[kItemFreq] = uint_after(args, "--item-freq", 0);
    for (uint32_t g = 0; g < kGroups; ++g) if (m.mode[g] > 2) fail(std::string("invalid mode for ")+kGroupNames[g]);
    m.bag_power = float(double_after(args, "--bag-power", .5));
    if (m.bag_power < 0 || m.bag_power > 1) fail("invalid --bag-power");
    const float init_std = float(double_after(args, "--init-std", .1));
    if (!(init_std >= 0)) fail("invalid --init-std");
    const float aux_std = float(double_after(args, "--aux-init-std", 0.0));
    if (!(aux_std >= 0)) fail("invalid --aux-init-std");
    m.size = {d.header.users, d.header.items, d.header.items, d.header.users,
              uint64_t(d.header.max_day-d.header.min_day)+1, uint64_t(d.header.items)*kTimeBins,
              t.session_days.size(), d.header.users, uint64_t(d.header.items)*kFrequencyBins};
    double dev2 = 0;
    for (uint64_t p = 0; p < d.header.train_count; ++p) { const double v = feats.dev[feats.train_session[p]]; dev2 += v*v; }
    m.drift_scale = float(std::sqrt(dev2/double(d.header.train_count)));
    if (!(m.drift_scale > 0)) m.drift_scale = 1;
    std::mt19937_64 init_rng(uint_after(args, "--seed", 2026));
    std::normal_distribution<float> normal(0, 1);
    for (uint32_t g = 0; g < kGroups; ++g) {
      if (!m.mode[g]) continue;
      m.w[g].assign(m.size[g], 0); m.w_lambda[g] = 1;
      if (m.mode[g] == 2) {
        m.v[g].resize(m.size[g]*m.rank);
        const float scale = (g == kUser || g == kItem) ? init_std : aux_std;
        for (float &x : m.v[g]) x = scale*normal(init_rng);
        m.v_mu[g].assign(m.rank, 0); m.v_lambda[g].assign(m.rank, 1);
      }
    }
    m.w0 = d.header.mean;
    m.pred_sum.assign(d.probe.size(), 0);
    std::ostringstream state; state << init_rng; m.rng_state = state.str();
  }
  if (m.pred_sum.size() != d.probe.size()) fail("bayes model probe size mismatch");
  if (m.size[kSession] != t.session_days.size() || m.size[kUser] != d.header.users)
    fail("bayes model dimensions differ from dataset");
  std::cout << "bayes factors=" << m.rank << " threads=" << threads << " samples=" << total
            << " burn_in=" << burn_in << " bag_power=" << m.bag_power
            << " drift_scale=" << m.drift_scale << " fingerprint=" << fingerprint << " modes=";
  for (uint32_t g = 0; g < kGroups; ++g) std::cout << (g ? "," : "") << kGroupNames[g] << ':' << m.mode[g];
  std::cout << " resume_sweep=" << m.sweep << std::endl;
  Sampler s(d, t, feats, m, threads);
  { std::istringstream state(m.rng_state); state >> s.rng; if (!state) fail("bad bayes rng state"); }
  s.recompute_residuals(s.E);
  auto snapshot_path = [&](uint32_t sweep) {
    std::string base = pred_path, ext;
    if (base.size() > 4 && base.compare(base.size()-4, 4, ".f32") == 0) { ext = ".f32"; base.resize(base.size()-4); }
    std::ostringstream name; name << base << ".s" << std::setw(4) << std::setfill('0') << sweep << ext;
    return name.str();
  };
  auto checkpoint = [&]() {
    std::ostringstream state; state << s.rng; m.rng_state = state.str();
    save(model_path, m, fingerprint);
  };
  for (uint32_t sweep = m.sweep+1; sweep <= total; ++sweep) {
    const auto start = std::chrono::steady_clock::now();
    const double train = s.sweep();
    m.sweep = sweep;
    for (double x : s.probe_pred) if (!std::isfinite(x)) fail("non-finite bayes prediction");
    if (sweep > burn_in) {
      for (size_t n = 0; n < s.probe_pred.size(); ++n) m.pred_sum[n] += s.probe_pred[n];
      ++m.kept;
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    std::cout << std::setprecision(10) << "sweep=" << sweep << " train_rmse=" << train
              << " noise_sd=" << 1/std::sqrt(m.alpha) << " w0=" << m.w0
              << " sample_probe_rmse=" << probe_rmse(d, s.probe_pred, 1)
              << " kept=" << m.kept;
    if (m.kept) std::cout << " mean_probe_rmse=" << probe_rmse(d, m.pred_sum, 1.0/m.kept);
    std::cout << " seconds=" << seconds;
    if (check) {
      std::vector<double> exact; s.recompute_residuals(exact);
      double worst = 0;
      for (size_t p = 0; p < exact.size(); ++p) worst = std::max(worst, std::abs(exact[p]-s.E[p]));
      std::cout << " residual_drift=" << worst;
    }
    if (profile) {
      for (uint32_t k = 0; k < kProfiles; ++k) std::cout << ' ' << kProfileNames[k] << '=' << s.prof[k];
      s.prof.fill(0);
    }
    std::cout << std::endl;
    if (snapshot_every && sweep % snapshot_every == 0 && m.kept && !pred_path.empty())
      write_predictions(snapshot_path(sweep), m);
    if ((checkpoint_every && sweep % checkpoint_every == 0) || sweep == total) checkpoint();
  }
  if (m.sweep >= total && total > 0 && !fs::exists(model_path)) checkpoint();
  if (!pred_path.empty() && !m.kept) fail("no samples kept after --burn-in");
  write_predictions(pred_path, m);
  if (m.kept) std::cout << std::setprecision(10) << "final probe_rmse=" << probe_rmse(d, m.pred_sum, 1.0/m.kept)
                        << " kept=" << m.kept << " sweep=" << m.sweep << '\n';
}

}  // namespace bayes
