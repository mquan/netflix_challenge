// Time-aware neighborhood regression inspired by Eq. 18 of BellKor (2009).
// The graph limits candidate pairs; the model learns both item-pair weights,
// per-user temporal decay, and the temporal baseline on training ratings.
// Static b_tilde residuals are used as explicit neighbor inputs.
struct TemporalNeighborhoodModel {
  TimeBaselineModel base;
  std::vector<float> explicit_weight, implicit_weight, user_decay;
  uint32_t epoch = 0;
};

void save_temporal_neighborhood(const fs::path &path, const TemporalNeighborhoodModel &m,
                                uint64_t data_hash, uint64_t graph_hash) {
  std::ofstream out(path.string()+".tmp", std::ios::binary);
  if (!out) fail("cannot write temporal neighborhood model");
  write_pod(out, uint64_t(0x4E46544E45494731ULL));
  write_pod(out, data_hash); write_pod(out, graph_hash); write_pod(out, m.epoch);
  const auto &b = m.base;
  write_vector(out,b.user_bias); write_vector(out,b.user_alpha);
  write_vector(out,b.item_bias); write_vector(out,b.item_bin_bias);
  write_vector(out,b.user_scale); write_vector(out,b.day_bias);
  write_vector(out,b.day_scale); write_vector(out,b.item_frequency_bias);
  write_vector(out,m.explicit_weight); write_vector(out,m.implicit_weight);
  write_vector(out,m.user_decay);
  out.close(); if (!out) fail("temporal neighborhood model write failed");
  fs::rename(path.string()+".tmp", path);
}

TemporalNeighborhoodModel load_temporal_neighborhood(const fs::path &path,
    const Dataset &d, const TemporalIndex &t, size_t edges,
    uint64_t data_hash, uint64_t graph_hash) {
  std::ifstream in(path, std::ios::binary);
  if (!in) fail("cannot open temporal neighborhood model");
  uint64_t magic=0, saved_data=0, saved_graph=0;
  TemporalNeighborhoodModel m;
  read_pod(in,magic); read_pod(in,saved_data); read_pod(in,saved_graph); read_pod(in,m.epoch);
  if (magic != 0x4E46544E45494731ULL || saved_data != data_hash || saved_graph != graph_hash)
    fail("temporal neighborhood model fingerprint mismatch");
  auto &b = m.base;
  read_vector(in,b.user_bias,d.header.users); read_vector(in,b.user_alpha,d.header.users);
  read_vector(in,b.item_bias,d.header.items);
  read_vector(in,b.item_bin_bias,size_t(d.header.items)*kTimeBins);
  read_vector(in,b.user_scale,d.header.users);
  read_vector(in,b.day_bias,t.session_days.size());
  read_vector(in,b.day_scale,t.session_days.size());
  read_vector(in,b.item_frequency_bias,size_t(d.header.items)*kFrequencyBins);
  read_vector(in,m.explicit_weight,edges); read_vector(in,m.implicit_weight,edges);
  read_vector(in,m.user_decay,d.header.users);
  if (in.peek() != std::char_traits<char>::eof()) fail("trailing temporal neighborhood data");
  return m;
}

void command_temporal_neighborhood(const std::vector<std::string> &args, bool eval_only) {
  const Dataset d = load_dataset(value_after(args,"--data"));
  const TemporalIndex t = build_temporal_index(d);
  const JointFeatures f = joint_features(d,t);
  const uint64_t data_hash = joint_fingerprint(d);
  const NeighborGraph g = load_graph(value_after(args,"--graph"),d.header.items,data_hash);
  const uint64_t graph_hash = graph_fingerprint(g);
  const fs::path model_path = value_after(args,"--model");
  const fs::path output = value_after(args,"--predictions");
  const uint32_t epochs = uint_after(args,"--iterations",25);
  const float neighbor_rate = float(double_after(args,"--neighbor-rate",.005));
  const float neighbor_reg = float(double_after(args,"--neighbor-regularization",.002));
  const std::string decay_rate_arg = optional_value_after(args,"--decay-rate");
  // double_after's default formatting uses six decimal places, which rounds
  // the paper's 1e-7 step size to zero.
  const float decay_rate = decay_rate_arg.empty() ? 1e-7F : float(double_after(args,"--decay-rate",0));
  const float decay_reg = float(double_after(args,"--decay-regularization",.01));
  const float baseline_rate = float(double_after(args,"--baseline-rate",1));
  const bool fixed_epoch = std::find(args.begin(),args.end(),"--fixed-epoch") != args.end();
  if (!epochs || neighbor_rate <= 0 || neighbor_reg < 0 || decay_rate < 0 ||
      decay_reg < 0 || baseline_rate < 0) fail("invalid temporal neighborhood settings");
  TemporalNeighborhoodModel m;
  if (eval_only) {
    m = load_temporal_neighborhood(model_path,d,t,g.neighbor.size(),data_hash,graph_hash);
  } else {
    const std::string init = optional_value_after(args,"--time-model");
    if (!init.empty()) {
      TimeBaselineHeader header{};
      m.base = load_time_baseline(init,d,t,header);
    } else {
      auto &b = m.base;
      b.user_bias.assign(d.header.users,0); b.user_alpha.assign(d.header.users,0);
      b.item_bias.assign(d.header.items,0);
      b.item_bin_bias.assign(size_t(d.header.items)*kTimeBins,0);
      b.user_scale.assign(d.header.users,1); b.day_bias.assign(t.session_days.size(),0);
      b.day_scale.assign(t.session_days.size(),0);
      b.item_frequency_bias.assign(size_t(d.header.items)*kFrequencyBins,0);
    }
    m.explicit_weight.assign(g.neighbor.size(),0);
    m.implicit_weight.assign(g.neighbor.size(),0);
    m.user_decay.assign(d.header.users,0);
  }
  std::vector<float> residual; static_residuals(d,residual);
  std::vector<uint32_t> rated(d.header.items,UINT32_MAX), known(d.header.items,UINT32_MAX);
  std::vector<uint16_t> rated_day(d.header.items), known_day(d.header.items);
  std::vector<float> rated_residual(d.header.items);
  std::vector<uint32_t> explicit_edges, implicit_edges;
  std::vector<float> explicit_input, implicit_input;
  std::vector<uint64_t> po(d.header.users+1,0);
  for (const auto &r : d.probe) ++po[r.user+1];
  std::partial_sum(po.begin(),po.end(),po.begin());
  auto cursor=po; std::vector<uint32_t> probe_order(d.probe.size());
  for (size_t p=0;p<d.probe.size();++p) probe_order[cursor[d.probe[p].user]++]=uint32_t(p);
  auto fill_user = [&](uint32_t u) {
    for (uint64_t p=d.implicit_offsets[u];p<d.implicit_offsets[u+1];++p) {
      const uint32_t i=d.implicit_items[p]-1;
      known[i]=u; known_day[i]=d.implicit_days[p];
    }
    for (uint64_t p=d.train_offsets[u];p<d.train_offsets[u+1];++p) {
      const uint32_t i=d.train_items[p]-1;
      rated[i]=u; rated_day[i]=d.train_days[p]; rated_residual[i]=residual[p];
    }
  };
  float decay_gradient=0;
  auto correction = [&](uint32_t u,uint32_t i,uint16_t day) {
    explicit_edges.clear(); implicit_edges.clear();
    explicit_input.clear(); implicit_input.clear();
    float value=0; decay_gradient=0;
    const float explicit_norm=1/std::sqrt(float(std::max<uint64_t>(1,d.train_offsets[u+1]-d.train_offsets[u])));
    const float implicit_norm=1/std::sqrt(float(std::max<uint64_t>(1,d.implicit_offsets[u+1]-d.implicit_offsets[u])));
    for (uint32_t a=0;a<g.k;++a) {
      const uint32_t edge=i*g.k+a;
      if (g.similarity[edge] <= 0) continue;
      const uint32_t other=g.neighbor[edge];
      if (rated[other]==u) {
        const uint16_t gap=uint16_t(std::abs(int(day)-int(rated_day[other])));
        const float input=rated_residual[other]*std::exp(-m.user_decay[u]*float(gap))*explicit_norm;
        explicit_edges.push_back(edge); explicit_input.push_back(input);
        value+=m.explicit_weight[edge]*input;
        decay_gradient-=m.explicit_weight[edge]*input*gap;
      }
      if (known[other]==u) {
        const uint16_t gap=uint16_t(std::abs(int(day)-int(known_day[other])));
        const float input=std::exp(-m.user_decay[u]*float(gap))*implicit_norm;
        implicit_edges.push_back(edge); implicit_input.push_back(input);
        value+=m.implicit_weight[edge]*input;
        decay_gradient-=m.implicit_weight[edge]*input*gap;
      }
    }
    return value;
  };
  auto base_prediction = [&](uint32_t u,uint32_t i,uint16_t day,uint32_t s) {
    auto &b=m.base;
    const uint32_t bin=i*kTimeBins+time_bin(d,day), freq=i*kFrequencyBins+f.freq[s];
    const float item=b.item_bias[i]+b.item_bin_bias[bin];
    return float(d.header.mean)+b.user_bias[u]+b.user_alpha[u]*f.dev[s]+b.day_bias[s]
         +item*(b.user_scale[u]+b.day_scale[s])+b.item_frequency_bias[freq];
  };
  std::vector<float> pred(d.probe.size());
  std::vector<uint32_t> users(d.header.users); std::iota(users.begin(),users.end(),0);
  std::mt19937 rng(2009);
  double best=std::numeric_limits<double>::infinity(); uint32_t stale=0;
  for (uint32_t epoch=1;epoch<=(eval_only ? 1 : epochs);++epoch) {
    const auto start=std::chrono::steady_clock::now();
    double train_sse=0;
    if (!eval_only) {
      std::shuffle(users.begin(),users.end(),rng);
      for (uint32_t u : users) {
        fill_user(u);
        for (uint64_t p=d.train_offsets[u];p<d.train_offsets[u+1];++p) {
          const uint32_t i=d.train_items[p]-1,s=f.train_session[p];
          const uint16_t day=d.train_days[p];
          auto &b=m.base;
          const uint32_t bin=i*kTimeBins+time_bin(d,day),freq=i*kFrequencyBins+f.freq[s];
          const float item=b.item_bias[i]+b.item_bin_bias[bin],scale=b.user_scale[u]+b.day_scale[s];
          const float estimate=base_prediction(u,i,day,s)+correction(u,i,day);
          const float err=float(d.train_ratings[p])-estimate;
          train_sse+=double(err)*err;
          for (size_t j=0;j<explicit_edges.size();++j) {
            float &w=m.explicit_weight[explicit_edges[j]];
            w+=neighbor_rate*(err*explicit_input[j]-neighbor_reg*w);
          }
          for (size_t j=0;j<implicit_edges.size();++j) {
            float &w=m.implicit_weight[implicit_edges[j]];
            w+=neighbor_rate*(err*implicit_input[j]-neighbor_reg*w);
          }
          m.user_decay[u]=std::clamp(m.user_decay[u]+decay_rate*(err*decay_gradient-decay_reg*m.user_decay[u]),0.0F,.1F);
          b.user_bias[u]+=baseline_rate*.00267F*(err-.0255F*b.user_bias[u]);
          b.user_alpha[u]+=baseline_rate*.00000311F*(err*f.dev[s]-3.95F*b.user_alpha[u]);
          b.day_bias[s]+=baseline_rate*.00257F*(err-.00231F*b.day_bias[s]);
          b.item_bias[i]+=baseline_rate*.000488F*(err*scale-.0255F*b.item_bias[i]);
          b.item_bin_bias[bin]+=baseline_rate*.000115F*(err*scale-.0929F*b.item_bin_bias[bin]);
          b.user_scale[u]+=baseline_rate*.00564F*(err*item-.0476F*(b.user_scale[u]-1));
          b.day_scale[s]+=baseline_rate*.00103F*(err*item-.019F*b.day_scale[s]);
          b.item_frequency_bias[freq]+=baseline_rate*.00236F*(err-1.1e-8F*b.item_frequency_bias[freq]);
        }
      }
    }
    double sse=0;
    for (uint32_t u=0;u<d.header.users;++u) {
      if (po[u]==po[u+1]) continue;
      fill_user(u);
      for (uint64_t q=po[u];q<po[u+1];++q) {
        const uint32_t p=probe_order[q]; const auto &r=d.probe[p];
        pred[p]=base_prediction(u,r.item-1,r.day,f.probe_session[p])+correction(u,r.item-1,r.day);
        if (!std::isfinite(pred[p])) fail("non-finite temporal neighborhood prediction");
        const double err=r.rating-clipped(pred[p]); sse+=err*err;
      }
    }
    const double score=std::sqrt(sse/d.probe.size());
    if (eval_only) {
      std::ofstream out(output,std::ios::binary); write_vector(out,pred);
      std::cout << std::setprecision(10) << "epoch=" << m.epoch << " probe_rmse=" << score << '\n';
      return;
    }
    m.epoch=epoch;
    if (fixed_epoch || score<best) {
      best=std::min(best,score); stale=0;
      save_temporal_neighborhood(model_path,m,data_hash,graph_hash);
      std::ofstream out(output,std::ios::binary); write_vector(out,pred);
    } else ++stale;
    const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    std::cout << std::setprecision(10) << "epoch=" << epoch << " train_rmse=" << std::sqrt(train_sse/d.header.train_count)
              << " probe_rmse=" << score << " best=" << best << " seconds=" << seconds << std::endl;
    if (!fixed_epoch && stale>=4) break;
  }
}
