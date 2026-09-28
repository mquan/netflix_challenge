// Dual to the item-history GP: nonlinear user features, fitted per movie.
#define main netflix_original_main
#include "netflix.cpp"
#undef main
#include <Accelerate/Accelerate.h>

static std::vector<double> solve_ridge(std::vector<double> a, std::vector<double> b, double penalty) {
  const size_t n = b.size();
  for (size_t i=0; i<n; ++i) {
    a[i*n+i] += penalty;
    for (size_t j=0; j<=i; ++j) {
      double v=a[i*n+j];
      for (size_t k=0; k<j; ++k) v-=a[i*n+k]*a[j*n+k];
      if (i==j) {
        if (!(v>0) || !std::isfinite(v)) fail("invalid ridge matrix");
        a[i*n+j]=std::sqrt(v);
      } else a[i*n+j]=v/a[j*n+j];
    }
  }
  for (size_t i=0; i<n; ++i) {
    for (size_t j=0; j<i; ++j) b[i]-=a[i*n+j]*b[j];
    b[i]/=a[i*n+i];
  }
  for (size_t i=n; i-->0;) {
    for (size_t j=i+1; j<n; ++j) b[i]-=a[j*n+i]*b[j];
    b[i]/=a[i*n+i];
  }
  return b;
}

int main(int argc, char** argv) {
  try {
    std::vector<std::string> args(argv+1,argv+argc);
    if (args.size()==1 && args[0]=="--self-test") {
      auto w=solve_ridge({1,.5,.5,1},{1,0},1);
      if (std::abs(w[0]-8./15)>1e-12 || std::abs(w[1]+2./15)>1e-12) fail("solve failed");
      std::cout << "ridge analytic check passed\n"; return 0;
    }
    const auto start=std::chrono::steady_clock::now();
    auto d=load_dataset(value_after(args,"--data"));
    auto t=build_temporal_index(d); auto f=joint_features(d,t);
    const auto m=load_joint(value_after(args,"--model"),d,t,joint_fingerprint(d));
    const auto prefix=value_after(args,"--output");
    const unsigned dim=uint_after(args,"--features",64), width=dim+1;
    const unsigned modulo=uint_after(args,"--item-modulo",1);
    const double gamma=double_after(args,"--gamma",4);
    if (!dim || dim>512 || !modulo || !std::isfinite(gamma) || gamma<=0) fail("invalid parameters");
    std::vector<float> effective, base;
    build_effective_users(d,m.factors,m.rank,effective); eval_joint(d,f,m,base);
    auto normalized=effective;
    for (uint32_t u=0; u<d.header.users; ++u) {
      double norm=0;
      for (uint32_t k=0; k<m.rank; ++k) norm+=std::pow(normalized[size_t(u)*m.rank+k],2);
      norm=std::sqrt(std::max(norm,1e-20));
      for (uint32_t k=0; k<m.rank; ++k) normalized[size_t(u)*m.rank+k]/=norm;
    }
    std::mt19937 rng(20260928);
    std::normal_distribution<float> normal(0,1);
    std::uniform_real_distribution<float> uniform(0,6.283185307179586f);
    std::vector<float> projection(size_t(m.rank)*dim), phase(dim), projected(size_t(d.header.users)*dim);
    for (auto& v:projection) v=normal(rng);
    for (auto& v:phase) v=uniform(rng);
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,d.header.users,dim,m.rank,
                1,normalized.data(),m.rank,projection.data(),dim,0,projected.data(),dim);
    normalized.clear(); normalized.shrink_to_fit();
    std::vector<std::vector<float>> feature(2,std::vector<float>(size_t(d.header.users)*width));
    for (uint32_t u=0; u<d.header.users; ++u) {
      for (unsigned k=0; k<dim; ++k) {
        const double v=projected[size_t(u)*dim+k];
        feature[0][size_t(u)*width+k]=float(v/std::sqrt(double(dim)));
        feature[1][size_t(u)*width+k]=float(std::sqrt(2./dim)*std::cos(std::sqrt(gamma)*v+phase[k]));
      }
      feature[0][size_t(u)*width+dim]=feature[1][size_t(u)*width+dim]=1;
    }
    projected.clear(); projected.shrink_to_fit();
    struct Entry { uint32_t user; float residual; };
    std::vector<std::vector<Entry>> by_item(d.header.items);
    std::vector<size_t> counts(d.header.items);
    for (const auto i:d.train_items) ++counts[i-1];
    for (uint32_t i=0; i<d.header.items; ++i) if (i%modulo==0) by_item[i].reserve(counts[i]);
    for (uint32_t u=0; u<d.header.users; ++u) {
      for (uint64_t h=d.train_offsets[u]; h<d.train_offsets[u+1]; ++h) {
        const uint32_t i=d.train_items[h]-1, s=f.train_session[h];
        if (i%modulo) continue;
        double p=joint_base(d,f,m,u,i,d.train_days[h],s);
        for (uint32_t k=0; k<m.rank; ++k)
          p+=m.factors.item_factors[size_t(i)*m.rank+k]*(effective[size_t(u)*m.rank+k]+m.drift[size_t(u)*m.rank+k]*f.dev[s]);
        for (uint32_t k=0; k<m.day_rank; ++k)
          p+=m.factors.item_factors[size_t(i)*m.rank+k]*m.day_factors[size_t(s)*m.day_rank+k];
        by_item[i].push_back({u,float(d.train_ratings[h]-p)});
      }
    }
    std::vector<std::vector<uint32_t>> queries(d.header.items);
    for (uint32_t p=0; p<d.probe.size(); ++p) queries[d.probe[p].item-1].push_back(p);
    const std::vector<double> penalties{1,10,100};
    std::vector<std::vector<float>> output(6,base);
    std::vector<float> mask(base.size(),0);
    std::cout << "features_ready dim=" << dim << " gamma=" << gamma << " item_modulo=" << modulo << std::endl;
    for (uint32_t i=0; i<d.header.items; ++i) {
      if (i%modulo || queries[i].empty() || by_item[i].empty()) continue;
      const size_t n=by_item[i].size();
      std::vector<float> x(n*width), y(n), gram(width*width), rhs(width);
      for (size_t h=0; h<n; ++h) y[h]=by_item[i][h].residual;
      for (size_t family=0; family<2; ++family) {
        for (size_t h=0; h<n; ++h)
          std::copy_n(feature[family].data()+size_t(by_item[i][h].user)*width,width,x.data()+h*width);
        cblas_sgemm(CblasRowMajor,CblasTrans,CblasNoTrans,width,width,int(n),1,x.data(),width,x.data(),width,0,gram.data(),width);
        cblas_sgemv(CblasRowMajor,CblasTrans,int(n),width,1,x.data(),width,y.data(),1,0,rhs.data(),1);
        for (size_t z=0; z<penalties.size(); ++z) {
          auto w=solve_ridge(std::vector<double>(gram.begin(),gram.end()),std::vector<double>(rhs.begin(),rhs.end()),penalties[z]);
          for (auto p:queries[i]) {
            double correction=0;
            const float* row=feature[family].data()+size_t(d.probe[p].user)*width;
            for (size_t k=0; k<width; ++k) correction+=row[k]*w[k];
            output[family*3+z][p]=float(base[p]+correction); mask[p]=1;
          }
        }
      }
      if (i%(modulo*500)==0) std::cout << "item=" << i << std::endl;
    }
    auto save=[&](std::string suffix,const std::vector<float>& values) {
      std::ofstream out(prefix+suffix,std::ios::binary); write_vector(out,values);
      out.close(); if (!out) fail("write failed");
    };
    size_t count=0; for (auto x:mask) if (x) ++count;
    if (!count) fail("no sampled probe rows");
    for (size_t z=0; z<output.size(); ++z) {
      save("."+std::to_string(z)+".f32",output[z]); double sse=0;
      for (size_t p=0;p<base.size();++p) if (mask[p]) sse+=std::pow(d.probe[p].rating-clipped(output[z][p]),2);
      std::cout << std::setprecision(10) << "variant=" << z << " sampled_rmse=" << std::sqrt(sse/count) << " count=" << count << std::endl;
    }
    save(".mask.f32",mask); save(".base.f32",base);
    std::cout << "seconds=" << std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count() << std::endl;
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
