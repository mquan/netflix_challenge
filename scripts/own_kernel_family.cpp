#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

constexpr int KMAX=72, GEOM=3, NF=7;
struct Rel {float a,b,c,t,same,p;};
static inline float kernel(const Rel& r,int family) {
    if(family==0)return .48f*r.a+.12f*r.b+.13f*r.t+.12f*r.t*r.a+.15f*r.same;
    if(family==1)return .14f*r.a+.26f*r.c+.15f*r.b+.20f*r.t*r.c+.15f*r.same+.10f*r.t;
    return .25f*r.a+.10f*r.a*r.a+.18f*r.c+.13f*r.b+.16f*r.p*r.t+.10f*r.same*r.p+.08f*r.t;
}
extern "C" void kernel_predict(int first,int last,int nitems,const int32_t* qu,
        const int16_t* qi,const int16_t* qd,const int32_t* off,const int16_t* ti,
        const int16_t* td,const int8_t* tr,const float* re,const float* ce,
        const float* movie,const float* user,const int16_t* ga,const int16_t* gb,
        const int16_t* gc,const float* trainp,const float* queryp,float* output) {
    std::vector<std::pair<float,int>> cand;
    std::array<float,4096> tlookup{};
    for(int i=0;i<4096;++i)tlookup[i]=.7f/(1.f+i/12.f)+.3f/(1.f+i/100.f);
    auto rel=[&](int a,int b,int dt,float dp){
        int64_t k=int64_t(a)*nitems+b;
        return Rel{ga[k]/16000.f,gb[k]/16000.f,gc[k]/16000.f,
            tlookup[dt],float(dt==0),1.f/(1.f+std::abs(dp)/.5f)};
    };
    for(int j=first;j<last;++j) {
        int u=qu[j],m=qi[j],lo=off[u],hi=off[u+1],d=qd[j],n=std::min(KMAX,hi-lo);
        if(!n)continue;
        float qp=queryp[j];
        cand.resize(hi-lo);
        for(int a=lo;a<hi;++a) {
            auto r=rel(m,ti[a],std::abs(d-td[a]),qp-trainp[a]);
            cand[a-lo]={std::max(std::abs(kernel(r,0)),std::abs(kernel(r,1))),a};
        }
        if(n<int(cand.size()))std::nth_element(cand.begin(),cand.begin()+n,cand.end(),std::greater<>());
        float cov[GEOM][KMAX*KMAX],target[GEOM][KMAX],mat[KMAX*KMAX],w[KMAX];
        for(int a=0;a<n;++a) {
            int ai=cand[a].second;
            auto query=rel(m,ti[ai],std::abs(d-td[ai]),qp-trainp[ai]);
            for(int g=0;g<GEOM;++g)target[g][a]=kernel(query,g);
            for(int b=0;b<=a;++b){
                int bi=cand[b].second;
                auto v=rel(ti[ai],ti[bi],std::abs(int(td[ai])-td[bi]),trainp[ai]-trainp[bi]);
                for(int g=0;g<GEOM;++g)cov[g][a*n+b]=kernel(v,g);
            }
        }
        int col=0;
        for(int g=0;g<GEOM;++g)for(float noise:{.18f,1.3f}){
            for(int a=0;a<n;++a){
                std::copy_n(cov[g]+a*n,a+1,mat+a*n);
                mat[a*n+a]+=noise;
            }
            std::copy_n(target[g],n,w);
            char part='U';
            int info=0,one=1;
            spotrf_(&part,&n,mat,&n,&info);
            if(info!=0){output[int64_t(j)*GEOM*2*NF]=NAN;break;}
            spotrs_(&part,&n,&one,mat,&n,w,&n,&info);
            float f[NF]{};
            for(int a=0;a<n;++a) {
                int k=cand[a].second;
                float x=tr[k]-trainp[k],wa=w[a];
                f[0]+=wa*re[k];f[1]+=wa*ce[k];f[2]+=wa*(tr[k]-user[u]-movie[ti[k]]);
                f[3]+=wa*std::clamp(x,-.75f,.75f);
                f[4]+=wa*x*std::min(2.f,std::abs(x));
                f[5]+=wa*target[g][a];f[6]+=std::abs(wa);
            }
            f[2]+=user[u]+movie[m];
            auto dest=output+int64_t(j)*GEOM*2*NF;
            for(float v:f)dest[col++]=v;
        }
    }
}
