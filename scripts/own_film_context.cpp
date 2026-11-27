#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

using Stat=std::array<double,7>;
static inline int category(float x, float width, float shift, int bins) {
    return std::clamp(int(std::floor((x-shift)/width)),0,bins-1);
}
extern "C" void film_context(int nitems,int ndays,const int32_t* starts,const int32_t* order,
        const int32_t* qs,const int32_t* qo,const int32_t* user,const int16_t* date,
        const int8_t* rating,const float* rank,const float* cf,const int32_t* counts,
        const double* totals,const int32_t* qu,const int16_t* qd,const float* qr,
        const float* qc,float* out) {
    constexpr int groups=12,ng=4, windows=5, width=(windows+ng)*7;
    for(int m=0;m<nitems;++m){
        if(qs[m]==qs[m+1]) continue;
        std::vector<Stat> days(ndays+1);
        Stat buckets[ng][groups]{};
        auto keys=[&](int u,float avg,float rankval,float cfval){
            return std::array<int,ng>{category(std::log1p(counts[u]),.65f,0,groups),
                category(avg,.2f,2.45f,groups),category(rankval,.28f,2,groups),category(cfval,.28f,2,groups)};
        };
        for(int j=starts[m];j<starts[m+1];++j){
            int a=order[j],u=user[a],r=rating[a];
            auto k=keys(u,(totals[u]-r+18.)/(counts[u]+4.),rank[a],cf[a]);
            Stat inc{1.,double(r),r-rank[a],r-cf[a],double(r==1),double(r<=2),double(r==5)};
            for(int v=0;v<7;++v){
                days[date[a]+1][v]+=inc[v];
                for(int g=0;g<ng;++g)buckets[g][k[g]][v]+=inc[v];
            }
        }
        for(int d=1;d<=ndays;++d)
            for(int v=0;v<7;++v)days[d][v]+=days[d-1][v];
        const int radii[windows]={0,3,14,60,5000};
        for(int j=qs[m];j<qs[m+1];++j){
            int a=qo[j],u=qu[a],d=qd[a];
            auto k=keys(u,(totals[u]+18.)/(counts[u]+5.),qr[a],qc[a]);
            auto dest=out+int64_t(a)*width;
            for(int g=0;g<windows+ng;++g){
                Stat s{};
                if(g<windows){
                    int lo=std::max(0,d-radii[g]),hi=std::min(ndays,d+1+radii[g]);
                    for(int v=0;v<7;++v)s[v]=days[hi][v]-days[lo][v];
                }else{s=buckets[g-windows][k[g-windows]];}
                dest[7*g]=std::log1p(s[0]);
                for(int v=1;v<7;++v)dest[7*g+v]=s[v]/std::max(1.,s[0]);
            }
        }
    }
}
