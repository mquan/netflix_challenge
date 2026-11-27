#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

constexpr int K=28,DIM=48;
using Vec=std::array<float,K>;
static inline float cosine(const float* v,int a,int b) {
    const float* x=v+int64_t(a)*DIM;const float* y=v+int64_t(b)*DIM;
    float dot=0; for(int k=0;k<DIM;++k)dot+=x[k]*y[k];
    return dot;
}
extern "C" void user_gp(int first,int last,int nitems,const int32_t* qu,const int16_t* qi,
        const int16_t* qd,const int32_t* off,const int16_t* ti,const int16_t* td,
        const int8_t* tr,const float* re,const float* ce,const float* movie_bias,
        const float* user_bias,const int16_t* sim_rank,const int16_t* sim_session,
        const float* rv,const float* sv,float* output) {
    std::vector<std::pair<float,int>> order;
    for(int j=first;j<last;++j){
        int u=qu[j],m=qi[j],d=qd[j],lo=off[u],hi=off[u+1];
        int count=std::min(K,hi-lo);
        if(count==0)continue;
        order.resize(hi-lo);
        auto rrow=sim_rank+int64_t(m)*nitems;
        auto srow=sim_session+int64_t(m)*nitems;
        for(int a=lo;a<hi;++a){
            float sr=rrow[ti[a]]/16000.f,ss=srow[ti[a]]/16000.f;
            float near=1.f/(1.f+std::abs(d-td[a])/12.f);
            order[a-lo]={std::abs(.48f*sr+.22f*ss+near*(.14f+.16f*sr)),a};
        }
        if(count<int(order.size()))std::nth_element(order.begin(),order.begin()+count,order.end(),std::greater<>());
        std::array<Vec,K> stat{},local{};
        Vec ks{},kt{};
        for(int a=0;a<count;++a){
            int ca=order[a].second,ma=ti[ca];
            float sr=rrow[ma]/16000.f,ss=srow[ma]/16000.f;
            float near=1.f/(1.f+std::abs(d-td[ca])/12.f);
            ks[a]=.64f*sr+.36f*ss;
            kt[a]=.48f*sr+.22f*ss+near*(.14f+.16f*sr);
            for(int b=0;b<=a;++b){
                int cb=order[b].second,mb=ti[cb];
                float r=cosine(rv,ma,mb),s=cosine(sv,ma,mb);
                float prox=1.f/(1.f+std::abs(int(td[ca])-td[cb])/12.f);
                stat[a][b]=.64f*r+.36f*s;
                local[a][b]=.48f*r+.22f*s+prox*(.14f+.16f*r);
            }
        }
        float *out=output+int64_t(j)*12;
        int col=0;
        for(int geometry=0;geometry<2;++geometry){
            for(float noise:{.20f,1.25f}){
                auto g=geometry?local:stat;
                Vec w=geometry?kt:ks;
                for(int a=0;a<count;++a){
                    for(int b=0;b<a;++b){
                        float v=g[a][b];
                        for(int k=0;k<b;++k)v-=g[a][k]*g[b][k];
                        g[a][b]=v/g[b][b];
                    }
                    float diag=g[a][a]+noise;
                    for(int k=0;k<a;++k)diag-=g[a][k]*g[a][k];
                    g[a][a]=std::sqrt(std::max(.001f,diag));
                    for(int k=0;k<a;++k)w[a]-=g[a][k]*w[k];
                    w[a]/=g[a][a];
                }
                for(int a=count-1;a>=0;--a){
                    for(int k=a+1;k<count;++k)w[a]-=g[k][a]*w[k];
                    w[a]/=g[a][a];
                }
                float er=0,ec=0,r=0;
                for(int a=0;a<count;++a){
                    int ca=order[a].second;
                    er+=w[a]*re[ca];ec+=w[a]*ce[ca];
                    r+=w[a]*(tr[ca]-user_bias[u]-movie_bias[ti[ca]]);
                }
                out[col++]=er;out[col++]=ec;out[col++]=r+user_bias[u]+movie_bias[m];
            }
        }
    }
}
