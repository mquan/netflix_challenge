#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

constexpr int W=44,L=12;
extern "C" int choose_masks(int n,int seed,const int* off,const int16_t* day,const int* qs,const int* qo,
        const int16_t* qd,int* starts,int* selected) {
    std::mt19937 gen(seed);std::uniform_real_distribution<float> unif(0.00001f,.99999f);
    std::vector<std::pair<float,int>> scores;int used=0;
    for(int u=0;u<n;++u) {
        starts[u]=used;int nr=off[u+1]-off[u],nq=qs[u+1]-qs[u];
        if(nr<3 || nq==0)continue;
        int want=std::min({32,std::max(1,nr/5),2+nq});
        scores.clear();
        for(int j=off[u];j<off[u+1];++j) {
            int gap=4096;for(int k=qs[u];k<qs[u+1];++k)gap=std::min(gap,std::abs(day[j]-qd[qo[k]]));
            float mass=.10f+.90f/(1.f+std::pow(gap/10.f,.8f));
            float priority=std::log(mass)-std::log(-std::log(unif(gen)));
            scores.emplace_back(priority,j);
        }
        std::nth_element(scores.begin(),scores.begin()+want,scores.end(),std::greater<>());
        for(int k=0;k<want;++k)selected[used++]=scores[k].second;
    }
    starts[n]=used;return used;
}
extern "C" void exclude_peers(int n,const int* ix,const int* u,const int16_t* item,const int16_t* day,
        const int8_t* r,const float* re,const float* ce,const float* coords,const int* starts,const int* mask,
        float* values) {
    for(int j=0;j<n;++j) {
        int target=ix[j],user=u[target];
        std::array<std::array<float,W>,4> sum={};
        for(int a=starts[user];a<starts[user+1];++a) {
            int row=mask[a];if(row==target)continue;
            int dt=std::abs(day[row]-day[target]),m=item[row];
            for(int t=dt==0?0:dt<=7?1:dt<=45?2:3;t<4;++t){
                auto& z=sum[t];z[0]++;z[r[row]]++;z[6]+=re[row];z[7]+=ce[row];
                for(int k=0;k<L;++k){
                    float v=coords[m*L+k];z[8+k]+=v;z[8+L+k]+=(r[row]-3.5f)*v;z[8+2*L+k]+=ce[row]*v;
                }
            }
        }
        for(int t=0;t<4;++t){
            float* v=values+(int64_t(j)*4+t)*W;
            float nc=std::round(std::expm1(5*v[0])),remain=std::max(0.f,nc-sum[t][0]);
            for(int k=1;k<W;++k)v[k]=(v[k]*(4+nc)-sum[t][k])/(4+remain);
            v[0]=std::log1p(remain)*.2f;
        }
    }
}
