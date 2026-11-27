#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>
extern "C" void slate_masks(int n,int width,int seed,const int* off,const int16_t* day,const int8_t* selected,
                            const int* qoff,const int* qo,const int16_t* qd,int* masks){
    std::mt19937 rng(seed);std::uniform_real_distribution<float> unif(.00001f,.99999f);
    std::vector<std::pair<float,int>> ranked;
    for(int u=0;u<n;++u){
        int lo=off[u],hi=off[u+1],nq=qoff[u+1]-qoff[u];
        if(hi-lo<2 || !nq)continue;
        int wanted=qd[qo[qoff[u]+rng()%nq]];
        int k=std::min({width,hi-lo-1,1+int(rng()%width)});
        ranked.clear();
        for(int j=lo;j<hi;++j){
            float w=.10f+.90f/(1.f+std::abs(wanted-day[j])/16.f);
            if(selected[j])w*=3;
            ranked.emplace_back(std::log(unif(rng))/w,j);
        }
        std::partial_sort(ranked.begin(),ranked.begin()+k,ranked.end(),std::greater<>());
        for(int j=0;j<k;++j)masks[int64_t(u)*width+j]=ranked[j].second;
    }
}
