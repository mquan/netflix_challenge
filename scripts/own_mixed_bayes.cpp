#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

// Keep every competition address, then sample other dated exposures without
// replacement, giving nearby activity more weight. Rating values are absent.
extern "C" int select_exposures(int nu,int nm,const uint64_t* off,const uint16_t* movies,const uint16_t* days,
    const uint64_t* qo,const uint16_t* qi,const uint16_t* qd,uint64_t* result,uint16_t* ri,uint16_t* rd){
    std::mt19937 rng(81731);std::uniform_real_distribution<float> uniform(.0000001f,1.f);
    std::vector<int> active(nm+1,-1);std::vector<std::pair<float,uint64_t>> pool;
    std::vector<uint64_t> chosen;
    uint64_t dest=0,ncohort=0;
    for(int u=0;u<nu;++u){
        for(uint64_t j=qo[u];j<qo[u+1];++j)active[qi[j]]=u;
        pool.clear();chosen.clear();
        for(uint64_t j=off[u];j<off[u+1];++j){
            if(active[movies[j]]==u){chosen.push_back(j);++ncohort;continue;}
            int gap=8000;
            for(uint64_t k=qo[u];k<qo[u+1];++k)gap=std::min(gap,std::abs(int(days[j])-qd[k]));
            float w=.23f+1.f/(1.f+gap/22.f)+.65f*(gap==0);
            pool.emplace_back(std::log(uniform(rng))/w,j);
        }
        int keep=std::min(int(pool.size()),int(22+1.3*std::sqrt(pool.size())));
        std::partial_sort(pool.begin(),pool.begin()+keep,pool.end(),std::greater<>());
        for(int k=0;k<keep;++k)chosen.push_back(pool[k].second);
        std::sort(chosen.begin(),chosen.end());
        for(auto j:chosen){ri[dest]=movies[j];rd[dest++]=days[j];}
        result[u+1]=dest;
    }
    return ncohort==qo[nu]?0:1;
}
