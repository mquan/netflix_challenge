#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

// Select peers using only dates and fitted movie coordinates. Training and
// query groups use the same rule, with a small random perturbation at training.
extern "C" void cavity_group(int n,int k,int seed,int mode,int train,const int* users,
    const int* target,const int16_t* movies,const int16_t* dates,const int* off,
    const int16_t* items,const int16_t* days,const float* geo,const int* qoff,
    const int16_t* qdays,const int* aoff,const int* arows,int* rows){
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uni(0.f,1.f);
    std::vector<std::pair<float,int>> candidates;
    for(int j=0;j<n;++j){
        int u=users[j],lo=off[u],hi=off[u+1],t=-1;
        int m,d;
        if(train){
            if(hi<=lo)continue;
            t=lo+rng()%(hi-lo);
            float s=uni(rng);
            if(s<.28f && aoff[u]<aoff[u+1])t=arows[aoff[u]+rng()%(aoff[u+1]-aoff[u])];
            else if(s<.82f && qoff[u]<qoff[u+1]){
                int day=qdays[qoff[u]+rng()%(qoff[u+1]-qoff[u])];
                float best=-100;
                for(int h=0;h<14;++h){
                    int r=lo+rng()%(hi-lo);
                    float v=uni(rng)-.45f*std::log1p(std::abs(int(days[r])-day)/14.f);
                    if(v>best){best=v;t=r;}
                }
            }
            m=items[t];d=days[t];rows[j*k]=t;
        }else{m=movies[target[j]];d=dates[target[j]];}
        candidates.clear();
        auto x=geo+m*24;
        for(int r=lo;r<hi;++r){
            if(r==t || items[r]==m)continue;
            float sim=0;
            auto v=geo+items[r]*24;
            for(int h=0;h<24;++h)sim+=x[h]*v[h];
            float dt=std::log1p(std::abs(int(days[r])-d)/8.f);
            float score=(mode?2.8f:1.1f)*sim-(mode?.13f:.82f)*dt+(train?.9f*uni(rng):0.f);
            candidates.emplace_back(score,r);
        }
        int cnt=std::min(int(candidates.size()),k-1);
        std::partial_sort(candidates.begin(),candidates.begin()+cnt,candidates.end(),std::greater<>());
        for(int h=0;h<cnt;++h)rows[j*k+1+h]=candidates[h].second;
    }
}
