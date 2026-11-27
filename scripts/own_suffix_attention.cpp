#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

extern "C" void verify_suffix(int nu,int nm,const uint64_t* off,const uint16_t* mi,const uint16_t* days,
    const uint64_t* qo,const uint16_t* qi,const uint16_t* qd,uint64_t* metrics){
    std::vector<int> flag(nm+1,-1);
    for(int u=0;u<nu;++u){
        int d=qd[qo[u]];
        for(uint64_t j=qo[u];j<qo[u+1];++j)flag[qi[j]]=u;
        uint64_t missing=0,inside=0,after=0;
        for(uint64_t j=off[u];j<off[u+1];++j){
            bool selected=flag[mi[j]]==u;
            if(selected)++inside;
            if(days[j]>d){++after;if(!selected)++missing;}
        }
        metrics[0]+=missing>0;metrics[1]+=missing;metrics[2]+=inside;
        metrics[3]+=inside!=qo[u+1]-qo[u];metrics[4]+=after;
    }
}

static void detailed_masks(int* prefix_end,int* withheld,int b,int hist,int targets,int seed,const int* users,const int64_t* off,
    const int16_t* movie,const int16_t* day,const int8_t* star,const int16_t* boundary,
    const int* cohort,const int* hidden,int* hi,int* hr,int* hd,int* ti,int* td,int* labels,float* counts){
    std::mt19937 rng(seed);std::uniform_real_distribution<float> coin(0,1);
    std::vector<int> tie,slate,visible,near,extra,kept;std::vector<int8_t> mask;
    for(int j=0;j<b;++j){
        int u=users[j],lo=off[u],end=off[u+1];
        int upto=std::lower_bound(day+lo,day+end,boundary[u])-day;
        if(upto-lo<3)upto=end;
        if(upto-lo<2)continue;
        int span=upto-lo;
        // Earlier complete stopping dates provide repeated chronological training examples.
        // The newest complete date is a larger part of the training distribution.
        int pick=upto-1;
        if(span>12 && coin(rng)<.72f){
            float lower=coin(rng)<.72f?.55f:.18f;
            pick=lo+std::clamp(int(span*(lower+(1-lower)*coin(rng))),10,span-1);
        }
        int stop=std::upper_bound(day+pick,day+upto,day[pick])-day;
        int k=std::min({targets,cohort[u],stop-lo-1});k=std::max(1,k);
        if(prefix_end)prefix_end[j]=stop;
        int first=stop-k;int start=std::lower_bound(day+lo,day+first+1,day[first])-day;
        int middle=std::upper_bound(day+first,day+stop,day[first])-day;
        tie.resize(middle-start);std::iota(tie.begin(),tie.end(),start);std::shuffle(tie.begin(),tie.end(),rng);
        slate.assign(tie.begin(),tie.begin()+k-(stop-middle));
        for(int r=middle;r<stop;++r)slate.push_back(r);
        std::shuffle(slate.begin(),slate.end(),rng);
        int n=std::clamp(int(std::round(float(k)*hidden[u]/std::max(1,cohort[u]))),1,k);
        slate.resize(n);mask.assign(stop-lo,0);std::vector<int> times;
        for(int t=0;t<n;++t){
            int r=slate[t];mask[r-lo]=1;times.push_back(day[r]);
            if(withheld)withheld[j*targets+t]=r;
            ti[j*targets+t]=movie[r];td[j*targets+t]=day[r];labels[j*targets+t]=star[r]-1;
        }
        std::sort(times.begin(),times.end());int center=(times[(n-1)/2]+times[n/2])/2;
        visible.clear();near.clear();extra.clear();kept.clear();
        for(int r=lo;r<stop;++r)if(!mask[r-lo])visible.push_back(r);
        counts[j]=visible.size();int cap=hist-n;
        if(int(visible.size())<=cap)kept=visible;
        else{
            for(int r:visible)if(std::abs(int(day[r])-center)<=14)near.push_back(r);
            std::shuffle(near.begin(),near.end(),rng);near.resize(std::min(int(near.size()),cap/2));
            for(int r:near)mask[r-lo]=1;
            for(int r:visible)if(!mask[r-lo])extra.push_back(r);
            std::shuffle(extra.begin(),extra.end(),rng);kept=near;
            kept.insert(kept.end(),extra.begin(),extra.begin()+(cap-int(near.size())));
        }
        int l=j*hist;
        for(int r:kept){hi[l]=movie[r];hd[l]=day[r];hr[l++]=star[r];}
        for(int r:slate){hi[l]=movie[r];hd[l]=day[r];hr[l++]=6;}
    }
}

extern "C" void suffix_masks(int b,int hist,int targets,int seed,const int* users,const int64_t* off,
    const int16_t* movie,const int16_t* day,const int8_t* star,const int16_t* boundary,
    const int* cohort,const int* hidden,int* hi,int* hr,int* hd,int* ti,int* td,int* labels,float* counts){
    detailed_masks(nullptr,nullptr,b,hist,targets,seed,users,off,movie,day,star,boundary,cohort,hidden,hi,hr,hd,ti,td,labels,counts);
}
