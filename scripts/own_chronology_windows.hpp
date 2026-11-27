#pragma once
#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

constexpr int Q=9,D=96,H=320,R=32,CASES=4;
using Vec=std::array<float,R>;
struct Window{int end=0,count=0,visible=0;std::array<int,Q> rows{};std::array<float,Q> mu{};};
struct Data{
    int nu,nm;float sd;const int64_t* off;const int16_t *item,*day;const int8_t* rating;const int16_t* bound;
    const int* grid;const int16_t *qi,*qd;const int8_t* anchor;const int* query_idx;
    const float *enc,*proj,*hb,*op,*bias,*ub,*drift;
};
static int raw_count(const Data& a,int u){int n=0;while(n<Q&&a.grid[int64_t(u)*Q+n]>=0)++n;return n;}
static void make_window(const Data& a,int u,int c,std::mt19937& rng,Window& w,float* law=nullptr,float* input=nullptr){
    int lo=a.off[u],end=a.off[u+1],maxend=std::lower_bound(a.day+lo,a.day+end,a.bound[u])-a.day;
    if(maxend-lo<3)maxend=end;
    int n=maxend-lo,nco=raw_count(a,u);if(n<2||!nco)return;
    int pick=maxend-1;
    if(c && n>12){
        float frac=(c==1?.73f:c==2?.45f:.18f);
        pick=lo+std::clamp(int(n*(frac+(1-frac)*std::generate_canonical<float,24>(rng))),10,n-1);
    }
    w.end=std::upper_bound(a.day+pick,a.day+maxend,a.day[pick])-a.day;
    w.count=std::min(nco,w.end-lo-1);int first=w.end-w.count,cut=a.day[first];
    int left=std::lower_bound(a.day+lo,a.day+first+1,cut)-a.day,right=std::upper_bound(a.day+first,a.day+w.end,cut)-a.day;
    std::vector<int> tie(right-left);std::iota(tie.begin(),tie.end(),left);std::shuffle(tie.begin(),tie.end(),rng);
    int added=0;for(int j=0;j<w.count-(w.end-right);++j)w.rows[added++]=tie[j];
    for(int j=right;j<w.end;++j)w.rows[added++]=j;
    std::shuffle(w.rows.begin(),w.rows.begin()+w.count,rng);
    int vis=0;for(int j=0;j<nco;++j)vis+=a.anchor[a.grid[int64_t(u)*Q+j]]>0;
    w.visible=std::clamp(int(std::round(w.count*float(vis)/nco)),0,w.count-1);
    double ds=0,total[D]={};
    for(int j=lo;j<w.end;++j){
        auto v=a.enc+(int64_t(a.rating[j]-1)*a.nm+a.item[j])*D;
        for(int h=0;h<D;++h)total[h]+=v[h];ds+=a.day[j];
    }
    for(int k=w.visible;k<w.count;++k){
        int j=w.rows[k];auto v=a.enc+(int64_t(a.rating[j]-1)*a.nm+a.item[j])*D;
        for(int h=0;h<D;++h)total[h]-=v[h];ds-=a.day[j];
    }
    int support=w.end-lo-w.count+w.visible;float md=ds/support,x[D],hidden[H];
    for(int h=0;h<D;++h)x[h]=total[h]/std::sqrt(float(support));
    if(input)std::copy(x,x+D,input);
    cblas_sgemv(CblasRowMajor,CblasNoTrans,H,D,1.f,a.proj,D,x,1,0.f,hidden,1);
    for(int h=0;h<H;++h)hidden[h]+=a.hb[h];const int dim=H;vvtanhf(hidden,hidden,&dim);
    for(int k=w.visible;k<w.count;++k){
        int j=w.rows[k],m=a.item[j];float rel=(a.day[j]-md)/a.sd,dev=std::copysign(std::pow(std::abs(rel),.4f),rel),z[5];
        for(int h=0;h<5;++h)z[h]=cblas_sdot(H,hidden,1,a.op+(int64_t(m)*5+h)*H,1)+
            a.bias[m*5+h]+a.ub[u*5+h]+a.drift[u*5+h]*dev;
        float mx=*std::max_element(z,z+5),s=0,v=0;for(int h=0;h<5;++h){float p=std::exp(z[h]-mx);s+=p;v+=(h+1)*p;}
        w.mu[k]=v/s;
        if(law)for(int h=0;h<5;++h)law[5*k+h]=std::exp(z[h]-mx)/s;
    }
}
