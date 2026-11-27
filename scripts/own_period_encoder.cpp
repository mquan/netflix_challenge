#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

constexpr int L=12,W=44;
static void contribution(int m,int rating,float re,float ce,const float* c,float* z) {
    z[0]=1;z[rating]=1;z[6]=re;z[7]=ce;
    for(int k=0;k<L;++k) {
        float v=c[m*L+k];z[8+k]=v;z[8+L+k]=(rating-3.5f)*v;z[8+2*L+k]=ce*v;
    }
}
extern "C" void periods(int n,const int* uoff,const int16_t* dates,int* starts) {
    starts[0]=0;std::array<int,4096> seen;seen.fill(-1);
    for(int u=0;u<n;++u) {
        int cnt=0;
        for(int j=uoff[u];j<uoff[u+1];++j)if(seen[dates[j]]!=u) {seen[dates[j]]=u;++cnt;}
        starts[u+1]=starts[u]+cnt;
    }
}
extern "C" void prefixes(int n,const int* uoff,const int16_t* dates,const int16_t* items,
       const int8_t* ratings,const float* re,const float* ce,const float* coords,const int* starts,
       int16_t* days,float* sums) {
    std::vector<int> ord;
    for(int u=0;u<n;++u) {
        ord.clear();for(int j=uoff[u];j<uoff[u+1];++j)ord.push_back(j);
        std::stable_sort(ord.begin(),ord.end(),[&](int a,int b){return dates[a]<dates[b];});
        std::array<float,W> accum={};
        int slot=starts[u]-1,prev=-1;
        for(int j:ord) {
            if(dates[j]!=prev){prev=dates[j];days[++slot]=prev;}
            std::array<float,W> z={};
            contribution(items[j],ratings[j],re[j],ce[j],coords,z.data());
            for(int k=0;k<W;++k)accum[k]+=z[k];
            std::copy(accum.begin(),accum.end(),sums+int64_t(slot)*W);
        }
    }
}
extern "C" void assemble(int n,int training,const int* rows,const int* user,const int16_t* item,
       const int16_t* date,const int8_t* r,const float* re,const float* ce,const float* c,
       const int* starts,const int16_t* days,const float* sums,float* out) {
    constexpr int halfwidth[]={0,7,45,4096};
    for(int i=0;i<n;++i) {
        int j=rows[i],u=user[j],m=item[j],d=date[j],lo=starts[u],hi=starts[u+1];
        std::array<float,W> own={};
        if(training)contribution(m,r[j],re[j],ce[j],c,own.data());
        for(int t=0;t<4;++t) {
            int l=t==3?lo:std::lower_bound(days+lo,days+hi,d-halfwidth[t])-days;
            int h=t==3?hi:std::upper_bound(days+lo,days+hi,d+halfwidth[t])-days;
            float* dest=out+(int64_t(i)*4+t)*W;
            const float* last=h>lo?sums+int64_t(h-1)*W:nullptr;
            const float* before=l>lo?sums+int64_t(l-1)*W:nullptr;
            float count=(last?last[0]:0)-(before?before[0]:0)-own[0];
            dest[0]=std::log1p(std::max(0.f,count))*.2f;
            float scale=1.f/(4.f+std::max(0.f,count));
            for(int k=1;k<W;++k)dest[k]=((last?last[k]:0)-(before?before[k]:0)-own[k])*scale;
        }
    }
}
