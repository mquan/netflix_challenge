#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>
constexpr int D=24;
using Vec=std::array<float,D>;
extern "C" void chronological(int n,const int* off,const int16_t* day,int* order) {
    for(int u=0;u<n;++u) {
        for(int j=off[u];j<off[u+1];++j)order[j]=j;
        std::stable_sort(order+off[u],order+off[u+1],[&](int a,int b){return day[a]<day[b];});
    }
}
extern "C" void sweep(int first,int last,const int* off,const int* ord,const int16_t* itm,
        const int16_t* day,const int8_t* cat,const float* encoded,float* out) {
    for(int u=first;u<last;++u)for(int window:{1,28}) {
        int end=off[u+1];
        for(int lo=off[u];lo<end;) {
            int bucket=day[ord[lo]]/window,hi=lo+1;
            while(hi<end && day[ord[hi]]/window==bucket)++hi;
            if(hi-lo>1) {
                Vec sum={};
                for(int j=lo;j<hi;++j) {
                    int a=ord[j], index=5*itm[a]+cat[a];
                    const float* v=encoded+index*D;
                    for(int k=0;k<D;++k)sum[k]+=v[k];
                }
                float weight=(window==1?1.f:.35f)/std::pow(hi-lo+2.f,.70f);
                for(int j=lo;j<hi;++j) {
                    int a=ord[j],index=5*itm[a]+cat[a];
                    const float* v=encoded+index*D;float* target=out+index*D;
                    for(int k=0;k<D;++k)target[k]+=weight*(sum[k]-v[k]);
                }
            }
            lo=hi;
        }
    }
}
extern "C" void decode(int first,int last,const int* off,const int* order,const int16_t* ti,
        const int16_t* td,const int8_t* cat,const int* qs,const int* qo,const int16_t* qi,
        const int16_t* qd,const float* enc,float* output) {
    std::vector<Vec> prefix;std::vector<int> days;
    for(int u=first;u<last;++u) {
        if(qs[u]==qs[u+1])continue;
        int n=off[u+1]-off[u];
        prefix.resize(n+1);prefix[0].fill(0);days.resize(n);
        for(int a=0;a<n;++a) {
            int l=order[off[u]+a];days[a]=td[l];
            const float* v=enc+(5*ti[l]+cat[l])*D;
            for(int k=0;k<D;++k)prefix[a+1][k]=prefix[a][k]+v[k];
        }
        for(int a=qs[u];a<qs[u+1];++a) {
            int j=qo[a],m=qi[j],d=qd[j],t=0;
            for(int width:{0,7,60,4096}) {
                int lo=width==4096?0:std::lower_bound(days.begin(),days.end(),d-width)-days.begin();
                int hi=width==4096?n:std::upper_bound(days.begin(),days.end(),d+width)-days.begin();
                Vec mean;
                float norm=0;
                for(int k=0;k<D;++k){mean[k]=(prefix[hi][k]-prefix[lo][k])/(2.f+hi-lo);norm+=mean[k]*mean[k];}
                float* row=output+int64_t(j)*28+t*7;
                row[0]=std::log1p(hi-lo);row[1]=std::sqrt(norm);
                for(int r=0;r<5;++r){
                    const float* v=enc+(5*m+r)*D;
                    float dot=0;for(int k=0;k<D;++k)dot+=v[k]*mean[k];
                    row[2+r]=dot;
                }
                ++t;
            }
        }
    }
}
