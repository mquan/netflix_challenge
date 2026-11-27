#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

extern "C" void session_graph(int first,int last,int nitems,const int32_t* starts,
        const int32_t* byitem,const int32_t* es,const int32_t* off,const int16_t* items,
        float* matrix) {
    for(int m=first;m<last;++m) {
        auto row=matrix+int64_t(m)*nitems;
        for(int j=starts[m];j<starts[m+1];++j){
            int a=byitem[j],s=es[a],n=off[s+1]-off[s];
            if(n<2)continue;
            float w=1.f/(n-1);
            for(int k=off[s];k<off[s+1];++k)if(k!=a)row[items[k]]+=w;
        }
    }
}

// Store profile sums only for sessions that have another film.
static inline void update(float* row,int d,const float* v,int r,float re,float ce,float sign){
    for(int j=0;j<d;++j) {
        row[j]+=sign*v[j];
        if(r){
            row[d+j]+=sign*(r-3.6f)*v[j];
            row[2*d+j]+=sign*re*v[j];
            row[3*d+j]+=sign*ce*v[j];
        }
    }
    row[4*d]+=sign;
    if(r){
        row[4*d+1]+=sign;
        row[4*d+1+r]+=sign;
        row[4*d+7]+=sign*re;
        row[4*d+8]+=sign*ce;
    }
}

extern "C" void session_sums(int first,int last,int d,const int32_t* off,const int32_t* mapping,
        const int16_t* items,const int8_t* r,const float* re,const float* ce,
        const float* vec,_Float16* sums){
    int width=4*d+9;
    std::vector<float> buf(width);
    for(int s=first;s<last;++s){
        int ix=mapping[s];if(ix<0)continue;
        std::fill(buf.begin(),buf.end(),0.f);
        for(int k=off[s];k<off[s+1];++k)
            update(buf.data(),d,vec+int(items[k])*d,r[k],re[k],ce[k],1.f);
        auto dest=sums+int64_t(ix)*width;
        for(int j=0;j<width;++j)dest[j]=buf[j];
    }
}

extern "C" void session_design(int n,int d,int item,const int32_t* sessions,const int32_t* mapping,
        const _Float16* sums,const float* vec,const int8_t* rating,const float* re,const float* ce,
        int training,float* out){
    int width=4*d+9;
    for(int a=0;a<n;++a){
        float* row=out+int64_t(a)*(width+1);
        int ix=mapping[sessions[a]];
        if(ix<0){std::fill_n(row,width,0.f);}
        else {
            auto src=sums+int64_t(ix)*width;
            for(int j=0;j<width;++j)row[j]=src[j];
            update(row,d,vec+item*d,training?rating[a]:0,training?re[a]:0.f,training?ce[a]:0.f,-1.f);
            float all=std::max(0.f,row[4*d]),known=std::max(0.f,row[4*d+1]);
            float ni=1.f/std::sqrt(all+1.f),nr=1.f/std::sqrt(known+1.f);
            for(int j=0;j<d;++j)row[j]*=ni;
            for(int j=d;j<4*d;++j)row[j]*=nr;
            for(int j=4*d+2;j<width;++j)row[j]/=std::max(1.f,known);
            row[4*d]=std::log1p(all);
            row[4*d+1]=std::log1p(known);
        }
        row[width]=1.f;
    }
}
