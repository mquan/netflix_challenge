#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

struct Pair {
    float count=0,stars=0,r=0,c=0,labeled=0,rcov=0,ccov=0,rr=0,cc=0;
};
constexpr int NG=3, NF=9;
extern "C" void session_pairs(int first,int last,int nitems,const int32_t* starts,
        const int32_t* rows,const int32_t* event_for_train,const int32_t* qs,const int32_t* qo,
        const int32_t* qsession,const int32_t* es,const int32_t* session_starts,
        const int16_t* items,const int8_t* ratings,const float* re,const float* ce,float* output) {
    std::vector<Pair> pair(nitems);
    for(int item=first;item<last;++item){
        if(qs[item]==qs[item+1])continue;
        std::fill(pair.begin(),pair.end(),Pair{});
        double mr=0,mc=0,ms=0;
        for(int j=starts[item];j<starts[item+1];++j){
            int a=event_for_train[rows[j]];
            mr+=re[a];mc+=ce[a];ms+=ratings[a];
        }
        int count=std::max(1,starts[item+1]-starts[item]);
        mr/=count;mc/=count;ms/=count;
        for(int j=starts[item];j<starts[item+1];++j){
            int a=event_for_train[rows[j]],sid=es[a];
            int lo=session_starts[sid],hi=session_starts[sid+1];
            float w=1.f/std::sqrt(1.f+(hi-lo)/20.f);
            float r=re[a]-mr,c=ce[a]-mc,s=ratings[a]-ms;
            for(int k=lo;k<hi;++k){
                if(k==a)continue;
                auto& t=pair[items[k]];
                t.count+=w;t.stars+=w*s;t.r+=w*r;t.c+=w*c;
                if(ratings[k]){
                    t.labeled+=w;t.rcov+=w*r*re[k];t.ccov+=w*c*ce[k];
                    t.rr+=w*re[k]*re[k];t.cc+=w*ce[k]*ce[k];
                }
            }
        }
        for(int j=qs[item];j<qs[item+1];++j){
            int row=qo[j],sid=qsession[row];
            float accum[NG][NF]{};
            for(int k=session_starts[sid];k<session_starts[sid+1];++k){
                int other=items[k];
                if(other==item)continue;
                const auto& s=pair[other];
                int g=ratings[k]?1:2;
                for(int z : {0,g}){
                    auto& v=accum[z];
                    float w=s.count/(s.count+50.f);
                    v[0]+=1;v[1]+=w;v[2]+=std::log1p(s.count);
                    v[3]+=s.stars/(s.count+50.f);
                    v[4]+=s.r/(s.count+50.f);
                    v[5]+=s.c/(s.count+50.f);
                    v[6]+=s.rcov/(s.rr+50.f)*re[k];
                    v[7]+=s.ccov/(s.cc+50.f)*ce[k];
                    v[8]+=s.c/(s.count+400.f);
                }
            }
            auto* out=output+int64_t(row)*NG*NF;
            for(int g=0;g<NG;++g){
                const auto& v=accum[g];
                out[g*NF]=std::log1p(v[0]);
                out[g*NF+1]=std::log1p(v[1]);
                out[g*NF+2]=v[2]/std::max(v[0],1.f);
                for(int s=3;s<NF;++s)out[g*NF+s]=v[s]/std::max(1.f,v[1]);
            }
        }
    }
}
