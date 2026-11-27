#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

struct Taste {
    int seen=-1,n=0,implicit=0;
    float err=0,stars=0,high=0;
    void touch(int u){if(seen!=u){*this=Taste{};seen=u;}}
};
extern "C" void title_taste(int users,int tokens,const int* off,const int16_t* movie,const int8_t* rating,const float* err,
        const int* movie_off,const int* term,const int8_t* family,const float* idf,const int* qoff,const int* qrow,
        const int16_t* qi,const int* ioff,const int16_t* implicit,float* out){
    std::vector<Taste> tally(tokens);
    for(int u=0;u<users;++u){
        for(int j=off[u];j<off[u+1];++j){int m=movie[j];
            for(int k=movie_off[m];k<movie_off[m+1];++k){auto &s=tally[term[k]];s.touch(u);++s.n;
                s.err+=err[j];s.stars+=rating[j]-3.5f;s.high+=rating[j]>=4;}}
        for(int j=ioff[u];j<ioff[u+1];++j){int m=implicit[j];
            for(int k=movie_off[m];k<movie_off[m+1];++k){auto &s=tally[term[k]];s.touch(u);++s.implicit;}}
        for(int j=qoff[u];j<qoff[u+1];++j){int row=qrow[j],m=qi[row];float* z=out+int64_t(row)*24;
            std::fill(z,z+24,0.f);float denom[3]={},best[3]={};
            for(int k=movie_off[m];k<movie_off[m+1];++k){int t=term[k],g=family[t];auto &s=tally[t];s.touch(u);
                float* p=z+8*g;float w=idf[t],shrink=s.n/(s.n+4.f);
                p[0]+=w*std::log1p(s.n);p[1]+=w*s.err/(s.n+4.f);p[2]+=w*s.stars/(s.n+4.f);
                p[3]+=w*(s.high-.5f*s.n)/(s.n+4.f);p[4]+=w*std::log1p(std::max(0,s.implicit-1));
                p[5]+=w*shrink;denom[g]+=w;
                if(w*shrink>best[g]){best[g]=w*shrink;p[6]=s.err/(s.n+2.f);p[7]=std::log1p(s.n);}}
            for(int g=0;g<3;++g)if(denom[g])for(int h=0;h<6;++h)z[g*8+h]/=denom[g];
        }
    }
}
