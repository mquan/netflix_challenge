#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>
constexpr int D=20,C=96,K=20;
extern "C" void audience_add(int n,const int* users,const int16_t* movie,const float* item,float* sum){
    for(int i=0;i<n;++i){
        float* dst=sum+int64_t(users[i])*D;const float* v=item+int(movie[i])*D;
        for(int k=0;k<D;++k)dst[k]+=v[k];
    }
}
extern "C" void audience_group(int n,int nm,const int* u,const int16_t* i,const int* cluster,int* off,int* order){
    int slots=nm*C;
    for(int j=0;j<n;++j)off[int(i[j])*C+cluster[u[j]]+1]++;
    for(int j=0;j<slots;++j)off[j+1]+=off[j];
    std::vector<int> pos(off,off+slots);
    for(int j=0;j<n;++j)order[pos[int(i[j])*C+cluster[u[j]]]++]=j;
}
static uint32_t random_u(uint32_t &v){v^=v<<13;v^=v>>17;v^=v<<5;return v;}
extern "C" void audience_neighbors(int n,int seed,const int* qu,const int16_t* qi,const int16_t* qd,
        const int* u,const int16_t* d,const int* cluster,const int* neighbors,const float* geo,
        const int* off,const int* order,int* result){
    std::vector<std::pair<float,int>> cand;cand.reserve(120);
    for(int j=0;j<n;++j){
        int who=qu[j],m=qi[j],day=qd[j],cl=cluster[who];cand.clear();
        uint32_t rng=uint32_t(seed)*2654435761u+uint32_t(who)*137857u+m+17;if(!rng)rng=1;
        const float* orig=geo+int64_t(who)*D;
        auto draw=[&](int lo,int hi,int num){
            int sz=hi-lo;
            for(int z=0;z<std::min(sz,num);++z){
                int ix=order[lo+(sz<=num?z:random_u(rng)%sz)],other=u[ix];
                if(other==who)continue;
                const float* v=geo+int64_t(other)*D;
                float sim=0;for(int k=0;k<D;++k)sim+=orig[k]*v[k];
                float temporal=1.f/(1.f+std::abs(day-d[ix])/30.f);
                cand.emplace_back(.65f*sim+.35f*temporal,ix);
            }
        };
        for(int k=0;k<4;++k){
            int b=m*C+neighbors[cl*4+k];draw(off[b],off[b+1],k==0?32:14);
        }
        draw(off[m*C],off[(m+1)*C],32);
        std::sort(cand.begin(),cand.end(),std::greater<>());
        int previous=-1,filled=0;int* out=result+int64_t(j)*K;
        for(auto [score,row]:cand){
            if(row==previous)continue;
            out[filled++]=row;previous=row;if(filled==K)break;
        }
        std::fill(out+filled,out+K,-1);
    }
}
