#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>
constexpr int K=8,D=16;
extern "C" void reverse_neighbors(int n,const int* qu,const int16_t* qi,const int16_t* qd,
        const int* off,const int16_t* item,const int16_t* day,const float* coord,int* result) {
    std::array<std::vector<std::pair<float,int>>,2> scores;
    for(int j=0;j<n;++j){
        auto &a=scores[0],&b=scores[1];a.clear();b.clear();
        const float* q=coord+int(qi[j])*D;
        for(int h=off[qu[j]];h<off[qu[j]+1];++h){
            int m=item[h];const float* v=coord+m*D;
            float sim=0;for(int k=0;k<D;++k)sim+=v[k]*q[k];
            float gap=std::abs(qd[j]-day[h]);
            a.emplace_back(.85f/(1+gap/10.f)+.15f*sim,h);
            b.emplace_back(.85f*sim+.15f/(1+gap/60.f),h);
        }
        for(int mode=0;mode<2;++mode){
            auto &s=scores[mode];int k=std::min(K,int(s.size()));
            std::partial_sort(s.begin(),s.begin()+k,s.end(),std::greater<>());
            for(int c=0;c<k;++c)result[(int64_t(j)*2+mode)*K+c]=s[c].second;
        }
    }
}
