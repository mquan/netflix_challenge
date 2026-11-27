#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

extern "C" void retrieve(int n,int k,int ni,const int32_t* qu,const int16_t* qi,const int16_t* qd,
        const int32_t* off,const int16_t* ti,const int16_t* td,const int16_t* gram,
        const int16_t* scene,int32_t* indices){
    std::vector<std::pair<float,int>> c;
    for(int row=0;row<n;++row){
        int u=qu[row],m=qi[row],d=qd[row];
        auto s=gram+int64_t(m)*ni,ss=scene+int64_t(m)*ni;
        c.clear();
        for(int a=off[u];a<off[u+1];++a){
            if(ti[a]==m)continue;
            float sim=s[ti[a]]/16000.f,sc=ss[ti[a]]/16000.f;
            float time=1.f/(1.f+std::abs(d-td[a])/12.f);
            float weight=.44f*sim+.25f*sc+.16f*time*(1.f+sim)+.15f*float(d==td[a]);
            c.emplace_back(weight,a);
        }
        int keep=std::min(k,static_cast<int>(c.size()));
        if(keep<int(c.size()))std::nth_element(c.begin(),c.begin()+keep,c.end(),std::greater<>());
        std::sort(c.begin(),c.begin()+keep,std::greater<>());
        auto dst=indices+int64_t(row)*k;
        for(int j=0;j<keep;++j)dst[j]=c[j].second;
        std::fill(dst+keep,dst+k,-1);
    }
}
extern "C" void global_profile(int nu,int dim,const int32_t* off,const int16_t* item,
        const int8_t* rating,const float* movie,float* profile){
    int width=2*dim+5;
    for(int u=0;u<nu;++u){
        auto row=profile+int64_t(u)*width;
        for(int k=off[u];k<off[u+1];++k){
            int r=rating[k];
            auto vec=movie+int(item[k])*dim;
            for(int j=0;j<dim;++j){row[j]+=vec[j];row[dim+j]+=(r-3.6f)*vec[j];}
            row[2*dim+r-1]+=1;
        }
    }
}
