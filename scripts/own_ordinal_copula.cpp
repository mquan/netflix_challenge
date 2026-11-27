#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>
namespace {
constexpr int K=11;
struct Normal {
    std::array<float,16001> cdf;
    Normal(){for(int k=0;k<=16000;++k)cdf[k]=.5f*std::erfc(-(k*.001f-8)*.70710678f);}
    float operator()(float x)const{float j=std::clamp((x+8)*1000,0.f,15999.f);int i=int(j);return cdf[i]+(j-i)*(cdf[i+1]-cdf[i]);}
};
}
extern "C" void copula(int n,int dims,const int* rows,const int* peers,const uint8_t* valid,const int* movie,const int* day,
        const int8_t* rating,const float* threshold,const float* geo,const float* nodes,const float* weights,float* out){
    static Normal phi;
    for(int j=0;j<n;++j){int row=rows[j],m=movie[row];auto target=threshold+int64_t(row)*4;float* o=out+int64_t(j)*8;
        float raw=5;for(int r=0;r<4;++r)raw-=phi(target[r]);
        for(int mode=0;mode<4;++mode){
            float mass[K];std::copy_n(weights,K,mass);float base=.48f+(mode%2)*.19f;
            for(int h=0;h<9;++h){if(!valid[j*9+h])continue;int a=peers[j*9+h],r=rating[a],am=movie[a];float sim=0;
                for(int t=0;t<dims;++t)sim+=geo[m*dims+t]*geo[am*dims+t];
                float near=1/(1+std::abs(day[row]-day[a])/22.f);
                float loading=base*(mode<2?(.30f+.70f*near):(.27f+.36f*near+.60f*sim));loading=std::clamp(loading,-.55f,.80f);
                float inv=1/std::sqrt(1-loading*loading);auto bounds=threshold+int64_t(a)*4;
                for(int k=0;k<K;++k){float hi=r==5?1.f:phi((bounds[r-1]-loading*nodes[k])*inv);
                    float lo=r==1?0.f:phi((bounds[r-2]-loading*nodes[k])*inv);mass[k]*=std::max(hi-lo,1e-8f);}
            }
            float total=0,val=0,mood=0;float inv=1/std::sqrt(1-base*base);
            for(int k=0;k<K;++k){float mu=5;for(int r=0;r<4;++r)mu-=phi((target[r]-base*nodes[k])*inv);
                total+=mass[k];val+=mass[k]*mu;mood+=mass[k]*nodes[k];}
            o[2*mode]=val/std::max(total,1e-30f)-raw;o[2*mode+1]=mood/std::max(total,1e-30f);
        }
    }
}
