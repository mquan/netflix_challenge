#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
constexpr int D=96,M=6;
extern "C" void cf_context(int lo,int hi,int nfilm,const int* qu,const int16_t* qi,const int16_t* qd,
        const int* off,const int16_t* ti,const int16_t* td,const int8_t* r,const float* a,
        const int16_t* cos,const int16_t* scene,float* output,float* support,float* date_mean) {
    for(int j=lo;j<hi;++j) {
        float* out=output+int64_t(j-lo)*D*M;
        int u=qu[j],m=qi[j],n=off[u+1]-off[u];float sumday=0;
        double exact[D]={};std::array<float,M> ss={},sw={};
        for(int l=off[u];l<off[u+1];++l) {
            int mm=ti[l];float dd=std::abs(qd[j]-td[l]);
            float sr=cos[int64_t(m)*nfilm+mm]/16000.f,sg=scene[int64_t(m)*nfilm+mm]/16000.f;
            float t=1.f/(1.f+dd/18.f),sim=sr*sr,genre=std::max(sg,0.f);
            float w[M]={1.f,.1f+t,.10f+(dd==0),.08f+sim,.06f+(.22f+sim)*t,.08f+genre*genre};
            const float* vec=a+(int64_t(r[l]-1)*nfilm+mm)*D;
            for(int k=0;k<D;++k)exact[k]+=vec[k];
            for(int h=1;h<M;++h) {
                float* ptr=out+h*D;float wt=w[h];
                ss[h]+=wt*wt;sw[h]+=wt;
                for(int k=0;k<D;++k)ptr[k]+=wt*vec[k];
            }
            sumday+=td[l];
        }
        date_mean[j-lo]=n?sumday/n:0;
        float den=std::sqrt(std::max(1,n));
        for(int k=0;k<D;++k)out[k]=exact[k]/den;
        for(int h=1;h<M;++h) {
            float div=1.f/std::sqrt(std::max(1e-8f,ss[h]));
            for(int k=0;k<D;++k)out[h*D+k]*=div;
            support[(j-lo)*(M-1)+h-1]=std::log1p(sw[h]*sw[h]/std::max(1e-8f,ss[h]));
        }
    }
}
