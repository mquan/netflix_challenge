#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>
namespace density {
constexpr int D=14;
struct Sufficient{float n=0;std::array<float,D> sum{};std::array<float,D*D> moment{};};
struct Gaussian {std::array<float,D> mean{};std::array<float,D*D> lower{};float logdet=0,prior=0;};
static void add(Sufficient& s,const float* v,float w){
    s.n+=w;for(int j=0;j<D;++j){s.sum[j]+=w*v[j];for(int k=0;k<=j;++k)s.moment[j*D+k]+=w*v[j]*v[k];}
}
static Gaussian fit(const Sufficient& s,const Sufficient& all){
    Gaussian a;float n=s.n,pull=32,base=all.n+24;
    for(int j=0;j<D;++j)a.mean[j]=(s.sum[j]+pull*all.sum[j]/base)/(n+pull);
    for(int j=0;j<D;++j)for(int k=0;k<=j;++k){
        float global=(all.moment[j*D+k]+24*(j==k))/base-all.sum[j]*all.sum[k]/(base*base);
        float v=(s.moment[j*D+k]-s.sum[j]*a.mean[k]-s.sum[k]*a.mean[j]+n*a.mean[j]*a.mean[k]+pull*global)/(n+pull);
        v+=.18f*(j==k);for(int h=0;h<k;++h)v-=a.lower[j*D+h]*a.lower[k*D+h];
        if(j==k){a.lower[j*D+k]=std::sqrt(std::max(.02f,v));a.logdet+=2*std::log(a.lower[j*D+k]);}
        else a.lower[j*D+k]=v/a.lower[k*D+k];
    }
    a.prior=std::log((n+5)/(all.n+25));return a;
}
static float logp(const Gaussian& a,const float* v){
    float z[D],d=0;for(int j=0;j<D;++j){float x=v[j]-a.mean[j];for(int k=0;k<j;++k)x-=a.lower[j*D+k]*z[k];
        z[j]=x/a.lower[j*D+j];d+=z[j]*z[j];}
    // A multivariate Student likelihood dampens extreme penalties for atypical films.
    return a.prior-.5f*a.logdet-.5f*(8+D)*std::log1p(d/8);
}
}
extern "C" void density_scores(int nu,const int* off,const int16_t* item,const int16_t* days,const int8_t* star,
        const float* residual,const int* qoff,const int* qr,const int16_t* qi,const int16_t* qd,const float* space,float* out){
    using namespace density;
    for(int u=0;u<nu;++u){
        if(qoff[u]==qoff[u+1])continue;int cut=32767;for(int h=qoff[u];h<qoff[u+1];++h)cut=std::min(cut,int(qd[qr[h]]));
        Sufficient stats[2][11]{};
        for(int j=off[u];j<off[u+1];++j){auto v=space+int(item[j])*D;float e=residual[j];
            int c=e<-.75f?0:e<-.23f?1:e<.23f?2:e<.75f?3:4;
            for(int window=0;window<2;++window){float w=window?1.f/(1+std::abs(int(days[j])-cut)/55.f):1;
                add(stats[window][star[j]-1],v,w);add(stats[window][5+c],v,w);add(stats[window][10],v,w);}
        }
        Gaussian state[2][10];
        for(int window=0;window<2;++window)for(int t=0;t<10;++t)state[window][t]=fit(stats[window][t],stats[window][10]);
        for(int h=qoff[u];h<qoff[u+1];++h){int j=qr[h];auto v=space+int(qi[j])*D;float* dst=out+int64_t(j)*20;
            for(int window=0;window<2;++window)for(int kind=0;kind<2;++kind){float z[5],den=0;
                for(int r=0;r<5;++r)z[r]=logp(state[window][kind*5+r],v);float mx=*std::max_element(z,z+5);
                for(float& x:z)den+=(x=std::exp(.65f*(x-mx)));for(int r=0;r<5;++r)dst[window*10+kind*5+r]=z[r]/den;
            }
        }
    }
}
