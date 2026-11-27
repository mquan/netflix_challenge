#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>
namespace {
constexpr int D=8,B=8,T=2300;
struct Session { int n=0;float age=0,pop=0,age2=0,pop2=0,geo[D]={}; };
struct Case { float typical,age,pop,spread,agespread;int count; };
struct Moment {float n=0,e=0,star=0,energy=0;};
void add(Moment& s,float err,int r,float w){s.n+=w;s.e+=w*err;s.star+=w*(r-3.5f);s.energy+=w*err*err;}
void output(const Moment& s,float* p){p[0]=std::log1p(s.n);p[1]=s.e/(s.n+12);p[2]=s.star/(s.n+12);p[3]=s.energy/(s.n+12);}
int bin(float x){return std::clamp(int((std::clamp(x,-2.f,2.f)+2)*2),0,7);}
}
extern "C" void rating_batches(int nu,int nm,const int* off,const int16_t* movies,const int16_t* days,const int8_t* ratings,
        const float* residual,const int* io,const int16_t* ii,const int16_t* id,const int* qo,const int* qr,const int16_t* qi,
        const int16_t* qd,const float* pop,const float* years,const float* geo,float* out){
    std::array<Session,T> sessions;std::array<int,T> stamp;stamp.fill(-1);
    std::vector<Moment> table(int64_t(nm)*4*B*B);
    auto index=[](int m,int group,int a,int b){return ((int64_t(m)*4+group)*B+a)*B+b;};
    std::vector<Case> observed;
    for(int mode=0;mode<2;++mode)for(int u=0;u<nu;++u){int key=mode*nu+u;
        auto insert=[&](int m,int d){if(stamp[d]!=key){stamp[d]=key;sessions[d]=Session{};}auto &s=sessions[d];++s.n;
            s.age+=years[m];s.age2+=years[m]*years[m];s.pop+=pop[m];s.pop2+=pop[m]*pop[m];
            for(int t=0;t<D;++t)s.geo[t]+=geo[m*D+t];};
        for(int j=off[u];j<off[u+1];++j)insert(movies[j],days[j]);
        for(int j=io[u];j<io[u+1];++j)insert(ii[j],id[j]);
        auto profile=[&](int m,int d){const auto& s=sessions[d];float inv=1.f/std::max(s.n-1,1),mu=(s.age-years[m])*inv,p=(s.pop-pop[m])*inv;
            float sim=0,len=0;for(int t=0;t<D;++t){float z=(s.geo[t]-geo[m*D+t])*inv;sim+=z*geo[m*D+t];len+=z*z;}
            float dv=std::sqrt(.12f+std::max(0.f,(s.age2-years[m]*years[m])*inv-mu*mu));
            float dp=std::sqrt(.12f+std::max(0.f,(s.pop2-pop[m]*pop[m])*inv-p*p));
            return Case{sim/std::sqrt(.16f+len),s.n>1?(years[m]-mu)/dv:0,s.n>1?(pop[m]-p)/dp:0,std::sqrt(len),dv,s.n};};
        auto bands=[&](int m,const Case& c){int n=std::min(7,int(std::log2(float(c.count))));return std::array<int64_t,4>{
            index(m,0,n,bin(c.age)),index(m,1,n,bin(c.pop)),index(m,2,n,bin(c.typical*2)),index(m,3,bin(c.age),bin(c.pop))};};
        observed.clear();if(mode)observed.reserve(off[u+1]-off[u]);
        for(int j=off[u];j<off[u+1];++j){int m=movies[j];auto c=profile(m,days[j]);if(mode)observed.push_back(c);
            else for(auto k:bands(m,c))add(table[k],residual[j],ratings[j],1);}
        if(!mode)continue;
        for(int k=qo[u];k<qo[u+1];++k){int j=qr[k],m=qi[j],day=qd[j];const auto c=profile(m,day);float* p=out+int64_t(j)*38;int col=0;
            for(auto index:bands(m,c)){output(table[index],p+col);col+=4;}
            std::array<Moment,4> eff{};
            for(int h=off[u];h<off[u+1];++h){const auto& v=observed[h-off[u]];
                float a=c.age-v.age,g=c.pop-v.pop,t=c.typical-v.typical;
                float size=std::min(c.count,v.count)/float(std::max(c.count,v.count)),w=size/(1+a*a+g*g+3*t*t);
                add(eff[0],residual[h],ratings[h],w*w);if(days[h]==day)add(eff[1],residual[h],ratings[h],1.f/(1+a*a+g*g+3*t*t));
                if((c.typical>.45f)==(v.typical>.45f))add(eff[2],residual[h],ratings[h],size/(1+g*g));
                add(eff[3],residual[h],ratings[h],w/(1+std::abs(day-days[h])/28.f));}
            for(auto& z:eff){output(z,p+col);col+=4;}
            p[32]=std::log1p(c.count);p[33]=c.age;p[34]=c.pop;p[35]=c.typical;p[36]=c.spread;p[37]=c.agespread;
        }
    }
}
