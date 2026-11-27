#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>
constexpr int DAYS=2300,B=8,N=4,OUTW=48;
struct Case{int life,age,tenure,bulk;float expected;};
struct Sum{float n=0,err=0,sq=0,star=0,high=0,low=0;};
static void add(Sum& z,float w,int r,float e){
    z.n+=w;z.err+=w*e;z.sq+=w*e*e;z.star+=w*(r-3.6f);z.high+=w*(r==5);z.low+=w*(r<=2);
}
static void write(const Sum& s,float* out){
    float den=1/(8+s.n);out[0]=std::log1p(s.n);out[1]=s.err*den;out[2]=s.sq*den;
    out[3]=s.star*den;out[4]=s.high*den;out[5]=s.low*den;
}
extern "C" void lifecycle(int nu,int nm,int nq,const int* off,const int16_t* it,const int16_t* day,const int8_t* star,
     const float* err,const int16_t* year,const int* qoff,const int* qr,const int16_t* qi,const int16_t* qd,
     const float* forecast,const int* ioff,const int16_t* ii,const int16_t* id,float* out){
    std::vector<int> cdf(int64_t(nm)*DAYS),earliest(nm),popular(nm);
    for(int j=0;j<off[nu];++j)++cdf[int64_t(it[j])*DAYS+day[j]];
    for(int j=0;j<ioff[nu];++j)++cdf[int64_t(ii[j])*DAYS+id[j]];
    for(int m=0;m<nm;++m){int* v=cdf.data()+int64_t(m)*DAYS;
        for(int h=1;h<DAYS;++h)v[h]+=v[h-1];int n=v[DAYS-1];
        earliest[m]=std::lower_bound(v,v+DAYS,std::max(1,n/100))-v;
        popular[m]=std::lower_bound(v,v+DAYS,std::max(1,n/2))-v;}
    std::vector<Sum> film(int64_t(nm)*B*B*3);std::vector<Case> history;
    std::array<int,DAYS> cnt{},owner;owner.fill(-1);int serial=0;
    for(int mode=0;mode<2;++mode)for(int u=0;u<nu;++u){
        ++serial;int first=DAYS,last=-1;auto session=[&](int d){
            if(owner[d]!=serial){owner[d]=serial;cnt[d]=0;}++cnt[d];first=std::min(first,d);last=std::max(last,d);};
        for(int j=off[u];j<off[u+1];++j)session(day[j]);
        for(int j=ioff[u];j<ioff[u+1];++j)session(id[j]);
        auto state=[&](int m,int d,float mu){
            int* c=cdf.data()+int64_t(m)*DAYS;
            int life=std::min(7,int((c[d]+(d?c[d-1]:0))*4.f/std::max(1,c[DAYS-1])));
            float age=std::max(0.f,1999.86f+d/365.25f-year[m]);
            int a=std::min(7,int(1.7f*std::log1p(age))),ten=std::min(7,int(std::log1p(std::max(0,d-first)))),
                bulk=std::min(7,int(std::log2(float(std::max(1,cnt[d])))));
            return Case{life,a,ten,bulk,mu};
        };
        auto idx=[](int m,int band,int x,int y){return ((int64_t(m)*3+band)*B+x)*B+y;};
        history.clear();history.reserve(off[u+1]-off[u]);
        for(int j=off[u];j<off[u+1];++j){
            int m=it[j],r=star[j];auto c=state(m,day[j],r-err[j]);
            if(mode)history.push_back(c);
            else{add(film[idx(m,0,c.life,c.bulk)],1,r,err[j]);add(film[idx(m,1,c.tenure,c.bulk)],1,r,err[j]);
                add(film[idx(m,2,c.life,c.tenure)],1,r,err[j]);}
        }
        if(!mode)continue;
        for(int z=qoff[u];z<qoff[u+1];++z){
            int j=qr[z],m=qi[j],d=qd[j];auto q=state(m,d,forecast[j]);std::array<Sum,4> nearby{};
            for(int k=off[u];k<off[u+1];++k){
                const auto& c=history[k-off[u]];int r=star[k];float e=err[k];
                float life=1.f/(1.f+std::abs(q.life-c.life)*1.5f),age=1.f/(1.f+std::abs(q.age-c.age)*1.5f),
                      tenure=1.f/(1.f+std::abs(q.tenure-c.tenure)),bulk=1.f/(1.f+std::abs(q.bulk-c.bulk)),
                      score=1.f/(1+4*(q.expected-c.expected)*(q.expected-c.expected));
                add(nearby[0],life*life*age*age,r,e);add(nearby[1],bulk*bulk*tenure*tenure,r,e);
                add(nearby[2],life*age*score*bulk,r,e);
                if((q.age<=1)==(c.age<=1)&&(q.bulk<2)==(c.bulk<2))add(nearby[3],score,r,e);
            }
            float* dst=out+int64_t(j)*OUTW;for(int h=0;h<4;++h)write(nearby[h],dst+h*6);
            write(film[idx(m,0,q.life,q.bulk)],dst+24);write(film[idx(m,1,q.tenure,q.bulk)],dst+30);
            write(film[idx(m,2,q.life,q.tenure)],dst+36);
            dst[42]=q.life;dst[43]=q.age;dst[44]=q.tenure;dst[45]=q.bulk;
            dst[46]=std::asinh((d-earliest[m])/90.f);dst[47]=std::asinh((d-popular[m])/180.f);
        }
    }
}
