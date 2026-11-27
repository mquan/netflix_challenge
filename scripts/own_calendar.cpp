#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>
namespace festive {
constexpr int T=2250,K=16;
struct Moment {float n=0,weight=0,e=0,positive=0;};
static Moment plus(Moment a,const Moment& b){return {a.n+b.n,a.weight+b.weight,a.e+b.e,a.positive+b.positive};}
static Moment minus(Moment a,const Moment& b){return {a.n-b.n,a.weight-b.weight,a.e-b.e,a.positive-b.positive};}
static void features(Moment z,float mean,float* out){
    out[0]=std::log1p(std::max(0.f,z.n));out[1]=(z.e-z.weight*mean)/(12+z.weight);
    out[2]=(z.positive-.5f*z.weight)/(12+z.weight);
}
}
extern "C" void calendar(int nu,int nm,int nq,const int* off,const int16_t* item,const int16_t* dates,const int8_t* stars,
       const float* errors,const int* qoff,const int* qr,const int16_t* qi,const int16_t* qd,int ni,const int16_t* ii,
       const int16_t* id,const int* nb,const int* season_map,const int* weekday,float* out){
    using namespace festive;
    std::vector<Moment> grid(int64_t(nm)*T),seasons(int64_t(nm)*366),week(int64_t(nm)*7);
    std::vector<float> mean(nm),global(T),user_sum(nu),user_week(nu*7),user_count(nu*7),user_peak(nu*3),peak_count(nu*3);
    for(int j=0;j<off[nu];++j){
        int m=item[j],d=dates[j],s=season_map[d],w=weekday[d];Moment a{1,1,errors[j],float(stars[j]>=4)};
        for(Moment* p:{&grid[int64_t(m)*T+d],&seasons[m*366+s],&week[m*7+w]})*p=plus(*p,a);
        ++global[d];
    }
    for(int j=0;j<ni;++j){int m=ii[j],d=id[j];++grid[int64_t(m)*T+d].n;++seasons[m*366+season_map[d]].n;++week[m*7+weekday[d]].n;++global[d];}
    for(int m=0;m<nm;++m){
        auto row=grid.data()+int64_t(m)*T;for(int d=1;d<T;++d)row[d]=plus(row[d],row[d-1]);
        mean[m]=row[T-1].e/(row[T-1].weight+55);
    }
    auto interval=[&](int m,int d,int width){
        auto row=grid.data()+int64_t(m)*T;int l=std::max(0,d-width),r=std::min(T-1,d+width);
        return l?minus(row[r],row[l-1]):row[r];
    };
    auto demand=[&](int m,int d){return std::log((interval(m,d,3).n+3)/(interval(m,d,42).n+36)*12);};
    for(int u=0;u<nu;++u){
        for(int j=off[u];j<off[u+1];++j){
            int m=item[j],d=dates[j],w=weekday[d];float e=errors[j],z=demand(m,d);
            user_sum[u]+=e;user_week[u*7+w]+=e;++user_count[u*7+w];int pk=z>.25?2:z<-.25?0:1;
            user_peak[u*3+pk]+=e;++peak_count[u*3+pk];
        }
        for(int h=qoff[u];h<qoff[u+1];++h){
            int j=qr[h],m=qi[j],d=qd[j],s=season_map[d],w=weekday[d];float* dst=out+int64_t(j)*49;int pos=0;
            auto append=[&](Moment c){features(c,mean[m],dst+pos);pos+=3;};
            for(int rad:{0,2,7,22,70,200})append(interval(m,d,rad));
            for(int rad:{4,20}){
                Moment c;for(int g=-rad;g<=rad;++g)c=plus(c,seasons[m*366+(s+g+366)%366]);append(c);
            }
            append(week[m*7+w]);Moment sameweekday;
            for(int g=-42;g<=42;g+=7)if(d+g>=0&&d+g<T)sameweekday=plus(sameweekday,interval(m,d+g,0));
            append(sameweekday);
            float demand0=demand(m,d);
            for(int rad:{2,15,65}){
                float mu=0,demand_neighbor=0,exposure=0;
                for(int k=0;k<K;++k){int other=nb[m*K+k];auto v=interval(other,d,rad);
                    mu+=(v.e-v.weight*mean[other])/(12+v.weight);demand_neighbor+=demand(other,d);
                    exposure+=std::log1p(v.n);}
                dst[pos++]=mu/K;dst[pos++]=demand_neighbor/K;dst[pos++]=exposure/K;
            }
            dst[pos++]=demand0;dst[pos++]=mean[m];dst[pos++]=std::log1p(global[d]);
            float um=user_sum[u]/(20+off[u+1]-off[u]);dst[pos++]=um;
            dst[pos++]=(user_week[u*7+w]-user_count[u*7+w]*um)/(8+user_count[u*7+w]);
            int k=demand0>.25f?2:demand0<-.25f?0:1;
            dst[pos++]=(user_peak[u*3+k]-peak_count[u*3+k]*um)/(8+peak_count[u*3+k]);
            dst[pos++]=std::sin(6.283185f*s/365.25f);dst[pos++]=std::cos(6.283185f*s/365.25f);dst[pos++]=w;dst[pos++]=d/365.25f;
        }
    }
}
