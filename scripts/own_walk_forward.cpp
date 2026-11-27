#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

constexpr int H=320,D=96,W=24,Q=9,ROUNDS=10,WIDTH=64;
struct Past {int item,day,lag,boundary;float mu,e,high,low,raw;};
extern "C" void walk_forward(int lo,int hi,int nm,float avg,float sd,const int64_t* off,const int16_t* item,const int16_t* day,
    const int8_t* star,const int16_t* boundary,const int* qoff,const int* order,const int16_t* qi,
    const int16_t* qd,const float* pred,const float* enc,const float* proj,const float* hb,const float* dec,
    const float* bias,const float* ub,const float* drift,const float* geom,const float* film,float* features){
    std::vector<Past> history;std::vector<int> equal,picked;
    for(int u=lo;u<hi;++u){
        if(qoff[u]==qoff[u+1])continue;
        std::mt19937 rng(20261337u^unsigned(u*2654435761u));
        const int start=off[u],end=off[u+1],bound=boundary[u];
        int stop=std::lower_bound(day+start,day+end,bound)-day;
        // A user with a single session can still support a genuine same-day holdout.
        if(stop-start<3)stop=std::upper_bound(day+start,day+end,bound)-day;
        history.clear();double total[D]={},ds=0;
        for(int j=start;j<stop;++j){auto e=enc+(int64_t(star[j]-1)*nm+item[j])*D;
            for(int k=0;k<D;++k)total[k]+=e[k];ds+=day[j];}
        for(int ep=0;ep<ROUNDS && stop-start>=3;++ep){
            int n=stop-start,k=std::min(Q,std::max(1,n-2));int first=stop-k,cut=day[first];
            int a=std::lower_bound(day+start,day+first+1,cut)-day;
            int b=std::upper_bound(day+first,day+stop,cut)-day;
            equal.resize(b-a);std::iota(equal.begin(),equal.end(),a);std::shuffle(equal.begin(),equal.end(),rng);
            picked.assign(equal.begin(),equal.begin()+k-(stop-b));
            for(int j=b;j<stop;++j)picked.push_back(j);
            float ctx[D],hidden[H];double sumday=ds;
            for(int t=0;t<D;++t)ctx[t]=total[t];
            for(int j:picked){auto v=enc+(int64_t(star[j]-1)*nm+item[j])*D;for(int t=0;t<D;++t)ctx[t]-=v[t];sumday-=day[j];}
            float inv=1/std::sqrt(float(n-k));float md=sumday/(n-k);
            cblas_sgemv(CblasRowMajor,CblasNoTrans,H,D,inv,proj,D,ctx,1,0.f,hidden,1);
            for(int j=0;j<H;++j)hidden[j]+=hb[j];const int dim=H;vvtanhf(hidden,hidden,&dim);
            for(int j:picked){
                int m=item[j],d=day[j],r=star[j];float rel=(d-md)/sd;
                float dev=std::copysign(std::pow(std::abs(rel),.4f),rel),z[5];
                for(int h=0;h<5;++h)z[h]=cblas_sdot(H,hidden,1,dec+(int64_t(m)*5+h)*H,1)+bias[m*5+h]+ub[u*5+h]+drift[u*5+h]*dev;
                float mx=*std::max_element(z,z+5),s=0,mu=0;
                for(int h=0;h<5;++h){z[h]=std::exp(z[h]-mx);s+=z[h];mu+=(h+1)*z[h];}mu/=s;
                history.push_back({m,d,ep,int(d==cut),mu,r-mu,float(r>=4)-(z[3]+z[4])/s,
                                  float(r<=2)-(z[0]+z[1])/s,r-film[m]});
            }
            for(int j=a;j<stop;++j){auto e=enc+(int64_t(star[j]-1)*nm+item[j])*D;
                 for(int k=0;k<D;++k)total[k]-=e[k];ds-=day[j];}
            stop=a;
        }
        for(int j=qoff[u];j<qoff[u+1];++j){
            int row=order[j],m=qi[row],d=qd[row];float p=pred[row];float sums[8][7]{};
            float recent=0,previous=0,nearest=0,nearest_w=-1;
            for(const auto& a:history){
                float sim=cblas_sdot(W,geom+m*W,1,geom+a.item*W,1);
                float pos=std::max(0.f,sim),s2=pos*pos;
                float near=1.f/(1.f+std::abs(d-a.day)/50.f);float match=std::exp(-std::abs(p-a.mu)/.65f);
                float terminal=(a.boundary==int(d==bound))?1.f:.22f;
                float weight[8]={1.f,near,s2,s2*near,match,match*terminal/(1.f+a.lag),terminal*near,s2*match*terminal};
                for(int l=0;l<8;++l){
                    float w=weight[l];auto v=sums[l];v[0]+=w;v[1]+=w*a.e;v[2]+=w*a.e*a.e;
                    v[3]+=w*a.high;v[4]+=w*a.low;v[5]+=w*a.raw;v[6]+=w*(a.mu-p);
                }
                if(a.lag==0)recent+=a.e;
                if(a.lag==1)previous+=a.e;
                if(weight[7]>nearest_w){nearest_w=weight[7];nearest=a.e;}
            }
            auto dest=features+int64_t(row)*WIDTH;
            for(int l=0;l<8;++l){dest[l*7]=std::log1p(sums[l][0]);
                for(int h=1;h<7;++h)dest[l*7+h]=sums[l][h]/(.8f+sums[l][0]);}
            dest[56]=recent/Q;dest[57]=previous/Q;dest[58]=nearest;dest[59]=std::max(0.f,nearest_w);
            dest[60]=float(d-bound);dest[61]=float(history.size());dest[62]=p;dest[63]=film[m];
        }
    }
}
