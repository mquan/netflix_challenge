#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>
namespace series {
constexpr int K=20,C=4,D=K*C,H=320;
struct Feature{int index;float x;};
static float leave_both(int u,int j,int k,int count,const int16_t* items,const int16_t* days,const int8_t* stars,
        float sd,const float* dates,const float* total,const float* ep,const float* op,const float* eb,const float* b,
        const float* ub,const float* drift){
    int n=std::max(1,count-2),m=items[k];float h[H],scale=1/std::sqrt(float(n));
    auto t=total+int64_t(u)*H;auto a=ep+(int64_t(items[j])*5+stars[j]-1)*H,v=ep+(int64_t(m)*5+stars[k]-1)*H;
    for(int z=0;z<H;++z)h[z]=(t[z]-a[z]-v[z])*scale+eb[z];int dim=H;vvtanhf(h,h,&dim);
    float mean=(dates[u]*count-days[j]-days[k])/n,rel=(days[k]-mean)/sd,dev=std::copysign(std::pow(std::abs(rel),.4f),rel);
    float logit[5];for(int r=0;r<5;++r)logit[r]=cblas_sdot(H,h,1,op+(int64_t(m)*5+r)*H,1)+b[m*5+r]+ub[u*5+r]+dev*drift[u*5+r];
    float mx=*std::max_element(logit,logit+5),den=0,mu=0;for(int r=0;r<5;++r){float p=std::exp(logit[r]-mx);den+=p;mu+=(r+1)*p;}
    return stars[k]-mu/den;
}
}
extern "C" void series_ridge(int nu,int nm,int nq,float sd,const int* off,const int16_t* item,const int16_t* day,
    const int8_t* rating,const float* residual,const int* qoff,const int* qr,const int16_t* qi,const int16_t* qd,
    const int* nb,const float* strength,const float* md,const float* total,const float* ep,const float* op,
    const float* eb,const float* b,const float* ub,const float* drift,float* out){
    using namespace series;std::vector<float> gram(int64_t(nm)*D*D),rhs(int64_t(nm)*D),fit(int64_t(nm)*D*3);
    std::vector<int> stamp(nm,-1),pos(nm);
    for(int phase=0;phase<2;++phase){
        std::fill(stamp.begin(),stamp.end(),-1);
        for(int u=0;u<nu;++u){
            for(int j=off[u];j<off[u+1];++j){stamp[item[j]]=u;pos[item[j]]=j;}
            int lo=phase?qoff[u]:off[u],hi=phase?qoff[u+1]:off[u+1];
            for(int g=lo;g<hi;++g){int j=phase?qr[g]:g,m=phase?qi[j]:item[j],d=phase?qd[j]:day[j];
                Feature features[D];int n=0;float present=0;int hits=0;
                for(int s=0;s<K;++s){int other=nb[m*K+s];if(other<0||stamp[other]!=u)continue;
                    int k=pos[other];float w=strength[m*K+s],dt=1.f/(1+std::abs(int(day[k])-d)/40.f);
                    float e=phase?residual[k]:leave_both(u,j,k,off[u+1]-off[u],item,day,rating,sd,md,total,ep,op,eb,b,ub,drift);
                    features[n++]={s*C,w};features[n++]={s*C+1,w*e};features[n++]={s*C+2,w*(rating[k]-3.5f)};
                    features[n++]={s*C+3,w*dt*e};present+=w;++hits;
                }
                if(!n)continue;
                if(!phase){
                    auto G=gram.data()+int64_t(m)*D*D;auto target=rhs.data()+int64_t(m)*D;
                    for(int l=0;l<n;++l){auto a=features[l];target[a.index]+=a.x*residual[j];auto row=G+a.index*D;
                        for(int h=0;h<n;++h)row[features[h].index]+=a.x*features[h].x;}
                }else{
                    auto dst=out+int64_t(j)*5;for(int p=0;p<3;++p){auto w=fit.data()+(int64_t(m)*3+p)*D;
                        for(int h=0;h<n;++h)dst[p]+=features[h].x*w[features[h].index];}
                    dst[3]=std::log1p(present);dst[4]=hits;
                }
            }
        }
        if(!phase){
            std::vector<float> buff(D*D);float pens[]={8.f,40.f,200.f};char side='L';int dim=D,nrhs=1,info=0;
            for(int m=0;m<nm;++m)for(int p=0;p<3;++p){
                auto src=gram.data()+int64_t(m)*D*D;std::copy(src,src+D*D,buff.begin());for(int h=0;h<D;++h)buff[h*(D+1)]+=pens[p];
                auto ans=fit.data()+(int64_t(m)*3+p)*D;std::copy_n(rhs.data()+int64_t(m)*D,D,ans);
                sposv_(&side,&dim,&nrhs,buff.data(),&dim,ans,&dim,&info);
                if(info)std::fill_n(ans,D,0.f);
            }
        }
    }
}
