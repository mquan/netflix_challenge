#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

constexpr int K=8,B=25,R=5;
using Vec=std::array<float,K>;
struct Session {int day,start,end;Vec evidence,post;};
static Vec posterior(Vec v) {
    float mx=*std::max_element(v.begin(),v.end()),sum=0;
    for(float &x:v)sum+=(x=std::exp(x-mx));
    for(float &x:v)x/=sum;
    return v;
}
static void sessions(int lo,int end,const int16_t* dates,const uint8_t* code,const float* likelihood,
                     const float* prior,std::vector<Session>& result,Vec& total) {
    result.clear();total.fill(0);
    for(int hi;lo<end;lo=hi) {
        hi=lo+1;while(hi<end && dates[hi]==dates[lo])++hi;
        Session s{};s.day=dates[lo];s.start=lo;s.end=hi;
        for(int j=lo;j<hi;++j){
            const float* row=likelihood+int(code[j])*K;
            for(int k=0;k<K;++k)s.evidence[k]+=row[k];
        }
        const float temp=1.f/std::sqrt(std::max(1.f,(hi-lo)/12.f));
        for(int k=0;k<K;++k){
            s.evidence[k]*=temp;
            s.post[k]=s.evidence[k]+std::log(prior[k]+2.f);
        }
        s.post=posterior(s.post);
        for(int k=0;k<K;++k)total[k]+=s.post[k];
        result.push_back(s);
    }
}
extern "C" void mix_step(int lo,int hi,const int* off,const int16_t* dates,const uint8_t* code,
                         const float* likelihood,float* priors,double* accum) {
    std::vector<Session> all;Vec total;
    for(int u=lo;u<hi;++u){
        float* prior=priors+int64_t(u)*K;
        sessions(off[u],off[u+1],dates,code,likelihood,prior,all,total);
        for(const auto& s:all)for(int j=s.start;j<s.end;++j){
            double* row=accum+int(code[j])*K;
            for(int k=0;k<K;++k)row[k]+=s.post[k];
        }
        std::copy(total.begin(),total.end(),prior);
    }
}
extern "C" void mix_query(int lo,int hi,const int* off,const int16_t* dates,const uint8_t* code,
                          const float* likelihood,const float* prob,const float* priors,const int* qs,const int* qo,
                          const int16_t* qd,const uint8_t* bins,float* out) {
    std::vector<Session> all;Vec total;
    for(int u=lo;u<hi;++u) {
        if(qs[u+1]==qs[u])continue;
        sessions(off[u],off[u+1],dates,code,likelihood,priors+int64_t(u)*K,all,total);
        for(int t=qs[u];t<qs[u+1];++t){
            int j=qo[t],day=qd[j],bin=bins[j];float* dst=out+int64_t(j)*22;
            Vec direct{},other{},near{};float nday=0,mass=0;
            for(const auto& s:all) {
                int gap=std::abs(s.day-day);
                if(!gap){direct=s.evidence;other=s.post;nday=s.end-s.start;continue;}
                if(gap>180)continue;
                float weight=1.f/(1.f+gap/9.f);
                mass+=weight;
                for(int k=0;k<K;++k)near[k]+=weight*s.post[k];
            }
            for(int mode=0;mode<4;++mode){
                Vec v;
                for(int k=0;k<K;++k){
                    float user=std::max(0.f,total[k]-other[k])+2.f;
                    float local=(near[k]+8.f*user/(std::max(0.f,float(all.size())-(nday>0))+2*K));
                    v[k]=std::log(mode==2?local:user)+(mode==0?0:mode==3?.45f:1.f)*direct[k];
                }
                v=posterior(v);
                for(int r=0;r<R;++r){
                    float sum=0;
                    for(int k=0;k<K;++k)sum+=v[k]*prob[(bin*R+r)*K+k];
                    dst[mode*R+r]=sum;
                }
            }
            dst[20]=std::log1p(nday);dst[21]=std::log1p(mass);
        }
    }
}
