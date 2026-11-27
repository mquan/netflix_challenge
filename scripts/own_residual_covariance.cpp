#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

extern "C" void accumulate(int nu,int dim,int recent,const int* off,const int16_t* mi,const int16_t* day,
    const float* residual,const int* bucket,float* gram){
    constexpr int batch=384;int width=2*dim;
    std::vector<float> design(int64_t(batch)*width);
    for(int first=0;first<nu;first+=batch){
        int n=std::min(batch,nu-first);std::fill(design.begin(),design.begin()+int64_t(n)*width,0.f);
        for(int k=0;k<n;++k){
            int u=first+k;auto x=design.data()+int64_t(k)*width;int latest=0;
            if(recent)for(int j=off[u];j<off[u+1];++j)latest=std::max(latest,int(day[j]));
            for(int j=off[u];j<off[u+1];++j){
                if(recent&&latest-day[j]>120)continue;
                int g=bucket[mi[j]];x[g]+=1.f;x[dim+g]+=residual[j];
            }
            for(int g=0;g<dim;++g)if(x[g]>0){x[dim+g]/=x[g];x[g]=1;}
        }
        cblas_ssyrk(CblasRowMajor,CblasLower,CblasTrans,width,n,1.f,design.data(),width,first?1.f:0.f,gram,width);
        if(first%(batch*320)==0){std::printf("moment window %d users %d\n",recent,first);std::fflush(stdout);}
    }
}

constexpr int K=52,Q=12,F=12;
static inline float when(int a,int b){return 1.f/(1.f+std::abs(a-b)/80.f);}
static inline float cross(const float* cov,int d,int a,int b){return a==b?.04f:cov[int64_t(a)*d+b];}

extern "C" void predict(int first,int last,int dim,int mode,const int* off,const int16_t* mi,const int16_t* day,
        const float* residual,const int* bucket,const int* qo,const int* qr,const int16_t* qi,const int16_t* qd,
        const int* co,const int16_t* ci,const int16_t* cd,const float* cov,const float* shift,const float* variance,float* result){
    std::vector<std::pair<float,int>> order;
    for(int u=first;u<last;++u){
        int nq=qo[u+1]-qo[u];if(!nq||nq>Q)continue;int n=off[u+1]-off[u];if(!n)continue;
        std::array<int,Q> groups{},days{},rows{};float sums[Q][4]{};
        for(int q=0;q<nq;++q){int j=qr[qo[u]+q];groups[q]=bucket[qi[j]];days[q]=qd[j];rows[q]=j;}
        order.clear();order.reserve(n);
        for(int j=off[u];j<off[u+1];++j){
            int b=bucket[mi[j]];float best=0;
            for(int q=0;q<nq;++q){
                int g=groups[q];float t=when(day[j],days[q]);
                float sh=b==g?0.f:shift[int64_t(g)*dim+b];
                float c=cross(cov,dim,g,b);
                best=std::max(best,(std::abs(c)+.22f*std::abs(sh))*(.42f+.58f*t));
                sums[q][0]+=sh;sums[q][1]+=t*sh;sums[q][2]+=t;sums[q][3]+=c*residual[j];
            }
            order.emplace_back(best,j);
        }
        int used=std::min(n,K);std::partial_sort(order.begin(),order.begin()+used,order.end(),std::greater<>());
        std::array<int,K> g{},day0{};std::array<float,K> e{};
        for(int h=0;h<used;++h){int j=order[h].second;g[h]=bucket[mi[j]];day0[h]=day[j];e[h]=residual[j];}
        for(int local=0;local<2;++local){
            float L[K*K]{},z[K]{};
            for(float strength:{.85f,.4f,0.f}){
                bool ok=true;
                for(int a=0;a<used&&ok;++a)for(int b=0;b<=a;++b){
                    float v=a==b?std::max(.32f,variance[g[a]])+.27f:strength*cross(cov,dim,g[a],g[b])*(local?when(day0[a],day0[b]):1.f);
                    for(int h=0;h<b;++h)v-=L[a*K+h]*L[b*K+h];
                    if(a==b){if(v<=.035f){ok=false;break;}L[a*K+b]=std::sqrt(v);}
                    else L[a*K+b]=v/L[b*K+b];
                }
                if(ok)break;
            }
            for(int a=0;a<used;++a){float v=e[a];for(int h=0;h<a;++h)v-=L[a*K+h]*z[h];z[a]=v/L[a*K+a];}
            for(int a=used-1;a>=0;--a){float v=z[a];for(int h=a+1;h<used;++h)v-=L[h*K+a]*z[h];z[a]=v/L[a*K+a];}
            for(int q=0;q<nq;++q){
                float correction=0,num=0,den=0,implicit=0;
                for(int h=0;h<used;++h){
                    float w=cross(cov,dim,groups[q],g[h])*(local?when(days[q],day0[h]):1.f);
                    correction+=w*z[h];num+=w*e[h];den+=std::abs(w);
                    implicit+=(groups[q]==g[h]?0.f:shift[int64_t(groups[q])*dim+g[h]])*(local?when(days[q],day0[h]):1.f);
                }
                float* out=result+int64_t(rows[q])*2*F+mode*F;
                out[3*local]=correction;out[3*local+1]=num/(1+den);out[3*local+2]=implicit/(1+used);
                if(!local){
                    out[6]=sums[q][0]/n;out[7]=sums[q][1]/(1+sums[q][2]);out[8]=sums[q][3]/std::sqrt(float(n));
                    float im=0,weighted=0,weight=0;
                    for(int j=co[u];j<co[u+1];++j){
                        if(ci[j]==qi[rows[q]])continue;int b=bucket[ci[j]];float v=b==groups[q]?0:shift[int64_t(groups[q])*dim+b];
                        float t=when(cd[j],days[q]);im+=v;weighted+=t*v;weight+=t;
                    }
                    out[9]=im/std::max(1,co[u+1]-co[u]-1);out[10]=weighted/(1+weight);out[11]=den;
                }
            }
        }
    }
}
