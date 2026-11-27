#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

constexpr int NK=9, NF=8;
extern "C" void ordinal_context(int first,int last,const int32_t* qu,const int16_t* qd,
        const float* qr,const float* qc,const int32_t* off,const int16_t* td,
        const int8_t* stars,const float* rank,const float* cf,float* output){
    for(int j=first;j<last;++j){
        float accum[NK][NF]{},ranks[6]{};
        int u=qu[j],d=qd[j];float target=.5f*(qr[j]+qc[j]);
        for(int a=off[u];a<off[u+1];++a){
            float rp=rank[a],cp=cf[a],yp=stars[a],pred=.5f*(rp+cp);
            float dr=rp-qr[j],dc=cp-qc[j],diff=pred-target;
            float t=1.f/(1.f+std::abs(int(td[a])-d)/20.f);
            float w=1.f/(1.f+diff*diff/.10f);
            float weights[NK]={1.f,1.f/(1.f+dr*dr/.10f),1.f/(1.f+dc*dc/.10f),w,
                1.f/(1.f+diff*diff/.035f),w*t,w*float(td[a]==d),t,
                w/(1.f+std::abs((rp-cp)-(qr[j]-qc[j]))/.24f)};
            ranks[0]+=rp<=qr[j];ranks[1]+=cp<=qc[j];ranks[2]+=pred<=target;
            ranks[3]+=dr*dr;ranks[4]+=dc*dc;ranks[5]+=yp*yp;
            for(int k=0;k<NK;++k){
                float wk=weights[k];
                accum[k][0]+=wk;accum[k][1]+=wk*(yp-rp);accum[k][2]+=wk*(yp-cp);
                for(int s=1;s<=5;++s)accum[k][2+s]+=wk*(yp==s);
            }
        }
        auto out=output+int64_t(j)*(NK*NF+6);
        for(int k=0;k<NK;++k){
            float n=accum[k][0];
            out[k*NF]=std::log1p(n);
            for(int h=1;h<NF;++h)out[k*NF+h]=accum[k][h]/std::max(.001f,n);
        }
        for(int k=0;k<6;++k)out[NK*NF+k]=ranks[k]/std::max(1,off[u+1]-off[u]);
    }
}
