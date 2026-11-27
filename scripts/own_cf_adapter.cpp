#include <algorithm>
#include <cmath>
#include <cstdint>
constexpr int D=96;
extern "C" void cf_totals(int users,int nm,const int* off,const int16_t* it,const int16_t* day,
        const int8_t* r,const float* enc,float* totals,float* mean) {
    for(int u=0;u<users;++u) {
        float* out=totals+int64_t(u)*D; double sums[D]={};float sd=0;
        for(int a=off[u];a<off[u+1];++a) {
            const float* v=enc+(int64_t(r[a]-1)*nm+it[a])*D;
            for(int k=0;k<D;++k)sums[k]+=v[k];sd+=day[a];
        }
        for(int k=0;k<D;++k)out[k]=sums[k];
        mean[u]=sd/std::max(1,off[u+1]-off[u]);
    }
}
extern "C" void cf_batch(int n,int nm,int training,const int* rows,const int* qu,const int16_t* qi,
        const int16_t* qd,const int* off,const int* ord,const int16_t* it,const int16_t* day,const int8_t* r,
        const float* enc,const float* totals,const int* starts,const int* masks,const int16_t* sim,
        float* out,float* support) {
    for(int j=0;j<n;++j) {
        int row=rows[j],u=qu[row],m=qi[row],d=qd[row],lo=off[u],hi=off[u+1];
        int ml=training?starts[u]:0,mh=training?starts[u+1]:0,count=hi-lo-(mh-ml);
        float* v=out+int64_t(j)*3*D;
        std::copy_n(totals+int64_t(u)*D,D,v);
        for(int p=ml;p<mh;++p){
            int a=masks[p];const float* e=enc+(int64_t(r[a]-1)*nm+it[a])*D;
            for(int k=0;k<D;++k)v[k]-=e[k];
        }
        float* near=v+D;float* genre=v+2*D;
        for(int k=0;k<D;++k){near[k]=.20f*v[k];genre[k]=.15f*v[k];}
        int pos=std::lower_bound(ord+lo,ord+hi,d,[&](int a,int b){return day[a]<b;})-ord;
        int l=pos-1,h=pos,np=0;
        float s1=.04f*count,s2=.0225f*count,total_w=0,similar_w=0;
        while(np<64 && (l>=lo || h<hi)) {
            int a=(h>=hi || (l>=lo && d-day[ord[l]]<=day[ord[h]]-d))?ord[l--]:ord[h++];
            if(training && std::find(masks+ml,masks+mh,a)!=masks+mh)continue;
            ++np;float wt=.8f/(1+std::abs(day[a]-d)/20.f);
            s1+=wt*(.4f+wt);total_w+=wt;
            const float* e=enc+(int64_t(r[a]-1)*nm+it[a])*D;
            for(int k=0;k<D;++k)near[k]+=wt*e[k];
        }
        int step=std::max(1,(hi-lo)/112);
        for(int z=lo;z<hi;z+=step) {
            int a=ord[z];
            if(training && std::find(masks+ml,masks+mh,a)!=masks+mh)continue;
            float cos=sim[int64_t(m)*nm+it[a]]/16000.f,wt=1.8f*cos*cos;
            s2+=wt*(.3f+wt);similar_w+=wt;
            const float* e=enc+(int64_t(r[a]-1)*nm+it[a])*D;
            for(int k=0;k<D;++k)genre[k]+=wt*e[k];
        }
        float norm[3]={1.f/std::sqrt(float(std::max(1,count))),1.f/std::sqrt(std::max(s1,1e-8f)),1.f/std::sqrt(std::max(s2,1e-8f))};
        for(int t=0;t<3;++t)for(int k=0;k<D;++k)v[t*D+k]*=norm[t];
        support[3*j]=std::log1p(count)/6.f;support[3*j+1]=std::log1p(total_w)/4.f;support[3*j+2]=std::log1p(similar_w)/4.f;
    }
}
