#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

constexpr int D=96,Q=9,L=4;
extern "C" void suffix_cf(int batch,int seed,int train,int nm,const int* us,const int64_t* off,
        const int16_t* it,const int16_t* day,const int8_t* rating,const int16_t* boundary,
        const int* grid,const int16_t* fi,const int16_t* fd,const int8_t* observed,const float* enc,
        int* movie,float* dates,int* anchor,int* target,float* context,float* norms,float* stat){
    std::mt19937 rng(seed);std::uniform_real_distribution<float> coin(0,1);
    std::vector<int> tie,cohort;std::vector<int8_t> marked;
    for(int j=0;j<batch;++j){
        int u=us[j],lo=off[u],end=off[u+1],stop=end;
        const int* g=grid+int64_t(u)*Q;
        int nco=0,nvisible=0;
        for(;nco<Q && g[nco]>=0;++nco)nvisible+=observed[g[nco]]>0;
        int cutoff=boundary[u];marked.assign(end-lo,0);
        if(train){
            int limit=std::lower_bound(day+lo,day+end,cutoff)-day;
            if(limit-lo<3)limit=end;
            int span=limit-lo;if(span<2 || nco==0)continue;
            int pick=limit-1;
            if(span>12 && coin(rng)<.78f){
                float lower=coin(rng)<.62f?.50f:.13f;
                pick=lo+std::clamp(int(span*(lower+(1-lower)*coin(rng))),10,span-1);
            }
            stop=std::upper_bound(day+pick,day+limit,day[pick])-day;
            int k=std::min(nco,stop-lo-1);int first=stop-k;cutoff=day[first];
            int a=std::lower_bound(day+lo,day+first+1,cutoff)-day;
            int b=std::upper_bound(day+first,day+stop,cutoff)-day;
            tie.resize(b-a);std::iota(tie.begin(),tie.end(),a);std::shuffle(tie.begin(),tie.end(),rng);
            cohort.assign(tie.begin(),tie.begin()+k-(stop-b));
            for(int t=b;t<stop;++t)cohort.push_back(t);
            std::shuffle(cohort.begin(),cohort.end(),rng);
            int vis=std::clamp(int(std::round(float(nvisible)*k/nco)),0,k-1);
            for(int l=0;l<k;++l){
                int r=cohort[l],pos=j*Q+l;
                movie[pos]=it[r];dates[pos]=day[r];
                if(l<vis){anchor[pos]=rating[r];marked[r-lo]=2;}
                else{target[pos]=rating[r]-1;marked[r-lo]=1;}
            }
        }else{
            for(int l=0;l<nco;++l){
                int row=g[l],pos=j*Q+l;movie[pos]=fi[row];dates[pos]=fd[row];anchor[pos]=observed[row];
                if(anchor[pos]){
                    auto a=std::lower_bound(day+lo,day+end,fd[row])-day;
                    for(int p=a;p<end&&day[p]==fd[row];++p)if(it[p]==fi[row]){marked[p-lo]=2;break;}
                }
            }
        }
        float* ctx=context+int64_t(j)*L*D;double sumday=0;
        float* norm=norms+j*L;int cnt=0,last=0,previous=0,nat=0;
        for(int r=lo;r<stop;++r){
            if(marked[r-lo]==1)continue;
            const float* v=enc+(int64_t(rating[r]-1)*nm+it[r])*D;
            int gap=std::max(0,cutoff-int(day[r]));
            float w0=1.f,w1=marked[r-lo]!=2,w2=w1/(1.f+gap/39.f),w3=w1*(gap>45);
            float w[L]={w0,w1,w2,w3};
            for(int l=0;l<L;++l){
                norm[l]+=w[l]*w[l];float* dest=ctx+l*D;
                for(int h=0;h<D;++h)dest[h]+=w[l]*v[h];
            }
            ++cnt;sumday+=day[r];last=std::max(last,int(day[r]));
            if(marked[r-lo]!=2)previous=std::max(previous,int(day[r]));
            nat+=day[r]==cutoff && marked[r-lo]!=2;
        }
        for(int l=0;l<L;++l){
            float inv=1/std::sqrt(std::max(1.f,norm[l]));
            for(int h=0;h<D;++h)ctx[l*D+h]*=inv;
        }
        float* s=stat+j*6;
        s[0]=cnt?float(sumday/cnt):0;s[1]=cutoff;s[2]=last;s[3]=previous;s[4]=std::log1p(float(nat));s[5]=cnt;
    }
}
