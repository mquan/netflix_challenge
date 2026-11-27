#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

struct Moment{
    float n{},e{},r{},hi{},low{};
    void add(float w,float err,int rating){n+=w;e+=w*err;r+=w*rating;hi+=w*(rating>=4);low+=w*(rating<=2);}
};
static uint32_t mix(uint32_t a){a^=a>>16;a*=0x7feb352du;a^=a>>15;a*=0x846ca68bu;return a^(a>>16);}
static int relation(int d){return d==0?0:(d<0?(d>=-20?1:3):(d<=20?2:4));}
extern "C" void transitions(int first,int last,int nm,const int* movie_off,const int* by_movie,const int* owner,
    const int* off,const int16_t* items,const int16_t* day,const int8_t* stars,const float* residual,
    const int* qmovies,const int* qrows,const int* qu,const int16_t* qdate,const int* coff,const int16_t* ci,
    const int16_t* cd,const int8_t* seen,const int* cluster,float* result){
    constexpr int C=96,S=5,V=8;
    std::vector<Moment> term((nm+C+1)*S);
    for(int m=first;m<last;++m){
        if(qmovies[m]==qmovies[m+1])continue;
        std::fill(term.begin(),term.end(),Moment{});
        for(int p=movie_off[m];p<movie_off[m+1];++p){
            int at=by_movie[p],u=owner[at],begin=off[u],end=off[u+1],d=day[at],n=end-begin;
            auto dl=std::lower_bound(day+begin,day+at+1,d)-day,dr=std::upper_bound(day+at,day+end,d)-day;
            float percentile=float(at-begin+1)/n,w0=.28f+.72f*percentile*percentile;
            int r=stars[at];float e=residual[at];
            auto add=[&](int j,float weight){
                if(j==at)return;
                int i=items[j],type=relation(int(day[j])-d);
                term[i*S+type].add(weight,e,r);term[(nm+cluster[i])*S+type].add(weight,e,r);
                term[(nm+C)*S+type].add(weight,e,r);
            };
            // Stratified draws include every title in short sessions and sample long sessions uniformly.
            auto sample=[&](int a,int b,int cap,float w,uint32_t seed){
                int len=b-a;if(len<=0)return;int k=std::min(len,cap);
                for(int t=0;t<k;++t){
                    int lo=t*len/k,hi=(t+1)*len/k;
                    int j=a+lo+mix(seed+t)%uint32_t(hi-lo);
                    add(j,w);
                }
            };
            sample(dl,dr,12,w0,mix(at));
            // Match the handful of earlier and later choices in a nine-title terminal cohort.
            // Expand the closest edge to include its entire tied session before sampling it.
            if(dl>begin){
                int a=std::max(begin,int(dl)-12);
                a=std::lower_bound(day+begin,day+a+1,day[a])-day;
                sample(a,dl,9,w0,mix(at+131));
            }
            if(dr<end){
                int b=std::min(end,int(dr)+12);
                b=std::upper_bound(day+b-1,day+end,day[b-1])-day;
                sample(dr,b,9,w0,mix(at+271));
            }
        }
        for(int p=qmovies[m];p<qmovies[m+1];++p){
            int row=qrows[p],u=qu[row],d=qdate[row];float sums[7][V]{},norm[7]{};
            for(int k=coff[u];k<coff[u+1];++k){
                int i=ci[k];if(i==m)continue;
                int gap=int(cd[k])-d,t=relation(gap);
                auto a=term[i*S+t],b=term[(nm+cluster[i])*S+t],g=term[(nm+C)*S+t];
                float gn=std::max(.01f,g.n),eg=g.e/gn,rg=g.r/gn,hg=g.hi/gn,lg=g.low/gn;
                float bc=(b.e-b.n*eg)/(b.n+35.f);
                float ec=(a.e-a.n*eg+18.f*bc)/(a.n+18.f);
                float er=(a.r-a.n*rg+18.f*(b.r-b.n*rg)/(b.n+35.f))/(a.n+18.f);
                float eh=(a.hi-a.n*hg+18.f*(b.hi-b.n*hg)/(b.n+35.f))/(a.n+18.f);
                float el=(a.low-a.n*lg+18.f*(b.low-b.n*lg)/(b.n+35.f))/(a.n+18.f);
                Moment all{};for(int j=0;j<S;++j){auto v=term[i*S+j];all.n+=v.n;all.e+=v.e;}
                float values[V]={ec,bc,er,eh,el,all.e/(all.n+80),std::log1p(a.n),eg};
                float weights[7]={1.f,1.f/(1.f+std::abs(gap)/15.f),float(!seen[k]),float(seen[k]),
                                  float(gap==0),float(gap<0),float(gap>0)};
                for(int j=0;j<7;++j){norm[j]+=weights[j];for(int z=0;z<V;++z)sums[j][z]+=weights[j]*values[z];}
            }
            auto dest=result+int64_t(row)*63;
            for(int j=0;j<7;++j){dest[j*9]=norm[j];for(int z=0;z<V;++z)dest[j*9+1+z]=sums[j][z]/(.1f+norm[j]);}
        }
    }
}
