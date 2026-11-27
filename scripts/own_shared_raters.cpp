#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

constexpr int H=12,Q=12,B=8,F=5;
static uint32_t mix(uint32_t v){
    v^=v>>16;v*=0x7feb352d;v^=v>>15;v*=0x846ca68b;return v^(v>>16);
}

// Index histories with random weighted minima of movie, movie/star and movie/residual tokens.
extern "C" void signatures(int nu,const int* off,const int16_t* item,const int8_t* star,
        const float* error,const float* rarity,uint64_t* index){
    for(int u=0;u<nu;++u){
        std::array<float,H> best;best.fill(1e30f);std::array<uint32_t,H> ids{};
        for(int j=off[u];j<off[u+1];++j){
            int m=item[j];int tokens[3]={m,m*5+star[j]-1,m*3+(error[j]>.48f?2:(error[j]<-.48f?0:1))};
            for(int h=0;h<H;++h){
                uint32_t a=mix(tokens[h%3]+(h+1)*2654435761U);
                float score=(a+.5f)*rarity[m];
                if(score<best[h]){best[h]=score;ids[h]=tokens[h%3];}
            }
        }
        for(int h=0;h<H;++h)index[int64_t(h)*nu+u]=(uint64_t(ids[h])<<32)|uint32_t(u);
    }
}
struct Moment{float n=0,sum=0,raw=0,w2=0,mx=0;};
extern "C" void neighbors(int from,int to,int nu,int nm,const int* off,const int16_t* item,
        const int8_t* star,const int16_t* day,const float* e,const float* mean,const float* rarity,
        const int* qo,const int* qr,const int16_t* qi,const int16_t* qd,const uint64_t* ix,
        const uint64_t* sorted,float* result){
    std::vector<int> own(nm,-1),query(nm,-1),last(nu,-1),ids;ids.reserve(400);
    for(int u=from;u<to;++u){
        int n=qo[u+1]-qo[u];if(!n||n>Q)continue;
        for(int j=off[u];j<off[u+1];++j)own[item[j]]=j;
        for(int k=0;k<n;++k)query[qi[qr[qo[u]+k]]]=k;
        ids.clear();last[u]=u;
        for(int h=0;h<H;++h){
            uint64_t token=ix[int64_t(h)*nu+u]>>32;
            auto start=sorted+int64_t(h)*nu;
            auto a=std::lower_bound(start,start+nu,token<<32);
            auto b=std::lower_bound(a,start+nu,(token+1)<<32);
            int count=int(b-a),cap=24;
            for(int k=0;k<std::min(count,cap);++k){
                int pick=count<=cap?k:int((int64_t(k)*count+mix(uint32_t(u*H+h+k))%count)/cap);
                int v=int(uint32_t(a[pick]));if(last[v]==u)continue;last[v]=u;ids.push_back(v);
            }
        }
        Moment sums[Q][B]{};
        for(int v:ids){
            float raw=0,ar=0,br=0,ex=0,ae=0,be=0,agre=0,nearby=0;
            int shared=0,hits=0;std::array<int,Q> obs,slot;
            for(int j=off[v];j<off[v+1];++j){
                int m=item[j],a=own[m],k=query[m];if(k>=0){obs[hits]=j;slot[hits++]=k;}
                if(a<0)continue;
                float x=star[a]-mean[m],y=star[j]-mean[m];
                float w=std::min(3.f,1.f/rarity[m]);raw+=w*x*y;ar+=w*x*x;br+=w*y*y;
                x=e[a];y=e[j];ex+=w*x*y;ae+=w*x*x;be+=w*y*y;
                agre+=(star[a]==star[j]);nearby+=1.f/(1.f+std::abs(int(day[a])-day[j])/90.f);++shared;
            }
            if(!hits||shared<2)continue;
            float r=raw/std::sqrt((ar+10)*(br+10)),c=ex/std::sqrt((ae+14)*(be+14));
            float agreement=(agre+2)/(shared+6),timing=(nearby+.3f)/(shared+1);
            float jac=shared/std::sqrt(float((off[u+1]-off[u])*(off[v+1]-off[v]))+1);
            for(int h=0;h<hits;++h){
                int k=slot[h],j=obs[h];float t=1.f/(1.f+std::abs(day[j]-qd[qr[qo[u]+k]])/120.f);
                float w[B]={1.f,std::exp(5*r),std::exp(8*c),std::max(0.f,c)*std::exp(8*c),
                       std::min(0.f,c)*std::exp(-6*c),std::exp(6*(r+c))*t,
                       std::exp(6*(agreement-.4f)+3*jac),std::exp(5*r+3*timing)*t};
                for(int b=0;b<B;++b){auto& s=sums[k][b];s.n+=std::abs(w[b]);s.sum+=w[b]*e[j];
                    s.raw+=w[b]*(star[j]-mean[item[j]]);s.w2+=w[b]*w[b];s.mx=std::max(s.mx,std::abs(w[b]));}
            }
        }
        for(int k=0;k<n;++k){auto out=result+int64_t(qr[qo[u]+k])*B*F;
            for(int b=0;b<B;++b){auto& s=sums[k][b];*out++=std::log1p(s.n);*out++=s.sum/(1+s.n);
              *out++=s.raw/(1+s.n);*out++=s.mx/(1+s.n);*out++=s.n*s.n/(1+s.w2);}}
        for(int j=off[u];j<off[u+1];++j)own[item[j]]=-1;
        for(int k=0;k<n;++k)query[qi[qr[qo[u]+k]]]=-1;
    }
}
