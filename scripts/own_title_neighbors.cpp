#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
constexpr int K=40,G=3,W=12;
struct Counts{int n=0;float error=0;};
extern "C" void title_neighbors(int nu,int nm,const int* off,const int16_t* movie,const int16_t* day,const int8_t* star,
        const float* err,const int* nb,const int8_t* band,const float* affinity,const int16_t* year,const int* qoff,const int* qr,
        const int16_t* qi,const int16_t* qd,const int* ioff,const int16_t* implicit,float* out){
    std::vector<int> position(nm,-1),visible(nm,-1),peers(nm,-1);
    std::vector<std::array<Counts,5>> tally(int64_t(nm)*K);std::vector<Counts> pooled(nm*K);
    for(int u=0;u<nu;++u){
        for(int j=off[u];j<off[u+1];++j){int m=movie[j];position[m]=j;visible[m]=u;}
        for(int j=off[u];j<off[u+1];++j){
            int m=movie[j];for(int h=0;h<K;++h){int id=m*K+h,l=nb[id];if(l<0)break;if(visible[l]!=u)continue;
                auto& c=tally[id][star[position[l]]-1];++c.n;c.error+=err[j];auto& all=pooled[id];++all.n;all.error+=err[j];}
        }
    }
    std::fill(visible.begin(),visible.end(),-1);
    for(int u=0;u<nu;++u){
        for(int j=off[u];j<off[u+1];++j){int m=movie[j];position[m]=j;visible[m]=u;}
        for(int j=ioff[u];j<ioff[u+1];++j)peers[implicit[j]]=u;
        for(int hq=qoff[u];hq<qoff[u+1];++hq){
            int j=qr[hq],m=qi[j];float* dst=out+int64_t(j)*G*W;std::fill(dst,dst+G*W,0.f);
            for(int h=0;h<K;++h){
                int id=m*K+h,l=nb[id];if(l<0)break;int g=band[id];float* z=dst+g*W;float w=affinity[id];
                if(peers[l]==u)z[10]+=w;
                if(visible[l]!=u)continue;int v=position[l],r=star[v];float t=1.f/(1.f+std::abs(int(day[v])-qd[j])/40.f);
                auto& c=tally[id][r-1];auto& all=pooled[id];float anchor=all.error/(all.n+90);
                float update=(c.error-std::min(float(c.n),float(all.n))*anchor)/(45+c.n);
                z[0]+=w;z[1]+=w*(r-3.5f);z[2]+=w*err[v];z[3]+=w*err[v]*t;z[4]+=w*(r==5);
                z[5]+=w*(r<=2);z[6]+=w*(year[l]<year[m])*err[v];z[7]+=w*update;z[8]+=w*anchor;
                z[9]+=w*t;z[11]=std::max(z[11],w);
            }
            for(int g=0;g<G;++g){auto z=dst+g*W;float norm=1.f/(.5f+z[0]);for(int h=1;h<=9;++h)z[h]*=norm;
                z[0]=std::log1p(z[0]);z[10]=std::log1p(z[10]);}
        }
    }
}
