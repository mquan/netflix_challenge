#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>
namespace {
constexpr int B=24,T=48,C=7;
using Cell=std::array<float,C>;
int bin(float x){return std::clamp(int((x-1)*6),0,B-1);}
void deposit(Cell& c,int r,float pred,float w){
    c[r-1]+=w;c[5]+=w*(r-pred);c[6]+=w*(r-pred)*(r-pred);
}
void save(const Cell& c,float* out){
    float n=0;for(int r=0;r<5;++r)n+=c[r];float den=1/(n+6);
    for(int r=0;r<7;++r)out[r]=c[r]*den;
    out[7]=std::log1p(n);
}
}
extern "C" void reliability(int nu,int nm,const int* off,const int16_t* item,const int16_t* day,
    const int8_t* star,const float* residual,const int* usoff,const int* usidx,const int16_t* qi,const int16_t* qd,
    const float* query,const int* movieoff,const int* movierow,const int* qmoff,const int* qmrow,float* out){
    std::array<float,24> kscore;std::array<float,4096> kdate;
    for(int j=0;j<24;++j)kscore[j]=std::exp(-.5f*j*j/2.25f);
    for(int j=0;j<4096;++j)kdate[j]=1/(1+float(j)/35);
    for(int u=0;u<nu;++u){
        for(int k=usoff[u];k<usoff[u+1];++k){
            int j=usidx[k],b=bin(query[j]);std::array<Cell,4> vals{};
            for(int l=off[u];l<off[u+1];++l){
                float p=star[l]-residual[l],s=kscore[std::abs(bin(p)-b)],t=kdate[std::min(4095,std::abs(day[l]-qd[j]))];
                deposit(vals[0],star[l],p,s);deposit(vals[1],star[l],p,s*t);deposit(vals[2],star[l],p,t);
                if(day[l]==qd[j])deposit(vals[3],star[l],p,.3f+.7f*s);
            }
            for(int t=0;t<4;++t)save(vals[t],out+int64_t(j)*64+t*8);
        }
    }
    std::vector<Cell> cells(T*B);
    for(int m=0;m<nm;++m){
        std::fill(cells.begin(),cells.end(),Cell{});std::array<Cell,B> global{};
        for(int k=movieoff[m];k<movieoff[m+1];++k){
            int j=movierow[k],r=star[j];float p=r-residual[j];int b=bin(p),t=std::clamp(int(day[j])/56,0,T-1);
            deposit(cells[t*B+b],r,p,1);deposit(global[b],r,p,1);
        }
        std::array<std::array<Cell,2>,B> all{};
        for(int b=0;b<B;++b)for(int v=0;v<B;++v)for(int c=0;c<C;++c){
            float w=kscore[std::abs(b-v)];all[b][0][c]+=w*global[v][c];all[b][1][c]+=std::sqrt(w)*global[v][c];}
        std::vector<std::array<Cell,2>> local(T*B);
        std::vector<bool> ready(T*B);
        for(int k=qmoff[m];k<qmoff[m+1];++k){
            int j=qmrow[k],b=bin(query[j]),t=std::clamp(int(qd[j])/56,0,T-1),i=t*B+b;
            if(!ready[i]){ready[i]=true;
                for(int time=std::max(0,t-4);time<=std::min(T-1,t+4);++time)for(int v=0;v<B;++v){
                    float w=kscore[std::abs(b-v)],s=1.f/(1+std::abs(time-t));
                    for(int c=0;c<C;++c){float z=cells[time*B+v][c];
                        local[i][0][c]+=w*s*s*z;local[i][1][c]+=s*s*z;}
                }}
            float* dst=out+int64_t(j)*64+32;
            save(all[b][0],dst);save(all[b][1],dst+8);save(local[i][0],dst+16);save(local[i][1],dst+24);
        }
    }
}
