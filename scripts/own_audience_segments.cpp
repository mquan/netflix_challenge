#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>
constexpr int D=12,G=64,B=10,H=3;
extern "C" void collect(int n,const int* users,const int16_t* items,const int16_t* days,const float* geo,const int16_t* year,const float* popularity,float* result){
    for(int j=0;j<n;++j){int u=users[j],m=items[j],d=days[j];float* dst=result+int64_t(u)*(D+12);
        for(int k=0;k<D;++k)dst[k]+=geo[m*D+k];
        float lt=(d-1400.f)/600.f,age=std::clamp((2000.f+d/365.f-year[m])/25.f,0.f,4.f),pop=popularity[m];
        float v[]={lt,lt*lt,age,age*age,pop,pop*pop,float(age<.13f),float(age>.8f),float(pop<.4f),lt*age,lt*pop,1.f};
        for(int k=0;k<12;++k)dst[D+k]+=v[k];
    }
}
struct Cell{int n=0;float err=0,star=0,five=0;};
extern "C" void audience_cells(int movies,const int* mo,const int* row,const int* users,const int8_t* rating,const float* err,
        const int* qo,const int* qr,const int* qu,const float* native,const int8_t* cluster,const float* mix,float* out){
    std::array<Cell,G*B*H> cells;
    auto bin=[](float x){return std::clamp(int((x-1)*2.5f),0,B-1);};
    for(int m=0;m<movies;++m){
        cells.fill(Cell{});
        for(int k=mo[m];k<mo[m+1];++k){int j=row[k],u=users[j],r=rating[j],b=bin(r-err[j]);
            for(int h=0;h<H;++h){auto& c=cells[(h*G+cluster[(u*H+h)*2])*B+b];++c.n;c.err+=err[j];c.star+=r-3.5f;c.five+=r==5;}}
        for(int k=qo[m];k<qo[m+1];++k){int j=qr[k],u=qu[j],b=bin(native[j]);float* p=out+int64_t(j)*24;
            for(int h=0;h<H;++h)for(int level=0;level<2;++level){
                float n=0,e=0,s=0,f=0;
                for(int t=0;t<2;++t){int l=(u*H+h)*2+t;auto c=cells.data()+(h*G+cluster[l])*B;
                    for(int v=0;v<B;++v){float w=mix[l]/(1.f+(level==0?1.5f:.04f)*(b-v)*(b-v));
                        n+=w*c[v].n;e+=w*c[v].err;s+=w*c[v].star;f+=w*c[v].five;}}
                auto z=p+8*h+4*level;z[0]=e/(n+25);z[1]=s/(n+25);z[2]=f/(n+25);z[3]=std::log1p(n);
            }}
    }
}
