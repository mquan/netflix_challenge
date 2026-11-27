#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

constexpr int D=28,H=6,STATIC=20;
static inline float drift(int day,float mean){return std::clamp((day-mean)/365.f,-2.f,2.f);}
static const auto sigmoid=[] {
    std::array<float,4097> table{};
    for(int i=0;i<=4096;++i)table[i]=1/(1+std::exp(-(i-2048)/160.f));
    return table;
}();
static inline float prob(float z){return sigmoid[std::clamp(int(std::round(z*160))+2048,0,4096)];}
static void target(int user,int movie,int star,const float* up,const float* ip,float rank,float cf,float* y){
    for(int k=0;k<4;++k)y[k]=1.8f*((star>k+1)-prob(up[user*4+k]+ip[movie*4+k]));
    y[4]=.8f*(star-rank);y[5]=.8f*(star-cf);
}
static void solve(int n,int h,const float* x,const float* y,float pen,const float* prior,float* out){
    float g[D*D],b[D*H],bt[D*H];
    cblas_sgemm(CblasRowMajor,CblasTrans,CblasNoTrans,D,D,n,1,x,D,x,D,0,g,D);
    cblas_sgemm(CblasRowMajor,CblasTrans,CblasNoTrans,D,h,n,1,x,D,y,h,0,b,h);
    for(int j=0;j<D;++j){
        g[j*D+j]+=pen;
        for(int k=0;k<h;++k)bt[k*D+j]=b[j*h+k]+(prior?pen*prior[k*D+j]:0);
    }
    char part='U';int dims=D,info=0;
    sposv_(&part,&dims,&h,g,&dims,bt,&dims,&info);
    if(info)std::fill_n(out,D*h,NAN);else std::copy_n(bt,D*h,out);
}
extern "C" void ord_users(int lo,int hi,const int* off,const int16_t* item,const int16_t* day,const int8_t* r,
    const float* md,const float* up,const float* ip,const float* rank,const float* cf,
    const float* film,const float* prior,float* users){
    std::vector<float>x,y;
    for(int u=lo;u<hi;++u){
        int n=off[u+1]-off[u];if(!n)continue;
        x.resize(int64_t(n)*H*D);y.resize(n*H);
        for(int p=0;p<n;++p){
            int row=off[u]+p,m=item[row];float dt=drift(day[row],md[u]);
            target(u,m,r[row],up,ip,rank[row],cf[row],y.data()+p*H);
            const float* v=film+int64_t(m)*H*D;
            float* dest=x.data()+int64_t(p)*H*D;
            for(int k=0;k<H;++k){
                std::copy_n(v+k*D,STATIC,dest+k*D);
                for(int j=STATIC;j<D;++j)dest[k*D+j]=dt*v[k*D+j];
            }
        }
        solve(n*H,1,x.data(),y.data(),5.f+.12f*std::sqrt(float(n)),prior+int64_t(u)*D,users+int64_t(u)*D);
    }
}
extern "C" void ord_films(int lo,int hi,const int* off,const int* order,const int* user,const int16_t* day,
    const int8_t* r,const float* md,const float* up,const float* ip,const float* rank,const float* cf,
    const float* users,float* film){
    std::vector<float>x,y;
    for(int m=lo;m<hi;++m){
        int n=off[m+1]-off[m];if(!n)continue;
        x.resize(int64_t(n)*D);y.resize(int64_t(n)*H);
        for(int j=0;j<n;++j){
            int row=order[off[m]+j],u=user[row];float dt=drift(day[row],md[u]);
            target(u,m,r[row],up,ip,rank[row],cf[row],y.data()+j*H);
            const float* v=users+int64_t(u)*D;float* dest=x.data()+j*D;
            std::copy_n(v,STATIC,dest);
            for(int k=STATIC;k<D;++k)dest[k]=dt*v[k];
        }
        solve(n,H,x.data(),y.data(),12.f+.25f*std::sqrt(float(n)),nullptr,film+int64_t(m)*H*D);
    }
}
extern "C" void ord_prior(int lo,int hi,const int* off,const int16_t* item,const int8_t* r,const float* geo,float* prior){
    for(int u=lo;u<hi;++u){
        float* dest=prior+int64_t(u)*D;int n=off[u+1]-off[u];
        for(int j=off[u];j<off[u+1];++j){
            const float* v=geo+int64_t(item[j])*D;
            for(int k=0;k<STATIC;k++)dest[k]+=v[k]*(r[j]-3.6f);
        }
        for(int k=0;k<STATIC;++k)dest[k]*=.16f/std::sqrt(15.f+n);
    }
}
extern "C" void ord_predict(int n,const int* user,const int16_t* item,const int16_t* day,const float* md,
        const float* up,const float* ip,const float* users,const float* film,float* result){
    for(int j=0;j<n;++j){
        int u=user[j],m=item[j];float dt=drift(day[j],md[u]);
        float tot=1,ss=0;float* out=result+int64_t(j)*8;
        for(int h=0;h<H;++h){
            const float* a=users+int64_t(u)*D;const float* b=film+(int64_t(m)*H+h)*D;
            float z=0;
            for(int k=0;k<STATIC;++k)z+=a[k]*b[k];
            for(int k=STATIC;k<D;++k)z+=dt*a[k]*b[k];
            out[h]=h<4?std::clamp(prob(up[u*4+h]+ip[m*4+h])+z/1.8f,0.f,1.f):z/.8f;
            if(h<4){tot+=out[h];ss+=out[h]*(1-out[h]);}
        }
        out[6]=tot;out[7]=ss;
    }
}
