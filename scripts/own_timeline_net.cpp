#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

constexpr int G=12,F=6+G*2,WIDTH=13;
extern "C" void timeline_counts(int nu,int nd,const int* off,const int16_t* days,int* sizes){
    std::vector<int> mark(nd,-1);
    for(int u=0;u<nu;++u)for(int j=off[u];j<off[u+1];++j)
        if(mark[days[j]]!=u){mark[days[j]]=u;++sizes[u];}
}
static void add(float* x,int m,int r,float sign,const float* geo,const float* movie_mean){
    x[r-1]+=sign;x[5]+=sign*movie_mean[m];
    for(int j=0;j<G;++j){float a=geo[m*G+j]*sign;x[6+j]+=a;x[6+G+j]+=a*(r-movie_mean[m]);}
}
extern "C" void timeline_build(int nu,int nd,const int* off,const int16_t* item,const int16_t* dates,
    const int8_t* star,const float* geo,const float* movie_mean,const int* soff,int16_t* day,float* sums){
    std::vector<int> map(nd,-1),used;
    for(int u=0;u<nu;++u){
        used.clear();
        for(int j=off[u];j<off[u+1];++j)if(map[dates[j]]<0){map[dates[j]]=0;used.push_back(dates[j]);}
        std::sort(used.begin(),used.end());
        for(size_t j=0;j<used.size();++j){int pos=soff[u]+j;map[used[j]]=pos;day[pos]=used[j];}
        for(int j=off[u];j<off[u+1];++j)add(sums+int64_t(map[dates[j]])*F,item[j],star[j],1,geo,movie_mean);
        for(int d:used)map[d]=-1;
    }
}
extern "C" void timeline_tokens(int batch,int targets,int masks,const int* users,const int16_t* queries,
    const int* removed,const int16_t* item,const int16_t* dates,const int8_t* star,const float* geo,
    const float* movie_mean,const int* soff,const int16_t* day,const float* sums,float* output){
    for(int b=0;b<batch;++b){
        int u=users[b],lo=soff[u],hi=soff[u+1];
        for(int t=0;t<targets;++t){
            int date=queries[b*targets+t];
            int at=std::lower_bound(day+lo,day+hi,date)-day;
            bool exact=at<hi&&day[at]==date;
            for(int k=0;k<WIDTH;++k){
                int j=at+k-WIDTH/2;
                if(k==WIDTH/2&&!exact)continue;
                if(k>WIDTH/2&&!exact)--j;
                if(j<lo||j>=hi)continue;
                float local[F];std::copy_n(sums+int64_t(j)*F,F,local);
                for(int h=0;h<masks;++h){int r=removed[b*masks+h];if(r>=0&&dates[r]==day[j])add(local,item[r],star[r],-1,geo,movie_mean);}
                float n=0;for(int h=0;h<5;++h)n+=local[h];
                if(n<.5f)continue;
                float* dest=output+((int64_t(b)*targets+t)*WIDTH+k)*(F+4);
                for(int h=0;h<F;++h)dest[h]=local[h]/n;
                int dt=int(day[j])-date;
                dest[F]=std::log1p(n)/4;dest[F+1]=std::copysign(std::log1p(std::abs(dt))/5.f,float(dt));
                dest[F+2]=1.f/(1.f+std::abs(dt)/14.f);dest[F+3]=1;
            }
        }
    }
}
