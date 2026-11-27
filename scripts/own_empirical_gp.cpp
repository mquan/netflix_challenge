#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

struct Moment {
    float n=0,xy=0,x=0,y=0,xx=0,yy=0;
    void add(float a,float b,float w){n+=w;xy+=w*a*b;x+=w*a;y+=w*b;xx+=w*a*a;yy+=w*b*b;}
    float get(float shrink) const {
        float den=std::max(n,1.f);
        return std::clamp((xy-x*y/den)/std::sqrt(std::max(0.f,xx-x*x/den)*std::max(0.f,yy-y*y/den)+.001f)
                          *(n/(n+shrink)),-.98f,.98f);
    }
};
extern "C" void learn_pairs(int lo,int hi,int ni,const int* items_off,const int* rows,const int* user,
                    const int* user_off,const int16_t* item,const int16_t* day,const float* error,
                    const float* scale_error,int16_t* global,int16_t* near,int16_t* scale){
    std::vector<std::array<Moment,3>> buf(ni);
    for(int m=lo;m<hi;++m) {
        std::fill(buf.begin(),buf.end(),std::array<Moment,3>{});
        for(int a=items_off[m];a<items_off[m+1];++a){
            int row=rows[a],u=user[row],start=user_off[u],end=user_off[u+1];
            float w=1.f/std::sqrt(1.f+(end-start)/200.f);
            for(int b=start;b<end;++b){
                auto &stat=buf[item[b]];
                stat[0].add(error[row],error[b],w);
                stat[2].add(scale_error[row],scale_error[b],w);
                int gap=std::abs(day[row]-day[b]);
                if(gap<=21)stat[1].add(error[row],error[b],gap==0?w:.4f*w);
            }
        }
        for(int k=0;k<ni;++k){
            int64_t pos=int64_t(m)*ni+k;
            global[pos]=std::lround(20000*buf[k][0].get(45));
            near[pos]=std::lround(20000*buf[k][1].get(32));
            scale[pos]=std::lround(20000*buf[k][2].get(55));
        }
    }
}
constexpr int TOP=48;
extern "C" void infer_pairs(int lo,int hi,int ni,const int* qu,const int16_t* qi,const int16_t* qd,
        const int* off,const int16_t* item,const int16_t* day,const float* re,const float* ce,
        const float* raw,const int16_t* global,const int16_t* near,const int16_t* scale,float* out){
    std::vector<std::pair<float,int>> cand;
    auto cov=[&](int i,int j,int dt,int mode){
        if(i==j)return 1.f;
        int64_t pos=int64_t(i)*ni+j;
        float g=global[pos]/20000.f,t=1.f/(1.f+dt/12.f);
        if(mode==0)return g;
        if(mode==1)return (1-.65f*t)*g+t*near[pos]/20000.f;
        return scale[pos]/20000.f;
    };
    for(int j=lo;j<hi;++j){
        int u=qu[j],m=qi[j],d=qd[j],a=off[u],b=off[u+1],n=std::min(TOP,b-a);
        if(!n)continue;
        cand.clear();
        for(int h=a;h<b;++h){
            float score=0;
            for(int mode=0;mode<3;++mode)score=std::max(score,std::abs(cov(m,item[h],std::abs(d-day[h]),mode)));
            cand.emplace_back(score,h);
        }
        if(n<int(cand.size()))std::nth_element(cand.begin(),cand.begin()+n,cand.end(),std::greater<>());
        for(int mode=0;mode<3;++mode){
            std::array<float,TOP*TOP> mat,tmp;
            std::array<float,TOP> target,w;
            for(int r=0;r<n;++r){
                int x=cand[r].second;
                target[r]=cov(m,item[x],std::abs(d-day[x]),mode);
                for(int c=0;c<=r;++c){
                    int y=cand[c].second;
                    mat[r*n+c]=cov(item[x],item[y],std::abs(day[x]-day[y]),mode);
                }
            }
            for(int pos=0;pos<2;++pos){
                int info=0,one=1;char part='U';
                float nugget=pos?.75f:.08f;
                do {
                    for(int r=0;r<n;++r){
                        std::copy_n(mat.data()+r*n,r+1,tmp.data()+r*n);
                        tmp[r*n+r]+=nugget;
                    }
                    spotrf_(&part,&n,tmp.data(),&n,&info);
                    if(info)nugget=2*nugget+.25f;
                } while(info && nugget<100.f);
                std::copy_n(target.data(),n,w.data());
                if(!info)spotrs_(&part,&n,&one,tmp.data(),&n,w.data(),&n,&info);
                else std::fill_n(w.data(),n,0.f);
                float *v=out+int64_t(j)*42+(2*mode+pos)*7;
                for(int k=0;k<n;++k){
                    int h=cand[k].second;float z=w[k];
                    v[0]+=z*re[h];v[1]+=z*ce[h];v[2]+=z*raw[h];v[3]+=z*std::clamp((re[h]+ce[h])*.5f,-.8f,.8f);
                    v[4]+=z*target[k];v[5]+=std::abs(z);
                }
                v[6]=nugget;
            }
        }
    }
}
