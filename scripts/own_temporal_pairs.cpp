#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>
struct Stat {float n=0, y=0, r=0, c=0, xy=0, xx=0, ord=0;};
constexpr int NB=4, NG=2, NF=9;
static inline void add(Stat& a,float w,float y,float r,float c,float other,float stars,bool rated){
    a.n+=w;a.y+=w*y;a.r+=w*r;a.c+=w*c;
    if(rated){a.xy+=w*c*other;a.xx+=w*other*other;a.ord+=w*c*(stars-3.6f);}
}
extern "C" void temporal_pairs(int first,int last,int ni,int nc,const int32_t* starts,
    const int32_t* rows,const int32_t* event,const int32_t* qs,const int32_t* qo,
    const int32_t* qsession,const int32_t* es,const int32_t* off,const int64_t* keys,
    const int32_t* lower,const int32_t* upper,const int16_t* items,const int8_t* rating,
    const float* er,const float* ec,const int32_t* cluster,float* output){
    std::vector<Stat> table(NB*(ni+nc));
    constexpr int stride=NB*NG*NF;
    for(int m=first;m<last;++m){
        if(qs[m]==qs[m+1])continue;
        std::fill(table.begin(),table.end(),Stat{});
        double my=0,mr=0,mc=0;
        for(int j=starts[m];j<starts[m+1];++j){int a=event[rows[j]];my+=rating[a];mr+=er[a];mc+=ec[a];}
        int n=std::max(1,starts[m+1]-starts[m]);my/=n;mr/=n;mc/=n;
        for(int j=starts[m];j<starts[m+1];++j){
            int a=event[rows[j]],sid=es[a];
            float y=rating[a]-my,r=er[a]-mr,c=ec[a]-mc;
            for(int s=lower[sid];s<upper[sid];++s){
                int dt=static_cast<int>(keys[s]-keys[sid]);
                int b=dt==0?0:(std::abs(dt)<=7?1:(std::abs(dt)<=45?2:3));
                auto base=table.data()+b*(ni+nc);
                int count=off[s+1]-off[s];
                float w=1.f/std::sqrt(1.f+count/16.f);
                for(int h=off[s];h<off[s+1];++h){
                    int movie=items[h];if(movie==m)continue;
                    add(base[movie],w,y,r,c,ec[h],rating[h],rating[h]!=0);
                    add(base[ni+cluster[movie]],w,y,r,c,ec[h],rating[h],rating[h]!=0);
                }
            }
        }
        for(int j=qs[m];j<qs[m+1];++j){
            int row=qo[j],sid=qsession[row];
            float acc[NB][NG][NF]{};
            for(int s=lower[sid];s<upper[sid];++s){
                int dt=static_cast<int>(keys[s]-keys[sid]);
                int b=dt==0?0:(std::abs(dt)<=7?1:(std::abs(dt)<=45?2:3));
                auto base=table.data()+b*(ni+nc);
                for(int h=off[s];h<off[s+1];++h){
                    int movie=items[h];if(movie==m)continue;
                    int group=rating[h]?0:1; auto& v=acc[b][group];
                    auto& a=base[movie];auto& c=base[ni+cluster[movie]];
                    float w=a.n/(a.n+70.f);
                    v[0]+=1;v[1]+=w;v[2]+=a.y/(a.n+70.f);
                    v[3]+=a.r/(a.n+70.f);v[4]+=a.c/(a.n+70.f);
                    v[5]+=a.xy/(a.xx+45.f)*ec[h];
                    v[6]+=c.c/(c.n+160.f);
                    v[7]+=c.xy/(c.xx+100.f)*ec[h];
                    v[8]+=a.ord/(a.n+120.f)*(rating[h]?rating[h]-3.6f:0.f);
                }
            }
            auto dest=output+int64_t(row)*stride;
            for(int b=0;b<NB;++b)for(int g=0;g<NG;++g){
                auto& v=acc[b][g];int col=(b*NG+g)*NF;
                dest[col]=std::log1p(v[0]);dest[col+1]=std::log1p(v[1]);
                for(int k=2;k<NF;++k)dest[col+k]=v[k]/std::max(1.f,k==6||k==7?v[0]:v[1]);
            }
        }
    }
}
