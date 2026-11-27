#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

constexpr int D=24,H=48,T=3;
using Profile=std::array<float,H>;
using Code=std::pair<unsigned,int>;
extern "C" void profiles(int users,const int* starts,const int16_t* item,const int8_t* rating,
                         const float* movie_mean,const float* coords,float* out) {
    for(int u=0;u<users;++u) {
        float* a=out+int64_t(u)*H;
        for(int j=starts[u];j<starts[u+1];++j) {
            int m=item[j];
            const float* v=coords+m*D;
            for(int k=0;k<D;++k) {
                a[k]+=(rating[j]-movie_mean[m])*v[k];
                a[k+D]+=v[k];
            }
        }
    }
}
static Profile profile(const float* a,const float* v,float res,bool exclude) {
    Profile p;
    for(int k=0;k<D;++k) {
        p[k]=a[k]-(exclude?res*v[k]:0);
        p[k+D]=a[k+D]-(exclude?v[k]:0);
    }
    for(int j: {0,D}) {
        float norm=.00001f;
        for(int k=0;k<D;++k)norm+=p[j+k]*p[j+k];
        norm=1/std::sqrt(norm);
        for(int k=0;k<D;++k)p[j+k]*=norm;
    }
    return p;
}
static unsigned code(const Profile& p,int t,const int* perm) {
    unsigned bits=0;
    for(int k=0;k<24;++k)bits=(bits<<1)|(p[perm[t*24+k]]>0);
    return bits;
}
extern "C" void other(int from,int to,const int* s,const int* rows,const int* qs,const int* qo,
        const int* u,const int16_t* date,const int8_t* r,const float* rankerr,const float* cferr,
        const int* qu,const int16_t* qd,const float* mm,const float* coords,const float* users,
        const int* perm,float* out) {
    for(int m=from;m<to;++m) {
        int n=s[m+1]-s[m];
        if(!n||qs[m]==qs[m+1])continue;
        const float* v=coords+m*D;
        std::vector<Profile> profiles(n);
        std::array<std::vector<Code>,T> orders;
        std::vector<int> seen(n,-1),candidates;
        for(int i=0;i<n;++i) {
            int j=rows[s[m]+i];
            profiles[i]=profile(users+int64_t(u[j])*H,v,r[j]-mm[m],true);
            for(int t=0;t<T;++t)orders[t].emplace_back(code(profiles[i],t,perm),i);
        }
        for(auto& o:orders)std::sort(o.begin(),o.end());
        for(int i=qs[m];i<qs[m+1];++i) {
            int j=qo[i];
            auto p=profile(users+int64_t(qu[j])*H,v,0,false);
            candidates.clear();
            for(int t=0;t<T;++t) {
                const auto& ord=orders[t];
                int pos=std::lower_bound(ord.begin(),ord.end(),Code(code(p,t,perm),0))-ord.begin();
                for(int h=std::max(0,pos-56);h<std::min(n,pos+56);++h) {
                    int z=ord[h].second;
                    if(seen[z]!=j) {seen[z]=j;candidates.push_back(z);}
                }
            }
            double sums[6][5]={};
            for(int z:candidates) {
                int l=rows[s[m]+z];
                if(u[l]==qu[j])continue;
                float sim=0,implicit=0;
                for(int k=0;k<D;++k) {sim+=p[k]*profiles[z][k];implicit+=p[D+k]*profiles[z][D+k];}
                float near=1.f/(1.f+std::abs(qd[j]-date[l])/55.f);
                float w[6];
                for(int t=0;t<3;++t) {
                    float dot=t==0?sim:t==1?implicit:(sim+implicit)*.5f;
                    float b=std::max(0.f,.5f+.5f*dot); b*=b; b*=b; b*=b;
                    w[t*2]=b*b; w[t*2+1]=b*b*near;
                }
                for(int t=0;t<6;++t) {
                    double wt=w[t];auto a=sums[t];
                    a[0]+=wt;a[1]+=wt*rankerr[l];a[2]+=wt*cferr[l];a[3]+=wt*r[l];a[4]+=wt*wt;
                }
            }
            float* dest=out+int64_t(j)*24;
            for(int t=0;t<6;++t) {
                auto a=sums[t]; double den=a[0]+1e-9;
                dest[4*t]=a[1]/den;dest[4*t+1]=a[2]/den;dest[4*t+2]=a[3]/den;
                dest[4*t+3]=std::log1p(a[0]*a[0]/(a[4]+1e-10));
            }
        }
    }
}
