#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

constexpr int D=16;
struct Filter {
    float cov[D][D]={},mean[3][D]={};
    int day=-1;
    Filter() { for(int k=0;k<D;++k)cov[k][k]=.4f; }
    void advance(int now,float tau) {
        if(day!=-1 && day!=now) {
            float a=std::exp(-std::abs(day-now)/tau);
            for(int k=0;k<D;++k) {
                for(int h=0;h<D;++h)cov[k][h]*=a*a;
                cov[k][k]+=.4f*(1-a*a);
                for(int r=0;r<3;++r)mean[r][k]*=a;
            }
        }
        day=now;
    }
    float values(const float* x,float* gain,float* pred) const {
        float var=0;
        for(int k=0;k<D;++k) {
            float v=0;
            for(int h=0;h<D;++h)v+=cov[k][h]*x[h];
            gain[k]=v;var+=v*x[k];
        }
        for(int r=0;r<3;++r) {
            float m=0;for(int k=0;k<D;++k)m+=mean[r][k]*x[k];pred[r]=m;
        }
        return var;
    }
    void observe(const float* x,float a,float b,float c) {
        float g[D],pred[3],obs[3]={a,b,c};
        float scale=1.f/(.75f+values(x,g,pred));
        for(int k=0;k<D;++k) {
            float z=g[k]*scale;
            for(int h=0;h<D;++h)cov[k][h]-=z*g[h];
            for(int r=0;r<3;++r)mean[r][k]+=z*(obs[r]-pred[r]);
        }
    }
};
extern "C" void drift(int lo,int hi,const int* off,const int* qs,const int* qo,const int16_t* it,
        const int16_t* d,const int8_t* r,const float* re,const float* ce,const float* mm,
        const int16_t* qi,const int16_t* qd,const float* coords,float* out) {
    std::vector<int> train,query;
    for(int u=lo;u<hi;++u) {
        if(qs[u]==qs[u+1])continue;
        train.clear();query.clear();
        for(int j=off[u];j<off[u+1];++j)train.push_back(j);
        for(int j=qs[u];j<qs[u+1];++j)query.push_back(qo[j]);
        std::stable_sort(train.begin(),train.end(),[&](int a,int b){return d[a]<d[b];});
        std::stable_sort(query.begin(),query.end(),[&](int a,int b){return qd[a]<qd[b];});
        for(int side=0;side<2;++side) {
            if(side) {std::reverse(train.begin(),train.end());std::reverse(query.begin(),query.end());}
            for(int h=0;h<3;++h) {
                float tau=h==0?20.f:h==1?180.f:2000.f;
                Filter state;
                unsigned pos=0;
                for(int j:query) {
                    while(pos<train.size() && (side?d[train[pos]]>qd[j]:d[train[pos]]<=qd[j])) {
                        int l=train[pos++], m=it[l];
                        state.advance(d[l],tau);
                        state.observe(coords+m*D,re[l],ce[l],r[l]-mm[m]);
                    }
                    state.advance(qd[j],tau);
                    float g[D], pred[3],var=state.values(coords+qi[j]*D,g,pred);
                    float* o=out+int64_t(j)*24+(side*3+h)*4;
                    o[0]=pred[0];o[1]=pred[1];o[2]=pred[2];o[3]=var;
                }
            }
        }
    }
}
