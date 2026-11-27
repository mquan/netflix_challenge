#include "own_chronology_windows.hpp"
namespace graph {
constexpr int L=16;
using Axis=std::array<float,L>;
using Law=std::array<float,5>;
struct Slate {int n=0;std::array<int,Q> item{},day{},star{};std::array<Law,Q> p{};};
static Law probabilities(Law z){
    float mx=*std::max_element(z.begin(),z.end()),sum=0;
    for(float& v:z)sum+=(v=std::exp(v-mx));
    for(float& v:z)v/=sum;
    return z;
}
static void prior(const Data& a,int u,int end,const int* omit,int n_omit,Slate& s){
    double total[D]{},ds=0;int cnt=0;
    for(int j=a.off[u];j<end;++j){
        bool drop=false;for(int h=0;h<n_omit;++h)drop|=a.item[j]==omit[h];if(drop)continue;
        const float* v=a.enc+(int64_t(a.rating[j]-1)*a.nm+a.item[j])*D;
        for(int h=0;h<D;++h)total[h]+=v[h];ds+=a.day[j];++cnt;
    }
    float x[D],h[H],norm=1/std::sqrt(float(std::max(1,cnt)));
    for(int k=0;k<D;++k)x[k]=total[k]*norm;
    cblas_sgemv(CblasRowMajor,CblasNoTrans,H,D,1.f,a.proj,D,x,1,0.f,h,1);
    for(int k=0;k<H;++k)h[k]+=a.hb[k];int width=H;vvtanhf(h,h,&width);
    for(int k=0;k<s.n;++k){
        int m=s.item[k];float dif=(s.day[k]-(cnt?ds/cnt:s.day[k]))/a.sd,dev=std::copysign(std::pow(std::abs(dif),.4f),dif);Law z;
        for(int r=0;r<5;++r)z[r]=cblas_sdot(H,h,1,a.op+(int64_t(m)*5+r)*H,1)+a.bias[m*5+r]+a.ub[u*5+r]+dev*a.drift[u*5+r];
        s.p[k]=probabilities(z);
    }
}
struct Net{
    std::vector<Axis> rating,choice;std::vector<Law> bias;std::vector<float> ra,ca,ba;
    Net(int nm,const float* geom):rating(nm*5),choice(nm),bias(nm),ra(nm*5,.08f),ca(nm,.08f),ba(nm,1){
        std::mt19937 g(78913);std::normal_distribution<float> noise(0,.065f);
        for(int m=0;m<nm;++m)for(int h=0;h<L;++h){
            choice[m][h]=noise(g);
            for(int r=0;r<5;++r)rating[5*m+r][h]=noise(g)+(r-2)*(.045f*geom[m*32+h]+(h==0?.19f:0.f));
        }
    }
    static void update(Axis& v,float& acc,const Axis& g,float lr,float reg){
        float sq=0;for(float h:g)sq+=h*h;acc+=sq/L;
        float rate=lr/std::sqrt(acc);
        for(int h=0;h<L;++h)v[h]+=rate*(g[h]-reg*v[h]);
    }
    Law predict(const Slate& s,int k,const Law* state,float lr=0){
        int m=s.item[k];Axis ctx{},dg{};float w[Q];Law z;
        for(int l=0;l<s.n;++l)if(l!=k){
            int other=s.item[l];w[l]=(.28f+.72f/(1+std::abs(s.day[l]-s.day[k])/40.f))/std::sqrt(float(s.n));
            for(int h=0;h<L;++h){
                float e=choice[other][h];for(int r=0;r<5;++r)e+=(state[l][r]-s.p[l][r])*rating[5*other+r][h];
                ctx[h]+=w[l]*e;
            }
        }
        for(int r=0;r<5;++r){float v=0;for(int h=0;h<L;++h)v+=rating[5*m+r][h]*ctx[h];
            z[r]=std::log(std::max(1e-9f,s.p[k][r]))+bias[m][r]+v;}
        Law p=probabilities(z);if(!lr)return p;
        float mu=0;for(int r=0;r<5;++r)mu+=p[r]*(r+1);
        Law grad;float sq=0;
        for(int r=0;r<5;++r){grad[r]=.55f*((s.star[k]==r+1)-p[r])+.75f*(s.star[k]-mu)*p[r]*(r+1-mu);sq+=grad[r]*grad[r];
            for(int h=0;h<L;++h)dg[h]+=grad[r]*rating[5*m+r][h];}
        ba[m]+=sq/5;float rate=.55f*lr/std::sqrt(ba[m]);
        for(int r=0;r<5;++r){Axis g;for(int h=0;h<L;++h)g[h]=grad[r]*ctx[h];
            update(rating[5*m+r],ra[5*m+r],g,lr,.003f);bias[m][r]+=rate*(grad[r]-.02f*bias[m][r]);}
        for(int l=0;l<s.n;++l)if(k!=l){
            int other=s.item[l];Axis g;for(int h=0;h<L;++h)g[h]=w[l]*dg[h];
            update(choice[other],ca[other],g,lr,.006f);
            for(int r=0;r<5;++r){float e=(state[l][r]-s.p[l][r]);if(e==0)continue;
                Axis v;for(int h=0;h<L;++h)v[h]=e*g[h];update(rating[5*other+r],ra[5*other+r],v,lr,.003f);}
        }
        return p;
    }
};
}
extern "C" void ordinal_graph(int nu,int nm,int nq,float sd,const int64_t* off,const int16_t* item,const int16_t* day,
    const int8_t* star,const int16_t* bound,const int* grid,const int16_t* qi,const int16_t* qd,const int8_t* anchor,
    const int* qidx,const float* enc,const float* proj,const float* hb,const float* op,const float* bias,const float* ub,
    const float* drift,const float* geometry,float* out){
    using namespace graph;
    Data a{nu,nm,sd,off,item,day,star,bound,grid,qi,qd,anchor,qidx,enc,proj,hb,op,bias,ub,drift};
    constexpr int NW=5;std::mt19937 rng(66237);std::vector<Slate> synthetic(int64_t(nu)*NW),real(nu);
    std::vector<int> users(nu);std::iota(users.begin(),users.end(),0);
    for(int u=0;u<nu;++u){
        for(int c=0;c<NW;++c){
            Window w;make_window(a,u,c,rng,w);auto& s=synthetic[u*NW+c];s.n=w.count;
            for(int k=0;k<s.n;++k){int j=w.rows[k];s.item[k]=item[j];s.day[k]=day[j];s.star[k]=star[j];}
            if(s.n)prior(a,u,w.end,s.item.data(),s.n,s);
        }
        auto& s=real[u];s.n=raw_count(a,u);
        for(int k=0;k<s.n;++k){int j=grid[u*Q+k];s.item[k]=qi[j];s.day[k]=qd[j];s.star[k]=anchor[j];}
        if(s.n)prior(a,u,off[u+1],s.item.data(),s.n,s);
    }
    Net net(nm,geometry);int snap=0;
    for(int ep=1;ep<=44;++ep){
        std::shuffle(users.begin(),users.end(),rng);float lr=.012f+.035f*(1+std::cos(ep*3.14159f/44))/2;
        for(int h=0;h<std::min(350000,nu);++h){
            auto& s=synthetic[users[h]*NW+rng()%NW];if(s.n==0)continue;int k=rng()%s.n;
            auto state=s.p;int chance=(rng()%4)*24+12;
            for(int l=0;l<s.n;++l)if(l!=k&&int(rng()%100)<chance){state[l].fill(0);state[l][s.star[l]-1]=1;}
            net.predict(s,k,state.data(),lr);
        }
        if(ep==8||ep==23||ep==44){
            float* dest=out+int64_t(snap++)*2*nq;
            for(int u=0;u<nu;++u){auto& s=real[u];auto state=s.p;
                for(int k=0;k<s.n;++k)if(s.star[k]){state[k].fill(0);state[k][s.star[k]-1]=1;}
                for(int sweep=0;sweep<3;++sweep){
                    auto next=state;
                    for(int k=0;k<s.n;++k)if(!s.star[k]){
                        auto p=net.predict(s,k,state.data());int j=qidx[grid[u*Q+k]];
                        if(j>=0&&(sweep==0||sweep==2)){
                            float mu=0;for(int r=0;r<5;++r)mu+=(r+1)*p[r];
                            dest[int64_t(sweep/2)*nq+j]=mu;
                        }
                        for(int r=0;r<5;++r)next[k][r]=.2f*state[k][r]+.8f*p[r];
                    }
                    state=next;
                }
            }
        }
        if(ep%4==0){std::printf("ordinal graph epoch %d\n",ep);std::fflush(stdout);}
    }
}
