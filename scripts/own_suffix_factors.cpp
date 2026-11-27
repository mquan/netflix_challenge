#include "own_chronology_windows.hpp"
struct Model{
    std::vector<Vec> q,w,g;std::vector<float> aq,aw,ag,mu,bu,am,au;
    Model(int nm,int nu,const float* initial):q(nm),w(nm*5),g(nm),aq(nm,.25f),aw(nm*5,.03f),ag(nm,.1f),
        mu(nm),bu(nu),am(nm,1),au(nu,1){
        std::mt19937 rng(2861);std::normal_distribution<float> noise(0,.035f);
        for(int i=0;i<nm;++i)for(int h=0;h<R;++h)q[i][h]=.08f*initial[i*R+h]+noise(rng);
    }
    static void step(Vec& v,float& acc,const Vec& grad,float lr,float reg){
        float sq=0;for(float x:grad)sq+=x*x;acc+=sq/R;
        float rate=lr/std::sqrt(acc);
        for(int h=0;h<R;++h)v[h]+=rate*(grad[h]-reg*v[h]);
    }
    void sample(const Data& a,int u,int begin,int end,int nco,int vis,const int* ids,const int* movies,const int* days,
            const float* expected,int cut,float lr,float* output,bool train){
        if(nco<=0)return;
        std::vector<int8_t> mask(end-begin);std::vector<float> recent(end-begin);
        if(train)for(int k=vis;k<nco;++k)mask[ids[k]-begin]=1;
        int n=0;float sr=0;Vec state{},grad{};
        for(int j=begin;j<end;++j)if(!mask[j-begin]){
            float s=1.f/(1.f+std::abs(int(a.day[j])-cut)/50.f);recent[j-begin]=s;sr+=s*s;++n;
            const auto& row=w[int(a.rating[j]-1)*a.nm+a.item[j]];
            for(int h=0;h<R/2;++h)state[h]+=row[h];
            for(int h=R/2;h<R;++h)state[h]+=s*row[h];
        }
        const float norm=1/std::sqrt(float(std::max(1,n))),rn=1/std::sqrt(std::max(1.f,sr)),gn=1/std::sqrt(float(std::max(1,nco-1)));
        for(int h=0;h<R;++h)state[h]*=h<R/2?norm:rn;
        std::array<Vec,Q> ggrad{};float ug=0,wt=train?1/std::sqrt(float(nco-vis)):1.f;
        for(int k=vis;k<nco;++k){
            int m=movies[k];Vec user=state;
            for(int l=0;l<nco;++l)if(l!=k){
                float near=1.f/(1.f+std::abs(days[l]-days[k])/30.f);const auto& v=g[movies[l]];
                for(int h=0;h<R;++h)user[h]+=gn*(h<R/2?1.f:near)*v[h];
            }
            float delta=mu[m]+bu[u];for(int h=0;h<R;++h)delta+=user[h]*q[m][h];
            if(!train){if(ids[k]>=0)output[ids[k]]=delta;continue;}
            float error=std::clamp(float(a.rating[ids[k]])-expected[k]-delta,-3.f,3.f)*wt;
            Vec qgrad{};for(int h=0;h<R;++h){qgrad[h]=error*user[h];grad[h]+=error*q[m][h];}
            for(int l=0;l<nco;++l)if(l!=k){
                float near=1.f/(1.f+std::abs(days[l]-days[k])/30.f);
                for(int h=0;h<R;++h)ggrad[l][h]+=gn*(h<R/2?1.f:near)*error*q[m][h];
            }
            step(q[m],aq[m],qgrad,lr,.004f);
            am[m]+=error*error;mu[m]+=.35f*lr*(error-.05f*mu[m])/std::sqrt(am[m]);ug+=error;
        }
        if(!train)return;
        au[u]+=ug*ug;bu[u]+=.3f*lr*(ug-.12f*bu[u])/std::sqrt(au[u]);
        for(int k=0;k<nco;++k)step(g[movies[k]],ag[movies[k]],ggrad[k],lr,.006f);
        for(int j=begin;j<end;++j)if(!mask[j-begin]){
            int idx=int(a.rating[j]-1)*a.nm+a.item[j];Vec vgrad;
            for(int h=0;h<R;++h)vgrad[h]=grad[h]*(h<R/2?norm:rn*recent[j-begin]);
            step(w[idx],aw[idx],vgrad,lr,.00065f);
        }
    }
};
extern "C" void terminal_factors(int nu,int nm,int nq,float sd,const int64_t* off,const int16_t* item,const int16_t* day,
    const int8_t* star,const int16_t* bound,const int* grid,const int16_t* qi,const int16_t* qd,const int8_t* anchor,
    const int* qidx,const float* enc,const float* proj,const float* hb,const float* op,const float* bias,const float* ub,
    const float* drift,const float* geometry,float* out){
    Data a{nu,nm,sd,off,item,day,star,bound,grid,qi,qd,anchor,qidx,enc,proj,hb,op,bias,ub,drift};
    std::vector<Window> windows(int64_t(nu)*CASES);std::mt19937 rng(81723);
    for(int u=0;u<nu;++u)for(int c=0;c<CASES;++c)make_window(a,u,c,rng,windows[u*CASES+c]);
    std::printf("prepared rolling CF targets for %d users\n",nu);std::fflush(stdout);
    Model model(nm,nu,geometry);std::vector<int> users(nu);std::iota(users.begin(),users.end(),0);
    int snapshot=0;
    for(int epoch=1;epoch<=36;++epoch){
        std::shuffle(users.begin(),users.end(),rng);float lr=.007f+.02f*(1.f+std::cos(epoch*3.14159f/36))/2;
        for(int h=0;h<std::min(nu,225000);++h){
            int u=users[h];auto& w=windows[u*CASES+rng()%CASES];if(!w.count)continue;
            int m[Q],d[Q],cut=30000;for(int k=0;k<w.count;++k){m[k]=item[w.rows[k]];d[k]=day[w.rows[k]];cut=std::min(cut,d[k]);}
            model.sample(a,u,off[u],w.end,w.count,w.visible,w.rows.data(),m,d,w.mu.data(),cut,lr,nullptr,true);
        }
        if(epoch==8||epoch==16||epoch==26||epoch==36){
            float* dest=out+int64_t(snapshot++)*nq;
            for(int u=0;u<nu;++u){
                int n=raw_count(a,u),ids[Q],m[Q],d[Q];bool needed=false;
                for(int k=0;k<n;++k){int j=grid[int64_t(u)*Q+k];ids[k]=qidx[j];m[k]=qi[j];d[k]=qd[j];needed|=ids[k]>=0;}
                if(needed)model.sample(a,u,off[u],off[u+1],n,0,ids,m,d,nullptr,bound[u],0,dest,false);
            }
        }
        std::printf("epoch %d\n",epoch);std::fflush(stdout);
    }
}
