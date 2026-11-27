#include "own_sparse_cohort.cpp"
#include "own_suffix_factors.cpp"

// First fit the sparse movie pairs, then learn low-rank structure in their remaining errors.
extern "C" void joint_cohort(int nu,int nm,int nq,float sd,const int64_t* off,const int16_t* item,const int16_t* day,
    const int8_t* star,const int16_t* bound,const int* grid,const int16_t* qi,const int16_t* qd,const int8_t* anchor,
    const int* qidx,const float* enc,const float* proj,const float* hb,const float* op,const float* bias,const float* ub,
    const float* drift,const int* top,const int* group,const float* geometry,float* out){
    Data a{nu,nm,sd,off,item,day,star,bound,grid,qi,qd,anchor,qidx,enc,proj,hb,op,bias,ub,drift};
    constexpr int NW=5;std::vector<Window> windows(int64_t(nu)*NW);std::mt19937 rng(27181);
    std::vector<int> users(nu);std::iota(users.begin(),users.end(),0);
    SparseModel sparse(a,top,group);Model factor(nm,nu,geometry);
    int m[Q],d[Q];auto unpack=[&](const Window& w){
        for(int k=0;k<w.count;++k){m[k]=item[w.rows[k]];d[k]=day[w.rows[k]];}
    };
    for(int u=0;u<nu;++u)for(int c=0;c<NW;++c)make_window(a,u,c,rng,windows[u*NW+c]);
    for(int ep=1;ep<=28;++ep){
        std::shuffle(users.begin(),users.end(),rng);
        float lr=.014f+.038f*(1+std::cos(ep*3.14159f/28))/2;
        for(int h=0;h<std::min(nu,275000);++h){
            int u=users[h];auto& w=windows[u*NW+rng()%NW];if(!w.count)continue;unpack(w);
            sparse.user(a,off[u],w.end,w.count,w.visible,w.rows.data(),m,d,w.mu.data(),lr,nullptr,true);
        }
    }
    // Resample the suffix windows for the second learner.
    for(int u=0;u<nu;++u)for(int c=0;c<NW;++c){
        auto& w=windows[u*NW+c];w=Window{};make_window(a,u,c,rng,w);if(!w.count)continue;
        float delta[Q];unpack(w);
        sparse.user(a,off[u],w.end,w.count,w.visible,w.rows.data(),m,d,w.mu.data(),0,delta,true,true);
        for(int k=w.visible;k<w.count;++k)w.mu[k]+=delta[k];
    }
    std::vector<float> first(nq);int snap=0;
    for(int ep=1;ep<=32;++ep){
        std::shuffle(users.begin(),users.end(),rng);float lr=.007f+.02f*(1+std::cos(ep*3.14159f/32))/2;
        for(int h=0;h<std::min(nu,225000);++h){
            int u=users[h];auto& w=windows[u*NW+rng()%NW];if(!w.count)continue;unpack(w);
            factor.sample(a,u,off[u],w.end,w.count,w.visible,w.rows.data(),m,d,w.mu.data(),*std::min_element(d,d+w.count),lr,nullptr,true);
        }
        if(ep==6||ep==14||ep==23||ep==32){
            float* dest=out+int64_t(snap++)*nq;
            for(int u=0;u<nu;++u){int n=raw_count(a,u),ids[Q];bool needed=false;
                for(int k=0;k<n;++k){int j=grid[int64_t(u)*Q+k];ids[k]=qidx[j];m[k]=qi[j];d[k]=qd[j];needed|=ids[k]>=0;}
                if(!needed)continue;
                if(snap==1)sparse.user(a,off[u],off[u+1],n,0,ids,m,d,nullptr,0,first.data(),false);
                factor.sample(a,u,off[u],off[u+1],n,0,ids,m,d,nullptr,bound[u],0,dest,false);
            }
            for(int j=0;j<nq;++j)dest[j]+=first[j];
        }
        std::printf("residual factor epoch %d\n",ep);std::fflush(stdout);
    }
}
