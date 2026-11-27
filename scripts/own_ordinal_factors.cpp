#include "own_chronology_windows.hpp"

constexpr int WIDTH=48,NW=7,CAP=84;
using Lat=std::array<float,WIDTH>;
struct OrdinalWindow{Window w;std::array<float,5*Q> law;};
struct Ordinal{
    int nm;std::vector<Lat> rated,exposed,cohort,item,user;
    std::vector<std::array<float,4>> mbi,ubi;
    std::vector<float> rr,ir,cr,orate,ur,ma,ua;
    std::mt19937 rng{15371};
    Ordinal(int n,int nu,const float* init):nm(n),rated(5*n),exposed(n),cohort(n),item(4*n),user(nu),
        mbi(n),ubi(nu),rr(5*n,.12f),ir(n,.12f),cr(n,.16f),orate(4*n,.24f),ur(nu,.65f),ma(n,2),ua(nu,4){
        std::normal_distribution<float> normal(0,.028f);
        for(int m=0;m<n;++m)for(int k=0;k<4;++k)for(int h=0;h<WIDTH;++h)
            item[m*4+k][h]=.045f*init[m*R+h%R]+normal(rng);
    }
    static void update(Lat& v,float& acc,const Lat& g,float lr,float reg){
        float ss=0;for(float x:g)ss+=x*x;acc+=ss/WIDTH;float rate=lr/std::sqrt(acc);
        for(int h=0;h<WIDTH;++h)v[h]+=rate*(g[h]-reg*v[h]);
    }
    void sample(const Data& a,int u,int lo,int end,int n,int vis,const int* row,const int* m,const int* day,
            const float* law,int cut,float lr,float* delta,float* spread,bool train){
        if(!n)return;std::array<int,CAP> observed{};std::array<float,CAP> near{};int available=end-lo-(train?n-vis:0);
        int cap=train?std::min(available,CAP):available;float count=0,wt=0;Lat hstate{};
        std::vector<int> all;std::vector<float> times;
        if(!train){all.reserve(available);times.reserve(available);}
        // Stratified history sampling gives long histories the same pooling scale at prediction.
        int offset=train&&cap?int(rng()%std::max(1,end-lo)):0,taken=0;
        for(int index=0;index<(train?cap:end-lo);++index){
            int j=train?lo+(int(int64_t(index)*(end-lo)/std::max(1,cap))+offset)%(end-lo):lo+index;
            if(train){
                bool hidden=false;for(int k=vis;k<n;++k)hidden|=row[k]==j;
                if(hidden)continue;
            }
            int film=a.item[j];float t=1.f/(1.f+std::abs(int(a.day[j])-cut)/62.f);
            auto& w=rated[int(a.rating[j]-1)*nm+film];auto& impl=exposed[film];
            for(int h=0;h<WIDTH;++h)hstate[h]+=(w[h]+.45f*impl[h])*(h<WIDTH/2?1.f:t);
            if(train){observed[taken]=j;near[taken]=t;}else{all.push_back(j);times.push_back(t);}
            ++taken;count+=1;wt+=t*t;
        }
        float global=std::sqrt(float(std::min(CAP,taken)))/std::max(1.f,count);
        float recent=std::sqrt(std::min(float(CAP),wt))/std::max(1.f,wt);
        for(int h=0;h<WIDTH;++h)hstate[h]=hstate[h]*(h<WIDTH/2?global:recent)+user[u][h];
        float peer=1.f/std::sqrt(float(std::max(1,n-1)));Lat grad{};std::array<Lat,Q> peergrad{};
        std::array<float,4> ug{};
        for(int k=vis;k<n;++k){
            if(!train&&row[k]<0)continue;int movie=m[k];Lat x=hstate;
            for(int l=0;l<n;++l)if(k!=l){float t=.3f+.7f/(1.f+std::abs(day[l]-day[k])/32.f);
                for(int h=0;h<WIDTH;++h)x[h]+=peer*t*cohort[m[l]][h];}
            float original[4],adjusted[4],adjust=0,tail=1;
            for(int t=0;t<4;++t){
                tail-=law[5*k+t];original[t]=std::clamp(tail,.0002f,.9998f);
                auto& q=item[4*movie+t];float slope=mbi[movie][t]+ubi[u][t];
                for(int h=0;h<WIDTH;++h)slope+=x[h]*q[h];
                float odds=original[t]/(1-original[t])*std::exp(std::clamp(slope,-6.f,6.f));
                adjusted[t]=odds/(1+odds);adjust+=adjusted[t]-original[t];
            }
            if(!train){delta[row[k]]=adjust;spread[row[k]]=(adjusted[0]-original[0])-(adjusted[3]-original[3]);continue;}
            float pred=1+std::accumulate(adjusted,adjusted+4,0.f),err=a.rating[row[k]]-pred;
            float normal=1/std::sqrt(float(n-vis));Lat gx{};
            for(int t=0;t<4;++t){
                float v=adjusted[t],y=a.rating[row[k]]>t+1?1.f:0.f;
                float e=normal*(.55f*(y-v)+1.15f*err*v*(1-v));Lat gq{};
                auto& q=item[movie*4+t];for(int h=0;h<WIDTH;++h){gx[h]+=e*q[h];gq[h]=e*x[h];}
                update(q,orate[movie*4+t],gq,lr,.004f);
                ma[movie]+=.25f*e*e;mbi[movie][t]+=.7f*lr*(e-.02f*mbi[movie][t])/std::sqrt(ma[movie]);ug[t]+=e;
            }
            for(int h=0;h<WIDTH;++h)grad[h]+=gx[h];
            for(int l=0;l<n;++l)if(l!=k){float t=peer*(.3f+.7f/(1.f+std::abs(day[l]-day[k])/32.f));
                for(int h=0;h<WIDTH;++h)peergrad[l][h]+=t*gx[h];}
        }
        if(!train)return;
        update(user[u],ur[u],grad,.55f*lr,.16f);
        for(int t=0;t<4;++t){ua[u]+=.25f*ug[t]*ug[t];ubi[u][t]+=.6f*lr*(ug[t]-.12f*ubi[u][t])/std::sqrt(ua[u]);}
        for(int l=0;l<n;++l)update(cohort[m[l]],cr[m[l]],peergrad[l],lr,.006f);
        for(int z=0;z<taken;++z){
            int j=observed[z],m0=a.item[j],token=(int(a.rating[j])-1)*nm+m0;Lat dw{},di{};
            for(int h=0;h<WIDTH;++h){dw[h]=grad[h]*(h<WIDTH/2?global:recent*near[z]);di[h]=.45f*dw[h];}
            update(rated[token],rr[token],dw,lr,.00085f);update(exposed[m0],ir[m0],di,lr,.0013f);
        }
    }
};

extern "C" void train_ordinal_factors(int nu,int nm,int nq,float sd,const int64_t* off,const int16_t* item,const int16_t* day,
    const int8_t* star,const int16_t* bound,const int* grid,const int16_t* qi,const int16_t* qd,const int8_t* anchor,
    const int* qidx,const float* enc,const float* proj,const float* hb,const float* op,const float* bias,const float* ub,
    const float* drift,const float* geometry,const float* prob,float* out){
    Data a{nu,nm,sd,off,item,day,star,bound,grid,qi,qd,anchor,qidx,enc,proj,hb,op,bias,ub,drift};
    std::vector<OrdinalWindow> windows(int64_t(nu)*NW);std::mt19937 rng(112847);
    for(int u=0;u<nu;++u)for(int c=0;c<NW;++c){auto& w=windows[u*NW+c];make_window(a,u,c,rng,w.w,w.law.data());}
    Ordinal net(nm,nu,geometry);std::vector<int> users(nu);std::iota(users.begin(),users.end(),0);int snap=0;
    for(int ep=1;ep<=64;++ep){
        std::shuffle(users.begin(),users.end(),rng);float lr=.008f+.028f*(1.f+std::cos(ep*3.14159f/64))/2;
        for(int h=0;h<std::min(200000,nu);++h){
            int u=users[h];auto& rec=windows[u*NW+rng()%NW];auto& w=rec.w;if(!w.count)continue;
            int m[Q],d[Q],cut=30000;
            for(int k=0;k<w.count;++k){m[k]=item[w.rows[k]];d[k]=day[w.rows[k]];cut=std::min(cut,d[k]);}
            net.sample(a,u,off[u],w.end,w.count,w.visible,w.rows.data(),m,d,rec.law.data(),cut,lr,nullptr,nullptr,true);
        }
        if(ep==12||ep==26||ep==44||ep==64){
            float* dest=out+int64_t(snap++)*2*nq;
            for(int u=0;u<nu;++u){
                int n=raw_count(a,u),rows[Q],m[Q],d[Q];float laws[5*Q]{};bool needed=false;
                for(int k=0;k<n;++k){int j=grid[int64_t(u)*Q+k];rows[k]=qidx[j];m[k]=qi[j];d[k]=qd[j];
                    if(rows[k]>=0){needed=true;std::copy(prob+int64_t(rows[k])*5,prob+int64_t(rows[k])*5+5,laws+5*k);}}
                if(needed)net.sample(a,u,off[u],off[u+1],n,0,rows,m,d,laws,bound[u],0,dest,dest+nq,false);
            }
        }
        std::printf("ordinal factor epoch %d\n",ep);std::fflush(stdout);
    }
}
