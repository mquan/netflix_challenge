#include "own_chronology_windows.hpp"
constexpr int TOP=128,GROUP=128,MAXH=96,CHANNEL=6;
struct Weight{std::array<float,CHANNEL> x{};float acc=.45f;};
struct Event{Weight* weight;std::array<float,CHANNEL> x;};
struct SparseModel{
    std::vector<Weight> exact,coarse,reverse;std::vector<float> mean,intercept,bacc;
    std::vector<int> vis,position;int tick=0;const int* top;const int* group;std::mt19937 rng{28915};
    SparseModel(const Data& a,const int* nn,const int* categories):exact(int64_t(a.nm)*TOP),coarse(int64_t(a.nm)*GROUP),
        reverse(int64_t(a.nm)*GROUP),mean(a.nm),intercept(a.nm),bacc(a.nm,5),vis(a.nm),position(a.nm),top(nn),group(categories){
        std::vector<int> count(a.nm);for(int j=0;j<a.off[a.nu];++j){mean[a.item[j]]+=a.rating[j];++count[a.item[j]];}
        for(int m=0;m<a.nm;++m)mean[m]=(mean[m]+108)/(count[m]+30);
    }
    void user(const Data& a,int lo,int end,int n,int visible,const int* row,const int* movies,const int* days,const float* expected,
            float lr,float* output,bool train,bool fit_window=false){
        if(!n)return;++tick;std::vector<int> hist;hist.reserve(end-lo);float avg=0;
        for(int j=lo;j<end;++j){
            bool hidden=false;if(train)for(int k=visible;k<n;++k)hidden|=row[k]==j;if(hidden)continue;
            int m=a.item[j];vis[m]=tick;position[m]=j;hist.push_back(j);avg+=a.rating[j]-mean[m];
        }
        avg/=20+hist.size();std::array<Event,2*MAXH+TOP+2*Q+2> events;
        int available=hist.size(),take=std::min(MAXH,available);
        int offset=train&&available?int(rng()%available):available/2;
        std::array<int,MAXH> subset{};
        for(int h=0;h<take;++h)subset[h]=hist[(int(int64_t(h)*available/take)+offset)%available];
        int first=fit_window?visible:train?visible+int(rng()%(n-visible)):0;
        int last=train&&!fit_window?first+1:n;
        for(int k=first;k<last;++k){
            if(!train&&row[k]<0)continue;int m=movies[k],target_group=group[m],nt=0;float pred=intercept[m];
            auto add=[&](Weight& w,float present,float residual,float time,float second,float peer,float joint){
                Event& ev=events[nt++];ev.weight=&w;ev.x={present,residual,time,second,peer,joint};
                for(int c=0;c<CHANNEL;++c)pred+=w.x[c]*ev.x[c];
            };
            auto features=[&](Weight& param,int j,float scale){
                float v=a.rating[j]-mean[a.item[j]]-avg,t=1.f/(1.f+std::abs(int(a.day[j])-days[k])/45.f);
                add(param,scale,scale*v,scale*t*v,scale*(std::abs(v)-.8f),0,0);
            };
            float w=.65f/std::sqrt(float(8+take));for(int h=0;h<take;++h){
                int j=subset[h],other=a.item[j];
                features(coarse[m*GROUP+group[other]],j,w);features(reverse[other*GROUP+target_group],j,w);
            }
            int present=0;for(int h=0;h<TOP;++h)present+=vis[top[m*TOP+h]]==tick;
            float wn=1/std::sqrt(float(3+present));for(int h=0;h<TOP;++h){
                int other=top[m*TOP+h];if(vis[other]==tick)features(exact[m*TOP+h],position[other],wn);
            }
            for(int l=0;l<n;++l)if(l!=k){
                int other=movies[l];float w=.72f/std::sqrt(float(n)),t=1.f/(1.f+std::abs(days[l]-days[k])/35.f);
                add(coarse[m*GROUP+group[other]],0,0,0,0,w,w*t);add(reverse[other*GROUP+target_group],0,0,0,0,w,w*t);
            }
            if(fit_window){output[k]=pred;continue;}
            if(!train){output[row[k]]=pred;continue;}
            float err=std::clamp(a.rating[row[k]]-expected[k]-pred,-3.f,3.f);
            bacc[m]+=err*err;intercept[m]+=.45f*lr*(err-.08f*intercept[m])/std::sqrt(bacc[m]);
            for(int h=0;h<nt;++h){
                auto& ev=events[h];auto& p=*ev.weight;float ss=0;for(float z:ev.x)ss+=z*z;
                p.acc+=err*err*ss/CHANNEL;float rate=lr/std::sqrt(p.acc);
                for(int c=0;c<CHANNEL;++c)p.x[c]+=rate*(err*ev.x[c]-.003f*p.x[c]);
            }
        }
    }
};
extern "C" void sparse_cohort(int nu,int nm,int nq,float sd,const int64_t* off,const int16_t* item,const int16_t* day,
    const int8_t* star,const int16_t* bound,const int* grid,const int16_t* qi,const int16_t* qd,const int8_t* anchor,
    const int* qidx,const float* enc,const float* proj,const float* hb,const float* op,const float* bias,const float* ub,
    const float* drift,const int* top,const int* group,float* out){
    Data a{nu,nm,sd,off,item,day,star,bound,grid,qi,qd,anchor,qidx,enc,proj,hb,op,bias,ub,drift};
    constexpr int NW=7;std::vector<Window> windows(int64_t(nu)*NW);std::mt19937 rng(41687);
    for(int u=0;u<nu;++u)for(int c=0;c<NW;++c)make_window(a,u,c,rng,windows[u*NW+c]);
    SparseModel net(a,top,group);std::vector<int> users(nu);std::iota(users.begin(),users.end(),0);int snap=0;
    for(int ep=1;ep<=42;++ep){
        std::shuffle(users.begin(),users.end(),rng);float lr=.012f+.040f*(1+std::cos(ep*3.14159f/42))/2;
        for(int h=0;h<std::min(300000,nu);++h){
            int u=users[h];auto& w=windows[u*NW+rng()%NW];if(!w.count)continue;int m[Q],d[Q];
            for(int k=0;k<w.count;++k){m[k]=item[w.rows[k]];d[k]=day[w.rows[k]];}
            net.user(a,off[u],w.end,w.count,w.visible,w.rows.data(),m,d,w.mu.data(),lr,nullptr,true);
        }
        if(ep==8||ep==20||ep==32||ep==42){float* dest=out+int64_t(snap++)*nq;
            for(int u=0;u<nu;++u){int n=raw_count(a,u),rows[Q],m[Q],d[Q];bool needed=false;
                for(int k=0;k<n;++k){int j=grid[int64_t(u)*Q+k];rows[k]=qidx[j];m[k]=qi[j];d[k]=qd[j];needed|=rows[k]>=0;}
                if(needed)net.user(a,off[u],off[u+1],n,0,rows,m,d,nullptr,0,dest,false);}}
        std::printf("sparse cohort epoch %d\n",ep);std::fflush(stdout);
    }
}
