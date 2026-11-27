#include "own_suffix_factors.cpp"

// Share the chronological window builder and its exact target-removed CF96 scores.
struct ChoiceModel {
    std::vector<Vec> choice,taste,context;
    std::vector<float> ca,ta,ea,offset,scale,oa;
    Vec history{},recent{},style{},query{};
    std::mt19937 rng{77193};std::vector<int> seen;int stamp=0;
    std::array<std::vector<int>,90> months;
    const float* geom;
    ChoiceModel(const Data& a,const float* geo):choice(a.nm),taste(a.nm),context(a.nm),ca(a.nm,.4f),ta(a.nm,.4f),
        ea(a.nm,.4f),offset(a.nm),scale(a.nm),oa(a.nm,4),seen(a.nm),geom(geo){
        std::normal_distribution<float> init(0,.025f);
        for(int m=0;m<a.nm;++m)for(int h=0;h<R;++h){
            choice[m][h]=.08f*geo[m*R+h]+init(rng);taste[m][h]=init(rng);context[m][h]=.05f*geo[m*R+h]+init(rng);
        }
        // Empirical monthly popularity from a uniform systematic sample of training events.
        for(int j=0;j<a.off[a.nu];j+=5)months[std::clamp(int(a.day[j])/30,0,89)].push_back(a.item[j]);
    }
    static float dot(const Vec& a,const Vec& b){float s=0;for(int h=0;h<R;++h)s+=a[h]*b[h];return s;}
    void sample(const Data& a,int first,int end,int n,int visible,const int* row,const int* m,const int* day,
            const float* expected,float lr,float* correction,float* utility,float* novelty,bool fit,bool labeled){
        if(n<1)return;
        std::array<Vec,Q> grad{};history.fill(0);recent.fill(0);style.fill(0);++stamp;
        for(int k=0;k<n;++k)seen[m[k]]=stamp;
        int cut=*std::min_element(day,day+n),cnt=0;float mass=0;
        for(int j=first;j<end;++j){
            int movie=a.item[j];bool hidden=false;
            if(labeled)for(int k=visible;k<n;++k)hidden|=j==row[k];
            if(hidden)continue;
            seen[movie]=stamp;float w=1.f/(1.f+std::abs(a.day[j]-cut)/45.f);++cnt;mass+=w;
            for(int h=0;h<R;++h){
                float v=geom[movie*R+h];history[h]+=v;recent[h]+=w*v;style[h]+=(a.rating[j]-3.6f)*v;
            }
        }
        for(int h=0;h<R;++h){history[h]/=cnt+8;style[h]/=cnt+8;recent[h]/=mass+4;}
        for(int k=visible;k<n;++k){
            if(!fit&&row[k]<0)continue;
            Vec x{};float den=0;
            for(int l=0;l<n;++l)if(l!=k){
                float w=.4f+.6f/(1.f+std::abs(day[l]-day[k])/45.f);den+=w*w;
                for(int h=0;h<R;++h)x[h]+=w*context[m[l]][h];
            }
            float norm=1.f/std::sqrt(1+den);
            for(int h=0;h<R;++h)x[h]=norm*x[h]+(h<R/2?history[h]+.8f*style[h]:recent[h]+.4f*style[h]);
            int movie=m[k];float s=dot(x,choice[movie]);
            float pred=offset[movie]+dot(x,taste[movie])+scale[movie]*std::tanh(s*.65f);
            if(!fit){
                correction[row[k]]=pred;utility[row[k]]=s;
                novelty[row[k]]=dot(choice[movie],recent)-dot(choice[movie],history);continue;
            }
            Vec gx{},der{};
            if(labeled){
                float e=.8f*std::clamp(a.rating[row[k]]-expected[k]-pred,-2.8f,2.8f);
                for(int h=0;h<R;++h){gx[h]=e*taste[movie][h];der[h]=e*x[h];}
                Model::step(taste[movie],ta[movie],der,lr,.01f);
                oa[movie]+=e*e;float rate=.55f*lr/std::sqrt(oa[movie]);
                offset[movie]+=rate*(e-.07f*offset[movie]);scale[movie]+=rate*(e*std::tanh(s*.65f)-.1f*scale[movie]);
            }
            auto& negatives=months[std::clamp(day[k]/30,0,89)];
            if(negatives.empty())continue;
            float pos=0;
            for(int t=0;t<4;++t){
                int v=negatives[rng()%negatives.size()];if(seen[v]==stamp)continue;
                float diff=std::clamp(s-dot(x,choice[v]),-8.f,8.f);
                float weight=.29f/(1+std::exp(diff));pos+=weight;
                for(int h=0;h<R;++h){gx[h]+=weight*(choice[movie][h]-choice[v][h]);der[h]=-weight*x[h];}
                Model::step(choice[v],ca[v],der,lr,.005f);
            }
            for(int h=0;h<R;++h)der[h]=pos*x[h];
            Model::step(choice[movie],ca[movie],der,lr,.005f);
            for(int l=0;l<n;++l)if(l!=k){
                float w=norm*(.4f+.6f/(1.f+std::abs(day[l]-day[k])/45.f));
                for(int h=0;h<R;++h)grad[l][h]+=w*gx[h];
            }
        }
        if(fit)for(int l=0;l<n;++l)Model::step(context[m[l]],ea[m[l]],grad[l],lr,.012f);
    }
};
extern "C" void choice_factors(int nu,int nm,int nq,float sd,const int64_t* off,const int16_t* item,const int16_t* day,
    const int8_t* star,const int16_t* bound,const int* grid,const int16_t* qi,const int16_t* qd,const int8_t* anchor,
    const int* qidx,const float* enc,const float* proj,const float* hb,const float* op,const float* bias,const float* ub,
    const float* drift,const float* geometry,float* out){
    Data a{nu,nm,sd,off,item,day,star,bound,grid,qi,qd,anchor,qidx,enc,proj,hb,op,bias,ub,drift};
    std::vector<Window> windows(int64_t(nu)*CASES);std::mt19937 rng(31871);
    for(int u=0;u<nu;++u)for(int c=0;c<CASES;++c)make_window(a,u,c,rng,windows[u*CASES+c]);
    ChoiceModel model(a,geometry);std::vector<int> users(nu);std::iota(users.begin(),users.end(),0);
    auto real=[&](int u,bool fit,float lr,float* dest){
        int n=raw_count(a,u),ids[Q],m[Q],d[Q];bool needed=false;
        for(int k=0;k<n;++k){int j=grid[int64_t(u)*Q+k];ids[k]=qidx[j];m[k]=qi[j];d[k]=qd[j];needed|=ids[k]>=0;}
        if(fit||needed)model.sample(a,off[u],off[u+1],n,0,ids,m,d,nullptr,lr,dest,dest?dest+nq:nullptr,dest?dest+2*nq:nullptr,fit,false);
    };
    int snapshot=0;
    for(int ep=1;ep<=32;++ep){
        std::shuffle(users.begin(),users.end(),rng);float lr=.009f+.022f*(1+std::cos(ep*3.14159f/32))/2;
        for(int j=0;j<std::min(nu,195000);++j){
            int u=users[j];if((j%5)==0){real(u,true,lr,nullptr);continue;}
            auto& w=windows[u*CASES+rng()%CASES];if(!w.count)continue;int m[Q],d[Q];
            for(int k=0;k<w.count;++k){m[k]=item[w.rows[k]];d[k]=day[w.rows[k]];}
            model.sample(a,off[u],w.end,w.count,w.visible,w.rows.data(),m,d,w.mu.data(),lr,nullptr,nullptr,nullptr,true,true);
        }
        if(ep==8||ep==18||ep==32){float* dest=out+int64_t(snapshot++)*3*nq;for(int u=0;u<nu;++u)real(u,false,0,dest);}
        std::printf("choice epoch %d\n",ep);std::fflush(stdout);
    }
}
