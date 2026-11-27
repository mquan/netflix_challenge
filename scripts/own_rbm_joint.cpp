#include <Accelerate/Accelerate.h>
#define main netflix_existing_cli_main
#include "../src/netflix.cpp"
#undef main

constexpr int H=100,N=9,S=64,WIDTH=7;
static uint32_t scramble(uint32_t x){x^=x>>16;x*=0x7feb352d;x^=x>>15;x*=0x846ca68b;return x^(x>>16);}
static float softplus(float x){return std::max(x,0.f)+std::log1p(std::exp(-std::abs(x)));}
static float distribution(const float* value,float* p){
    float mx=*std::max_element(value,value+5),sum=0,mu=0;
    for(int k=0;k<5;++k){p[k]=std::exp(value[k]-mx);sum+=p[k];}
    for(int k=0;k<5;++k){p[k]/=sum;mu+=(k+1)*p[k];}return mu;
}
static void infer(const Dataset& d,const JointFeatures& f,const rbm_visible_context::RbmModel& model,int u,const int* ids,int n,size_t n0,
        const float* visible,const float* quasi,const float* original,float* output,float& worst){
    float z[H];std::copy(model.hbias.begin(),model.hbias.end(),z);
    for(auto j=d.train_offsets[u];j<d.train_offsets[u+1];++j){
        auto w=model.w.data()+((int64_t(d.train_items[j]-1)*5)+d.train_ratings[j]-1)*H;
        for(int h=0;h<H;++h)z[h]+=w[h];
    }
    auto lo=d.implicit_offsets[u],hi=d.implicit_offsets[u+1];float norm=1/std::sqrt(float(std::max<uint64_t>(1,hi-lo)));
    for(auto j=lo;j<hi;++j){auto v=model.d.data()+int64_t(d.implicit_items[j]-1)*H;for(int h=0;h<H;++h)z[h]+=norm*v[h];}
    float w[N*5*H],bias[N*5],p0[N*5],native[N],hidden[H];
    for(int h=0;h<H;++h)hidden[h]=sigmoid(z[h]);
    for(int k=0;k<n;++k){
        int j=ids[k],m=d.probe[j].item-1,s=f.probe_session[j];float extra[5]{};
        for(int h=0;h<5;++h){
            if(model.visible_context&1)extra[h]+=model.user_visible[int64_t(u)*5+h];
            if(model.visible_context&2)extra[h]+=model.day_visible[int64_t(s)*5+h];
            if(model.visible_context&4)extra[h]+=model.frequency_visible[(int64_t(m)*kFrequencyBins+f.freq[s])*5+h];
        }
        float value[5];for(int t=0;t<5;++t){
            auto row=model.w.data()+(int64_t(m)*5+t)*H;auto dest=w+(k*5+t)*H;
            std::copy(row,row+H,dest);bias[k*5+t]=model.vbias[m*5+t]+extra[t];
            float v=bias[k*5+t];for(int h=0;h<H;++h)v+=hidden[h]*row[h];value[t]=v;
        }
        native[k]=distribution(value,p0+k*5);
        if(size_t(j)<n0)worst=std::max(worst,std::abs(native[k]-original[j]));
        auto out=output+int64_t(j)*WIDTH;for(int b=0;b<5;++b)out[b]=native[k];out[5]=1;out[6]=0;
    }
    float rotation[H];for(int h=0;h<H;++h)rotation[h]=float(scramble(u*3671U+h))/4294967296.f;
    for(int phase=0;phase<2;++phase){
        float observed[H];std::copy(z,z+H,observed);bool active[N]{};int available=0;bool requested=false;
        for(int k=0;k<n;++k){
            int j=ids[k];bool present=size_t(j)<n0&&(phase||visible[j]>.5f);
            if(present){int t=d.probe[j].rating-1;for(int h=0;h<H;++h)observed[h]+=w[(k*5+t)*H+h];}
            else{active[k]=true;++available;requested|=(phase?size_t(j)>=n0:size_t(j)<n0);}
        }
        if(!requested)continue;
        float estimated[N*5],proposal[H],ph[H];
        std::copy(p0,p0+n*5,estimated);
        for(int h=0;h<H;++h)ph[h]=sigmoid(observed[h]);
        for(int k=0;k<n;++k)if(active[k]){
            float exact[5],ordinary[5];
            for(int t=0;t<5;++t){
                auto v=w+(k*5+t)*H;float g=0,a=0;
                for(int h=0;h<H;++h){g+=softplus(observed[h]+v[h])-softplus(observed[h]);a+=ph[h]*v[h];}
                exact[t]=bias[k*5+t]+g;ordinary[t]=bias[k*5+t]+a;
            }
            int j=ids[k];bool keep=phase?size_t(j)>=n0:size_t(j)<n0;
            float p[5];float ordinary_mean=distribution(ordinary,p),single=distribution(exact,estimated+k*5);
            if(keep){auto out=output+int64_t(j)*WIDTH;out[1]=ordinary_mean;out[2]=single;}
        }
        for(int rep=0;rep<2;++rep){
            std::copy(observed,observed+H,proposal);
            for(int k=0;k<n;++k)if(active[k])for(int t=0;t<5;++t){
                float a=.74f*estimated[k*5+t];auto v=w+(k*5+t)*H;
                for(int h=0;h<H;++h)proposal[h]+=a*v[h];
            }
            for(int h=0;h<H;++h)ph[h]=sigmoid(proposal[h]);
            if(rep==0)for(int k=0;k<n;++k)if(active[k]){
                float logits[5];for(int t=0;t<5;++t)logits[t]=bias[k*5+t]+cblas_sdot(H,ph,1,w+(k*5+t)*H,1);
                distribution(logits,estimated+k*5);
            }
        }
        float sampled[S*H],base[S]{},logz[S]{},scores[S*5*N],means[S*N];
        for(int a=0;a<S;++a)for(int h=0;h<H;++h){
            float v=quasi[a*H+h]+rotation[h];v-=v>=1;float bit=v<ph[h];sampled[a*H+h]=bit;
            base[a]+=(bit-ph[h])*(observed[h]-proposal[h]);
        }
        cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,S,n*5,H,1,sampled,H,w,H,0,scores,n*5);
        for(int a=0;a<S;++a)for(int k=0;k<n;++k)if(active[k]){
            float* s=scores+a*n*5+k*5;for(int t=0;t<5;++t)s[t]+=bias[k*5+t];
            float mx=*std::max_element(s,s+5),sum=0,y=0;
            for(int t=0;t<5;++t){float v=std::exp(s[t]-mx);sum+=v;y+=(t+1)*v;}
            logz[a]+=mx+std::log(sum);means[a*N+k]=y/sum;
        }
        for(int mode=0;mode<2;++mode){
            float weights[S],mx=-1e30f;for(int a=0;a<S;++a){weights[a]=base[a]+(mode?.45f:1.f)*logz[a];mx=std::max(mx,weights[a]);}
            float den=0,square=0;for(auto& v:weights){v=std::exp(v-mx);den+=v;square+=v*v;}
            for(int k=0;k<n;++k)if(active[k]){
                int j=ids[k];if((phase&&size_t(j)<n0)||(!phase&&size_t(j)>=n0))continue;
                float v=0,half=0,dhalf=0;
                for(int a=0;a<S;++a){v+=weights[a]*means[a*N+k];if(a<S/2){half+=weights[a]*means[a*N+k];dhalf+=weights[a];}}
                auto out=output+int64_t(j)*WIDTH;out[3+mode]=v/den;
                if(!mode){out[5]=den*den/(S*square);out[6]=half/std::max(dhalf,1e-15f)-v/den;}
            }
        }
    }
}
int main(int argc,char** argv){
    try{
        if(argc<8||(argc-5)%3)fail("expected dataset, queries, anchor mask, Sobol points and model/reference/output triples");
        auto d=load_dataset(argv[1]);auto fp=joint_fingerprint(d);auto t=build_temporal_index(d);size_t n0=d.probe.size();
        std::ifstream input(argv[2],std::ios::binary);std::vector<ProbeRec> query;read_vector(input,query,2817131);
        for(auto& r:query)if(r.rating)fail("nonempty qualifying outcome");
        d.probe.insert(d.probe.end(),query.begin(),query.end());d.header.probe_count=d.probe.size();auto f=joint_features(d,t);
        std::vector<std::array<int,N>> grid(d.header.users);std::vector<int> count(d.header.users);
        for(size_t j=0;j<d.probe.size();++j){auto u=d.probe[j].user;if(count[u]>=N)fail("slate size");grid[u][count[u]++]=j;}
        std::ifstream vi(argv[3],std::ios::binary),si(argv[4],std::ios::binary);
        std::vector<float> visible,sobol;read_vector(vi,visible,n0);read_vector(si,sobol,H*S);
        for(int arg=5;arg<argc;arg+=3){
            auto m=rbm_visible_context::load_rbm(argv[arg],d.header.items,d.header.users,t.session_days.size(),fp);if(m.hidden!=H)fail("incompatible hidden state");
            std::vector<float> orig,out(d.probe.size()*WIDTH);std::ifstream ref(argv[arg+1],std::ios::binary);read_vector(ref,orig,n0);
            std::array<float,2> errs{};parallel_for(2,d.header.users,[&](uint32_t thread,size_t u){
                if(count[u])infer(d,f,m,u,grid[u].data(),count[u],n0,visible.data(),sobol.data(),orig.data(),out.data(),errs[thread]);
                if(u%120000==0){std::printf("%s users %zu\n",argv[arg],u);std::fflush(stdout);}
            });
            float err=std::max(errs[0],errs[1]);if(err>.00035f)fail("original RBM replay mismatch");
            std::ofstream file(argv[arg+2],std::ios::binary);write_vector(file,out);std::cout<<"maximum native error "<<err<<std::endl;
        }
    }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
}
