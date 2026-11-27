#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <Accelerate/Accelerate.h>
#define main netflix_existing_cli_main
#include "../src/netflix.cpp"
#undef main

struct MappedDayModel{
    JointModel model;void* mapping=MAP_FAILED;size_t bytes=0;const float* daily=nullptr;
    MappedDayModel(const char* path,const Dataset& d,const TemporalIndex& t,uint64_t expected){
        std::ifstream in(path,std::ios::binary);uint64_t magic=0,fp=0;uint32_t rank=0,epoch=0;
        read_pod(in,magic);read_pod(in,fp);read_pod(in,rank);read_pod(in,epoch);
        if((magic!=0x4E46504A4F494E32ULL&&magic!=0x4E46504A4F494E33ULL)||fp!=expected||rank!=128)fail("J4 fingerprint or rank");
        model=new_joint(d,t,rank,0);auto& b=model.base;auto& v=model.factors;
        for(auto* a:{&b.user_bias,&b.user_alpha,&b.user_scale,&b.item_bias,&b.item_bin_bias,&b.item_frequency_bias,
                     &b.day_bias,&b.day_scale,&v.user_factors,&v.item_factors,&v.implicit_factors,&model.drift}){
            uint64_t n=0;read_pod(in,n);if(n!=a->size())fail("J4 vector shape");read_vector(in,*a,n);
        }
        uint32_t dr=0;uint64_t n=0;read_pod(in,dr);read_pod(in,n);
        if(dr!=rank||n!=t.session_days.size()*rank)fail("J4 day shape");
        size_t offset=in.tellg();in.seekg(n*sizeof(float),std::ios::cur);
        if(magic==0x4E46504A4F494E33ULL){uint64_t fr=0;read_pod(in,fr);if(fr)fail("unexpected frequency factors");}
        int fd=open(path,O_RDONLY);struct stat st{};if(fd<0||fstat(fd,&st))fail("J4 file");
        bytes=st.st_size;if(size_t(in.tellg())!=bytes)fail("J4 length");
        mapping=mmap(nullptr,bytes,PROT_READ,MAP_PRIVATE,fd,0);close(fd);if(mapping==MAP_FAILED)fail("J4 mmap");
        daily=reinterpret_cast<const float*>(static_cast<char*>(mapping)+offset);model.day_rank=dr;
    }
    ~MappedDayModel(){if(mapping!=MAP_FAILED)munmap(mapping,bytes);}
};
constexpr int OUTFEAT=20,M=9,H=128,C=16;
int main(int argc,char** argv){
    try{
        if(argc!=7)fail("expected dataset, queries, J4, anchors, features, probe reference");
        auto d=load_dataset(argv[1]);auto fp=joint_fingerprint(d);auto time=build_temporal_index(d);
        MappedDayModel file(argv[3],d,time,fp);auto& model=file.model;
        size_t n0=d.probe.size();std::ifstream input(argv[2],std::ios::binary);std::vector<ProbeRec> extra;
        read_vector(input,extra,2817131);for(auto& r:extra)if(r.rating||r.user>=d.header.users||!r.item||r.item>d.header.items)fail("query input");
        d.probe.insert(d.probe.end(),extra.begin(),extra.end());d.header.probe_count=d.probe.size();extra.clear();
        auto f=joint_features(d,time);std::vector<float> effective;build_effective_users(d,model.factors,H,effective);
        std::vector<float> anchor,ref;std::ifstream ai(argv[4],std::ios::binary),ri(argv[6],std::ios::binary);
        read_vector(ai,anchor,2*n0);read_vector(ri,ref,n0);
        std::vector<std::array<int,M>> grid(d.header.users);std::vector<int> counts(d.header.users);
        for(size_t j=0;j<d.probe.size();++j){int u=d.probe[j].user;if(counts[u]>=M)fail("cohort width");grid[u][counts[u]++]=j;}
        std::vector<float> features(d.probe.size()*OUTFEAT);
        const float* film=model.factors.item_factors.data();float maxerr=0;
        for(int u=0;u<int(d.header.users);++u){
            int nq=counts[u];if(!nq)continue;int begin=time.session_offsets[u],ns=time.session_offsets[u+1]-begin;
            std::vector<int> hist(ns);std::vector<std::array<float,C>> center(ns);
            for(auto j=d.train_offsets[u];j<d.train_offsets[u+1];++j){
                int s=f.train_session[j]-begin,m=d.train_items[j]-1;++hist[s];
                auto v=film+int64_t(m)*H;for(int k=0;k<C;++k)center[s][k]+=v[k];
            }
            for(int s=0;s<ns;++s){float norm=.0001f;for(float x:center[s])norm+=x*x;norm=1/std::sqrt(norm);for(auto& x:center[s])x*=norm;}
            std::array<float,M*H> queries{};float film_bias[M],native_day[M],error[M]{},native[M];int day[M],sid[M];int m_id[M];
            for(int k=0;k<nq;++k){
                int j=grid[u][k];auto& r=d.probe[j];int s=f.probe_session[j],m=r.item-1;m_id[k]=m;day[k]=r.day;sid[k]=s-begin;
                auto q=film+int64_t(m)*H;std::copy(q,q+H,queries.data()+k*H);
                film_bias[k]=model.base.item_bias[m]+model.base.item_bin_bias[m*kTimeBins+time_bin(d,r.day)];
                auto a=model.drift.data()+int64_t(u)*H,p=effective.data()+int64_t(u)*H;
                double pred=joint_base(d,f,model,u,m,r.day,s);
                for(int h=0;h<H;++h)pred+=q[h]*(p[h]+a[h]*f.dev[s]);
                auto ds=file.daily+int64_t(s)*H;double effect=0;for(int h=0;h<H;++h)effect+=q[h]*ds[h];
                pred+=effect;native[k]=float(pred);native_day[k]=float(effect)+model.base.day_bias[s]+film_bias[k]*model.base.day_scale[s];
                if(size_t(j)<n0){maxerr=std::max(maxerr,std::abs(native[k]-ref[j]));error[k]=r.rating-anchor[2*j];}
            }
            std::vector<float> shifts(ns*nq),similar(ns*nq);
            cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,ns,nq,H,1.f,file.daily+int64_t(begin)*H,H,queries.data(),H,0.f,shifts.data(),nq);
            for(int k=0;k<nq;++k){
                auto q=queries.data()+k*H;float norm=.0001f;for(int h=0;h<C;++h)norm+=q[h]*q[h];norm=1/std::sqrt(norm);
                for(int s=0;s<ns;++s){
                    shifts[s*nq+k]+=model.base.day_bias[begin+s]+film_bias[k]*model.base.day_scale[begin+s]-native_day[k];
                    float sc=0;for(int h=0;h<C;++h)sc+=q[h]*center[s][h];similar[s*nq+k]=sc*norm;
                }
            }
            for(int k=0;k<nq;++k){
                int j=grid[u][k];auto out=features.data()+int64_t(j)*OUTFEAT;out[0]=native[k];out[1]=native_day[k];
                out[2]=std::log1p(float(hist[sid[k]]));out[3]=f.dev[begin+sid[k]];
                double count[4]{},prior[4]{},sq[4]{},pcount[4]{},post[4]{},linear[4]{};
                for(int s=0;s<ns;++s){
                    if(!hist[s])continue;int delta=std::abs(time.session_days[begin+s]-day[k]);
                    float near=1.f/(1.f+delta/65.f);
                    float similarity=0;for(int g=0;g<nq;++g)similarity+=similar[s*nq+g]/nq;
                    float sel=std::exp(2.3f*(.65f*similar[s*nq+k]+.35f*similarity));
                    float size=1.f/(1.f+.55f*std::abs(std::log1p(float(hist[s]))-std::log1p(float(time.session_counts[begin+sid[k]]))));
                    float w[4]={1.f,near,sel*size,near*sel*size};
                    float e=shifts[s*nq+k],loglike=0,cross=0;
                    for(int l=0;l<nq;++l){
                        int row=grid[u][l];if(l==k||size_t(row)>=n0||(size_t(j)<n0&&anchor[2*row+1]<.5f))continue;
                        float x=shifts[s*nq+l],time_weight=.18f+.82f/(1.f+std::abs(day[l]-day[k])/24.f);
                        loglike+=time_weight*(x*error[l]-.5f*x*x)/.73f;cross+=time_weight*x*error[l];
                    }
                    float ratio=std::exp(std::clamp(loglike,-5.f,5.f));
                    for(int h=0;h<4;++h){count[h]+=w[h];prior[h]+=w[h]*e;sq[h]+=w[h]*e*e;
                        pcount[h]+=w[h]*ratio;post[h]+=w[h]*ratio*e;linear[h]+=w[h]*e*cross;}
                }
                for(int h=0;h<4;++h){
                    double mean=prior[h]/(count[h]+1.5);
                    out[4+4*h]=mean;out[5+4*h]=post[h]/(pcount[h]+1.5)-mean;
                    out[6+4*h]=std::sqrt(std::max(0.,sq[h]/(count[h]+1.5)-mean*mean));
                    out[7+4*h]=linear[h]/(count[h]+2.5);
                }
            }
            if(u%100000==0){std::printf("day-state users %d replay %.8g\n",u,maxerr);std::fflush(stdout);}
        }
        if(maxerr>.00025f)fail("stored original probe replay mismatch");
        std::ofstream out(argv[5],std::ios::binary);write_vector(out,features);
        std::cout<<"maximum original-replay difference "<<maxerr<<std::endl;
    }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
}
