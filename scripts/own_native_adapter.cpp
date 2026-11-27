#include "own_chronology_windows.hpp"
namespace adapter {
constexpr int L=16;
using Small=std::array<float,L>;
struct Grad {Small x{};float acc=.18f;};
struct Example{Window win;std::array<float,D> context{};};
static void update(Grad& p,const Small& gradient,float weight,float lr,float reg){
    float square=0;for(float v:gradient)square+=v*v;
    p.acc+=square*weight*weight/L;
    float rate=lr/std::sqrt(p.acc);
    for(int j=0;j<L;++j)p.x[j]+=rate*(weight*gradient[j]-reg*p.x[j]);
}
struct Net{
    const Data& a;const float *input,*output;
    std::vector<Grad> inc,implicit,out;std::vector<std::array<float,5>> intercept;std::vector<float> ia;
    Net(const Data& d,const float* in,const float* read):a(d),input(in),output(read),inc(d.nm*5),
       implicit(d.nm),out(d.nm*5),intercept(d.nm),ia(d.nm,2){}
    void sample(int u,int end,int n,int vis,const int* rows,const int* m,const int* day,const float* original,
              int first,int last,bool fit,float lr,float* pred,float* replay){
        int lo=a.off[u],cut=*std::min_element(day,day+n);Small delta{},grad{};std::vector<float> weights(end-lo);
        float mass=0,sdays=0;int count=0;
        for(int j=lo;j<end;++j){
            bool hide=false;if(fit)for(int k=vis;k<n;++k)hide|=rows[k]==j;
            if(hide)continue;float t=1/(1+std::abs(int(a.day[j])-cut)/52.f);
            weights[j-lo]=t;mass+=t*t;sdays+=a.day[j];++count;
            auto& v=inc[int(a.rating[j]-1)*a.nm+a.item[j]].x;
            for(int h=0;h<L;++h)delta[h]+=(h<L/2?1.f:t)*v[h];
        }
        float scale=1/std::sqrt(float(std::max(1,count))),recent=1/std::sqrt(std::max(1.f,mass));
        float peer=1/std::sqrt(float(std::max(1,n-1)));
        for(int h=0;h<L;++h)delta[h]*=h<L/2?scale:recent;
        float pre[H];cblas_sgemv(CblasRowMajor,CblasNoTrans,H,D,1.f,a.proj,D,original,1,0.f,pre,1);
        for(int h=0;h<H;++h)pre[h]+=a.hb[h];
        int width=H;float frozen[H];vvtanhf(frozen,pre,&width);
        for(int k=first;k<last;++k){
            if(!fit&&rows[k]<0)continue;int movie=m[k];auto state=delta;
            for(int l=0;l<n;++l)if(l!=k){auto& v=implicit[m[l]].x;
                for(int h=0;h<L;++h)state[h]+=peer*v[h];}
            float h[H],next[H];std::copy(pre,pre+H,next);
            cblas_sgemv(CblasRowMajor,CblasNoTrans,H,L,1.f,input,L,state.data(),1,1.f,next,1);vvtanhf(h,next,&width);
            Small read{};cblas_sgemv(CblasRowMajor,CblasNoTrans,L,H,1.f,output,H,h,1,0.f,read.data(),1);
            float md=count?sdays/count:cut,d=(day[k]-md)/a.sd,dev=std::copysign(std::pow(std::abs(d),.4f),d);
            std::array<float,5> pi{},base{};float mu=0;
            for(int r=0;r<5;++r){
                auto old=a.op+(int64_t(movie)*5+r)*H;float bias=a.bias[movie*5+r]+a.ub[u*5+r]+dev*a.drift[u*5+r];
                pi[r]=bias+intercept[movie][r]+cblas_sdot(H,old,1,h,1)+cblas_sdot(L,out[movie*5+r].x.data(),1,read.data(),1);
                if(!fit)base[r]=bias+cblas_sdot(H,old,1,frozen,1);
            }
            auto normal=[](std::array<float,5>& z){float mx=*std::max_element(z.begin(),z.end()),den=0,v=0;
                for(float& p:z)den+=(p=std::exp(p-mx));for(int r=0;r<5;++r){z[r]/=den;v+=(r+1)*z[r];}return v;};
            mu=normal(pi);
            if(!fit){float mz=normal(base);pred[rows[k]]=mu-mz;if(replay)replay[rows[k]]=mz;continue;}
            float err=a.rating[rows[k]]-mu,g[H]{},sq=0;Small readgrad{};std::array<float,5> dz;
            for(int r=0;r<5;++r){float z=.40f*((a.rating[rows[k]]==r+1)-pi[r])+.85f*err*pi[r]*(r+1-mu);
                dz[r]=z;sq+=z*z;auto old=a.op+(int64_t(movie)*5+r)*H;
                for(int h=0;h<H;++h)g[h]+=z*old[h];
                for(int j=0;j<L;++j)readgrad[j]+=z*out[movie*5+r].x[j];
                update(out[movie*5+r],read,z,lr,.012f);}
            ia[movie]+=sq/5;for(int r=0;r<5;++r)intercept[movie][r]+=.45f*lr*(dz[r]-.05f*intercept[movie][r])/std::sqrt(ia[movie]);
            cblas_sgemv(CblasRowMajor,CblasTrans,L,H,1.f,output,H,readgrad.data(),1,1.f,g,1);
            for(int j=0;j<H;++j)g[j]*=(1-h[j]*h[j]);
            cblas_sgemv(CblasRowMajor,CblasTrans,H,L,1.f,input,L,g,1,0.f,grad.data(),1);
            for(int l=0;l<n;++l)if(l!=k)update(implicit[m[l]],grad,peer,lr,.009f);
        }
        if(fit){
            for(int j=lo;j<end;++j)if(weights[j-lo]){
                Small v=grad;for(int h=0;h<L;++h)v[h]*=h<L/2?scale:weights[j-lo]*recent;
                update(inc[int(a.rating[j]-1)*a.nm+a.item[j]],v,1.f,lr,.0012f);
            }
        }
    }
};
}
extern "C" void native_adapter(int nu,int nm,int nq,float sd,const int64_t* off,const int16_t* item,const int16_t* day,
    const int8_t* star,const int16_t* bound,const int* grid,const int16_t* qi,const int16_t* qd,const int8_t* anchor,
    const int* qidx,const float* enc,const float* proj,const float* hb,const float* op,const float* bias,const float* ub,
    const float* drift,const float* in,const float* read,float* out){
    using namespace adapter;Data a{nu,nm,sd,off,item,day,star,bound,grid,qi,qd,anchor,qidx,enc,proj,hb,op,bias,ub,drift};
    constexpr int NW=4;std::vector<Example> cases(int64_t(nu)*NW);std::mt19937 rng(2026116);
    for(int u=0;u<nu;++u)for(int c=0;c<NW;++c){
        auto& e=cases[u*NW+c];make_window(a,u,c,rng,e.win,nullptr,e.context.data());}
    std::vector<int> users(nu);std::iota(users.begin(),users.end(),0);Net model(a,in,read);int snap=0;
    for(int ep=1;ep<=36;++ep){
        std::shuffle(users.begin(),users.end(),rng);float lr=.006f+.027f*(1+std::cos(ep*3.14159f/36))/2;
        for(int z=0;z<std::min(200000,nu);++z){
            int u=users[z];auto& e=cases[u*NW+rng()%NW];auto& w=e.win;if(!w.count)continue;
            int m[Q],d[Q];for(int k=0;k<w.count;++k){m[k]=item[w.rows[k]];d[k]=day[w.rows[k]];}
            int k=w.visible+rng()%(w.count-w.visible);
            model.sample(u,w.end,w.count,w.visible,w.rows.data(),m,d,e.context.data(),k,k+1,true,lr,nullptr,nullptr);
        }
        if(ep==8||ep==18||ep==36){
            float* dest=out+int64_t(snap++)*nq;
            for(int u=0;u<nu;++u){int m[Q],d[Q],idx[Q],n=raw_count(a,u);bool needed=false;
                for(int k=0;k<n;++k){int j=grid[u*Q+k];idx[k]=qidx[j];m[k]=qi[j];d[k]=qd[j];needed|=idx[k]>=0;}
                if(!needed)continue;double sum[D]{};int count=int(off[u+1]-off[u]);
                for(int j=off[u];j<off[u+1];++j){auto v=enc+(int64_t(star[j]-1)*nm+item[j])*D;
                    for(int h=0;h<D;++h)sum[h]+=v[h];}
                float context[D];for(int h=0;h<D;++h)context[h]=sum[h]/std::sqrt(float(std::max(1,count)));
                model.sample(u,off[u+1],n,0,idx,m,d,context,0,n,false,0,dest,snap==1?out+int64_t(3)*nq:nullptr);
            }
        }
        if(ep%3==0){std::printf("native adapter epoch %d\n",ep);std::fflush(stdout);}
    }
}
