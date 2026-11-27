#include <Accelerate/Accelerate.h>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <vector>

extern "C" void jacobian_kernel(int lo,int hi,int dim,int width,const uint64_t* off,const uint16_t* item,
    const uint16_t* day,const int* ids,const int16_t* qi,const int16_t* qd,const float* vec,float* out){
    std::vector<float> x,local,g(dim*dim),g2(dim*dim),z(width*dim),w(width*dim);
    for(int u=lo;u<hi;++u){
        const int* grid=ids+int64_t(u)*width;
        int nq=0;float mean=0;
        for(;nq<width && grid[nq]>=0;++nq)mean+=qd[grid[nq]];
        if(nq==0)continue;
        mean/=nq;
        int n=off[u+1]-off[u];
        x.resize(n*dim);local.resize(n*dim);
        for(int j=0;j<n;++j){
            auto r=off[u]+j;auto v=vec+int64_t(item[r]-1)*dim;
            float scale=std::sqrt(1.f/(1.f+std::abs(float(day[r])-mean)/30.f));
            for(int k=0;k<dim;++k){x[j*dim+k]=v[k];local[j*dim+k]=v[k]*scale;}
        }
        auto whiten=[&](const std::vector<float>& design,std::vector<float>& gram,std::vector<float>& whitened,float pen){
            std::fill(gram.begin(),gram.end(),0.f);
            for(int j=0;j<dim;++j)gram[j*dim+j]=pen;
            if(n)cblas_ssyrk(CblasRowMajor,CblasLower,CblasTrans,dim,n,1.f,design.data(),dim,1.f,gram.data(),dim);
            for(int j=0;j<dim;++j){
                auto row=gram.data()+j*dim;
                for(int k=0;k<j;++k){
                    float val=row[k];
                    for(int h=0;h<k;++h)val-=row[h]*gram[k*dim+h];
                    row[k]=val/gram[k*dim+k];
                }
                float v=row[j];for(int k=0;k<j;++k)v-=row[k]*row[k];row[j]=std::sqrt(std::max(v,.0001f));
            }
            for(int a=0;a<nq;++a){
                auto v=vec+int64_t(qi[grid[a]])*dim;auto h=whitened.data()+a*dim;
                for(int j=0;j<dim;++j){
                    float val=v[j];for(int k=0;k<j;++k)val-=gram[j*dim+k]*h[k];
                    h[j]=val/gram[j*dim+j];
                }
            }
        };
        whiten(x,g,z,16);whiten(local,g2,w,10);
        for(int a=0;a<nq;++a)for(int b=0;b<nq;++b){
            float global=0,near=0;
            for(int j=0;j<dim;++j){global+=z[a*dim+j]*z[b*dim+j];near+=w[a*dim+j]*w[b*dim+j];}
            size_t ix=((size_t(u)*width+a)*width+b)*2;
            out[ix]=global;out[ix+1]=near/(1.f+std::abs(int(qd[grid[a]])-qd[grid[b]])/30.f);
        }
    }
}
