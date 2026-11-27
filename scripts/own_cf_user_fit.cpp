#include <Accelerate/Accelerate.h>
#include <cstdint>
extern "C" void cf_equations(int users,int d,int b,const int* starts,const float* x,
                             const float* target,float* grams,float* rhs) {
    for(int u=0;u<users;++u){
        int n=starts[u+1]-starts[u];
        if(!n)continue;
        auto a=x+int64_t(starts[u])*d;
        cblas_sgemm(CblasRowMajor,CblasTrans,CblasNoTrans,d,d,n,1.f,a,d,a,d,0.f,grams+int64_t(u)*d*d,d);
        cblas_sgemm(CblasRowMajor,CblasTrans,CblasNoTrans,d,b,n,1.f,a,d,target+int64_t(starts[u])*b,b,0.f,rhs+int64_t(u)*d*b,b);
    }
}
