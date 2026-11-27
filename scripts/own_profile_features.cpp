#include <algorithm>
#include <cmath>
#include <cstdint>

// Six complementary sums of film coordinates. Remove a target rating from
// all sums before evaluating its own training example.
static inline void add(float* out, int k, const float* imp, const float* exp,
                       int r, int date, float residual, float sign) {
    float y=r-3.6f, t=(date-1500.f)/600.f;
    for(int j=0;j<k;++j) {
        out[j]+=sign*imp[j];
        out[k+j]+=sign*y*imp[j];
        out[2*k+j]+=sign*y*exp[j];
        out[3*k+j]+=sign*residual*exp[j];
        out[4*k+j]+=sign*t*imp[j];
        out[5*k+j]+=sign*t*y*exp[j];
    }
    out[6*k+r-1]+=sign;
    out[6*k+5]+=sign*t;
    out[6*k+6]+=sign*t*t;
    out[6*k+7]+=sign*residual;
}

extern "C" void profiles(int users,int k,const int32_t* starts,const int16_t* items,
        const int16_t* dates,const int8_t* ratings,const float* residuals,
        const float* imp,const float* exp,float* sums) {
    int width=6*k+8;
    for(int u=0;u<users;++u) {
        float* row=sums+int64_t(u)*width;
        for(int a=starts[u];a<starts[u+1];++a)
            add(row,k,imp+int(items[a])*k,exp+int(items[a])*k,ratings[a],dates[a],residuals[a],1.f);
    }
}

extern "C" void profile_rows(int n,int k,int item,const int32_t* users,const int16_t* dates,
        const int8_t* ratings,const float* residuals,const int32_t* counts,
        const float* sums,const float* imp,const float* exp,const float* scale,
        int training,float* design) {
    int width=6*k+8;
    for(int a=0;a<n;++a) {
        int u=users[a];
        float* row=design+int64_t(a)*(width+4);
        std::copy_n(sums+int64_t(u)*width,width,row);
        if(training) add(row,k,imp+item*k,exp+item*k,ratings[a],dates[a],residuals[a],-1.f);
        float count=std::max(0,counts[u]-training);
        float denom=std::sqrt(count+15.f), t=(dates[a]-1500.f)/600.f;
        for(int j=0;j<6*k;++j) row[j] = std::clamp(row[j]/denom*scale[j],-12.f,12.f);
        for(int j=6*k;j<width;++j) row[j]/=std::max(count,1.f);
        row[width]=std::log1p(count)/5.f;
        row[width+1]=t;
        row[width+2]=t-row[6*k+5];
        row[width+3]=1.f;
    }
}
