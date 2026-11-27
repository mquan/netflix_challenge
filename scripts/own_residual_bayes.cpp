#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

// Every native training row must be an observed coordinate with the same dated star
// in the corresponding CFNADE history before its encoder contribution is removed.
extern "C" int verify_rows(int nu, int nm, const int* map, const uint64_t* off,
    const uint16_t* item,const uint16_t* day,const uint8_t* rating,
    const int* cf_off,const int16_t* cf_item,const int16_t* cf_day,const int8_t* cf_rating){
    std::vector<int> position(nm,-1);
    for(int v=0;v<nu;++v){
        int u=map[v];
        for(int j=cf_off[u];j<cf_off[u+1];++j)position[cf_item[j]]=j;
        for(uint64_t j=off[v];j<off[v+1];++j){
            int k=position[item[j]-1];
            if(k<0 || int(day[j])-314!=cf_day[k] || rating[j]!=cf_rating[k])return v+1;
        }
        for(int j=cf_off[u];j<cf_off[u+1];++j)position[cf_item[j]]=-1;
    }
    return 0;
}

extern "C" void predict_rows(int n, float sd, float avg, const int* user,const uint16_t* movies,
    const uint16_t* days,const uint8_t* star,const int* counts,const float* mean_day,const float* total,
    const float* enc,const float* dec,const float* hidden_bias,const float* bias,const float* user_bias,
    const float* drift,float* out){
    const int dim=320;
    float h[320],logits[5];
    for(int j=0;j<n;++j){
        const int u=user[j],m=movies[j]-1,d=int(days[j])-314,r=star?star[j]:0;
        const int nr=counts[u]-(r>0);
        float norm=1.f/std::sqrt(float(std::max(1,nr)));
        const float* all=total+int64_t(u)*dim;
        if(r){
            const float* subtract=enc+(int64_t(m)*5+r-1)*dim;
            for(int k=0;k<dim;++k)h[k]=(all[k]-subtract[k])*norm+hidden_bias[k];
        }else{
            for(int k=0;k<dim;++k)h[k]=all[k]*norm+hidden_bias[k];
        }
        vvtanhf(h,h,&dim);
        float center=r?(nr?(mean_day[u]*counts[u]-d)/nr:avg):mean_day[u];
        float t=(d-center)/sd;
        float dev=std::copysign(std::pow(std::abs(t),.4f),t);
        for(int k=0;k<5;++k){
            logits[k]=cblas_sdot(dim,h,1,dec+(int64_t(m)*5+k)*dim,1)+bias[m*5+k]+user_bias[u*5+k]+drift[u*5+k]*dev;
        }
        float mx=*std::max_element(logits,logits+5),s=0,v=0;
        for(int k=0;k<5;++k){float p=std::exp(logits[k]-mx);s+=p;v+=(k+1)*p;}
        out[j]=v/s;
    }
}
