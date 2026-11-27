#include <algorithm>
#include <cmath>
#include <cstdint>

extern "C" void peers(int n,int stride,const int* starts,const int* order,const int16_t* days,
                      const int16_t* movies,const float* coords,const float* x,float* out) {
    for(int u=0;u<n;++u) {
        for(int i=starts[u];i<starts[u+1];++i) {
            int a=order[i],m=movies[a];
            double sums[4][9]={},w[4]={};
            int mind=10000;
            for(int j=starts[u];j<starts[u+1];++j) {
                if(j==i)continue;
                int b=order[j],delta=std::abs(days[a]-days[b]);
                mind=std::min(mind,delta);
                float sim=0;
                for(int h=0;h<48;++h)sim+=coords[m*48+h]*coords[movies[b]*48+h];
                double ww[4]={1.,1./(1.+delta/7.),std::exp((sim-1)*3.),delta==0?1.:0.};
                for(int t=0;t<4;++t) {
                    w[t]+=ww[t];
                    for(int h=0;h<stride;++h)sums[t][h]+=ww[t]*x[b*stride+h];
                    sums[t][stride]+=ww[t]*x[b*stride]*x[b*stride];
                }
            }
            float* row=out+int64_t(a)*(4*(stride+2)+1);
            for(int t=0;t<4;++t) {
                float* target=row+t*(stride+2);
                target[0]=std::log1p(w[t]);
                for(int h=0;h<stride;++h)target[h+1]=w[t]>0?sums[t][h]/w[t]-x[a*stride+h]:0;
                if(w[t])target[stride+1]=std::sqrt(std::max(0.,sums[t][stride]/w[t]-std::pow(sums[t][0]/w[t],2)));
            }
            row[4*(stride+2)]=std::log1p(mind);
        }
    }
}
