#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

constexpr int DIM=24,W=3*DIM+15;
extern "C" void cohort_design(int n,int movie,const int* u,const int16_t* d,const int* off,
       const int16_t* peers,const int16_t* days,const float* anchors,const float* coords,float* x){
    for(int j=0;j<n;++j){
        float* z=x+int64_t(j)*W;
        int user=u[j],day=d[j];float mass[3]{},ac=0,ad=0,mind=4096;
        const float* target=coords+movie*DIM;
        for(int i=off[user];i<off[user+1];++i) {
            int m=peers[i];if(m==movie)continue;
            int dt=std::abs(days[i]-day);mind=std::min(mind,float(dt));
            const float* peer=coords+m*DIM;
            float dot=0;
            for(int k=0;k<DIM;++k)dot+=peer[k]*target[k];
            float time=1.f/(1+dt/25.f),similar=std::clamp(dot/DIM,-1.f,1.f);
            float w[3]={1.f,time,.5f+.5f*similar};
            for(int b=0;b<3;++b){
                mass[b]+=w[b];for(int k=0;k<DIM;++k)z[b*DIM+k]+=w[b]*peer[k];
            }
            float* v=z+3*DIM;
            v[0]+=similar;v[1]+=time*similar;v[2]+=float(dt==0);v[3]+=float(days[i]<day);
            v[4]+=similar*similar;
            if(std::isfinite(anchors[i])){
                float e=anchors[i];
                ac++;ad+=time;
                v[5]+=e;v[6]+=time*e;v[7]+=similar*e;v[8]+=e*e;
            }
        }
        for(int b=0;b<3;++b)for(int k=0;k<DIM;++k)z[b*DIM+k]/=std::sqrt(mass[b]+1);
        float* v=z+3*DIM;
        v[0]/=1+mass[0];v[1]/=1+mass[1];v[4]/=1+mass[0];
        v[5]/=1+ac;v[6]/=1+ad;v[7]/=1+ac;v[8]/=1+ac;
        v[9]=std::log1p(mass[0]);v[10]=std::log1p(mass[1]);v[11]=std::log1p(ac);v[12]=std::log1p(mind);
        v[13]=(day-1500.f)/600.f;v[14]=1;
    }
}
