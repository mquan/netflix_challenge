#include <cstdint>
extern "C" void ease_score(int nu,int landmarks,const int* off,const int* feat,const float* val,
       const int* qoff,const int* qrow,const int16_t* movie,const float* weights,const float* scale,float* out){
    int width=3*landmarks;
    for(int u=0;u<nu;++u)for(int g=qoff[u];g<qoff[u+1];++g){
        int j=qrow[g];const float* w=weights+int64_t(movie[j])*width;float total[3]{};
        for(int z=off[u];z<off[u+1];++z){int k=feat[z];total[k/landmarks]+=val[z]*w[k];}
        for(int c=0;c<3;++c)out[int64_t(j)*3+c]=total[c]/scale[u];
    }
}
