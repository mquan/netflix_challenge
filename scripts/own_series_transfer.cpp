#include "own_series_ridge.cpp"
extern "C" void series_rows(int nu,int n,int train,float sd,const int* off,const int16_t* item,const int16_t* day,
    const int8_t* rating,const float* residual,const int* user,const int* row,const int16_t* qi,const int16_t* qd,
    const float* mu,const int* nb,const float* strength,const int16_t* year,const float* geo,const float* md,const float* total,
    const float* ep,const float* op,const float* eb,const float* b,const float* ub,const float* drift,float* out){
    using namespace series;std::vector<int> pos(17770),stamp(17770,-1);int previous=-1;
    for(int g=0;g<n;++g){
        int u=user[g],j=row[g],m=train?item[j]:qi[j],d=train?day[j]:qd[j];
        if(u!=previous){for(int h=off[u];h<off[u+1];++h){pos[item[h]]=h;stamp[item[h]]=u;}previous=u;}
        float q=train?rating[j]-residual[j]:mu[j];float* dest=out+int64_t(g)*72;
        int nseen=0;
        for(int t=0;t<K&&nseen<6;++t){
            int other=nb[m*K+t];if(other<0||stamp[other]!=u)continue;int k=pos[other];float* f=dest+nseen++*12;
            float e=train?leave_both(u,j,k,off[u+1]-off[u],item,day,rating,sd,md,total,ep,op,eb,b,ub,drift):residual[k];
            float diff=day[k]-d,sim=0;for(int h=0;h<16;++h)sim+=geo[m*16+h]*geo[other*16+h];f[0]=1;f[1]=rating[k]-3.5f;
            f[2]=e;f[3]=std::asinh(diff/30.f);f[4]=(diff==0);f[5]=(year[other]-year[m])/20.f;
            f[6]=strength[m*K+t];f[7]=t>=12;f[8]=sim;f[9]=q-3.5f;f[10]=std::log1p(float(off[u+1]-off[u]))/6.f;f[11]=rating[k]-e-q;
        }
    }
}
