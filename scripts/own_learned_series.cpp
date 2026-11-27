#include "own_title_neighbors.cpp"
extern "C" void edge_moments(int nu,int nm,int width,const int* off,const int16_t* items,const int16_t* days,const int8_t* star,
        const float* residual,const int* graph,float* moments){
    std::vector<int> owner(nm,-1),index(nm);
    for(int u=0;u<nu;++u){
        if((uint32_t(u)*2654435761U)>>31)continue;
        for(int j=off[u];j<off[u+1];++j){int m=items[j];owner[m]=u;index[m]=j;}
        for(int j=off[u];j<off[u+1];++j){
            int m=items[j];for(int h=0;h<width;++h){
                int k=m*width+h,l=graph[k];if(l<0)break;if(owner[l]!=u)continue;
                int t=index[l],dt=int(days[j])-days[t];float a=residual[j],b=residual[t],s=star[j],r=star[t],*v=moments+int64_t(k)*10;
                v[0]+=1;v[1]+=(dt==0);v[2]+=1.f/(1+std::abs(dt)/30.f);v[3]+=(dt>0);v[4]+=a*b;
                v[5]+=a*a;v[6]+=b*b;v[7]+=(r==s);v[8]+=(r==5&&s==5);v[9]+=(r<=2&&s<=2);
            }
        }
    }
}
