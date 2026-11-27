#include "own_suffix_attention.cpp"
extern "C" void hybrid_masks(int b,int hist,int targets,int seed,int nm,const int* users,const int64_t* off,
    const int16_t* movie,const int16_t* day,const int8_t* star,const int16_t* boundary,const int* cohort,const int* hidden,
    int* hi,int* hr,int* hd,int* ti,int* td,int* labels,float* counts,const float* enc,float* context,float* mean){
    std::vector<int> end(b),rows(b*targets,-1);
    detailed_masks(end.data(),rows.data(),b,hist,targets,seed,users,off,movie,day,star,boundary,cohort,hidden,hi,hr,hd,ti,td,labels,counts);
    for(int k=0;k<b;++k){
        int lo=off[users[k]];double sum[96]{},days=0;
        for(int j=lo;j<end[k];++j){auto v=enc+(int64_t(star[j]-1)*nm+movie[j])*96;
            for(int h=0;h<96;++h)sum[h]+=v[h];days+=day[j];}
        for(int t=0;t<targets;++t){int j=rows[k*targets+t];if(j<0)continue;
            auto v=enc+(int64_t(star[j]-1)*nm+movie[j])*96;for(int h=0;h<96;++h)sum[h]-=v[h];days-=day[j];}
        float n=std::max(1.f,counts[k]);mean[k]=days/n;
        for(int h=0;h<96;++h)context[k*96+h]=sum[h]/std::sqrt(n);
    }
}
