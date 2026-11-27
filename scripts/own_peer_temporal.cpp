#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>
extern "C" void background_rows(int n,int cap,int pass,const int* user,const uint64_t* off,const uint16_t* item,
        const uint16_t* day,const uint8_t* star,const int* median,int* movies,int* dates,int* ratings){
    for(int k=0;k<n;++k){
        int u=user[k],lo=off[u],hi=off[u+1],len=hi-lo;std::vector<int> selected,other;
        if(len<=cap){selected.resize(len);std::iota(selected.begin(),selected.end(),lo);}
        else{
            std::mt19937 rng(14621u+u*277u+pass*91673u);
            for(int j=lo;j<hi;++j)if(std::abs(int(day[j])-median[k])<=14)selected.push_back(j);else other.push_back(j);
            std::shuffle(selected.begin(),selected.end(),rng);
            if(int(selected.size())>cap/2){other.insert(other.end(),selected.begin()+cap/2,selected.end());selected.resize(cap/2);}
            std::shuffle(other.begin(),other.end(),rng);selected.insert(selected.end(),other.begin(),other.begin()+cap-selected.size());
        }
        for(int j=0;j<int(selected.size());++j){int r=selected[j],i=k*cap+j;movies[i]=item[r]-1;dates[i]=day[r]-314;ratings[i]=star[r];}
    }
}
