#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

// Return peer row numbers in the same unlabeled address table, nearest date first.
extern "C" void query_sets(int n,int width,const int* qu,const int16_t* qi,const int16_t* qd,
                           const int* off,const int* sorted,int* neighbors) {
    std::vector<std::pair<int,int>> pool;
    for(int j=0;j<n;++j){
        int user=qu[j];pool.clear();
        for(int a=off[user];a<off[user+1];++a){
            int other=sorted[a];
            if(qi[j]==qi[other])continue;
            pool.emplace_back(std::abs(qd[other]-qd[j]),other);
        }
        int take=std::min(int(pool.size()),width);
        std::partial_sort(pool.begin(),pool.begin()+take,pool.end());
        for(int k=0;k<take;++k)neighbors[int64_t(j)*width+k]=pool[k].second;
    }
}
