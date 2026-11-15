// Reconstruct gtakacs/netflix_prize split_3 (commit 49c9b367), using IDs/dates
// only. The original Probe is entirely absent from that split's fitting data.
// RNG semantics follow rand 0.9.2 (MIT/Apache-2.0): PCG32 seed expansion,
// ChaCha12, and StandardUniform f64. See artifacts/public-bank/rand__* sources.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

struct ChaCha12 {
  std::array<uint32_t,16> state{}, buffer{};
  unsigned position=16;
  static uint32_t rot(uint32_t v,unsigned n) { return (v<<n)|(v>>(32-n)); }
  static void quarter(uint32_t& a,uint32_t& b,uint32_t& c,uint32_t& d) {
    a+=b; d=rot(d^a,16); c+=d; b=rot(b^c,12);
    a+=b; d=rot(d^a,8); c+=d; b=rot(b^c,7);
  }
  explicit ChaCha12(const std::array<uint32_t,8>& key) {
    state={0x61707865,0x3320646e,0x79622d32,0x6b206574,0,0,0,0,0,0,0,0,0,0,0,0};
    std::copy(key.begin(),key.end(),state.begin()+4);
  }
  explicit ChaCha12(uint64_t seed):ChaCha12(expand(seed)) {}
  static std::array<uint32_t,8> expand(uint64_t seed) {
    std::array<uint32_t,8> key{};
    for(auto& k:key) {
      seed=seed*0x5851f42d4c957f2dULL+0xa17654e46fbe17f3ULL;
      uint32_t x=uint32_t(((seed>>18)^seed)>>27), r=uint32_t(seed>>59);
      k=(x>>r)|(x<<((-r)&31));
    }
    return key;
  }
  uint32_t next() {
    if(position==16) {
      buffer=state;
      for(int round=0;round<6;++round) {
        quarter(buffer[0],buffer[4],buffer[8],buffer[12]);
        quarter(buffer[1],buffer[5],buffer[9],buffer[13]);
        quarter(buffer[2],buffer[6],buffer[10],buffer[14]);
        quarter(buffer[3],buffer[7],buffer[11],buffer[15]);
        quarter(buffer[0],buffer[5],buffer[10],buffer[15]);
        quarter(buffer[1],buffer[6],buffer[11],buffer[12]);
        quarter(buffer[2],buffer[7],buffer[8],buffer[13]);
        quarter(buffer[3],buffer[4],buffer[9],buffer[14]);
      }
      for(unsigned j=0;j<16;++j) buffer[j]+=state[j];
      if(++state[12]==0) ++state[13];
      position=0;
    }
    return buffer[position++];
  }
  uint64_t next64() { uint64_t low=next(); return low|(uint64_t(next())<<32); }
  double uniform() { return double(next64()>>11)*(1.0/9007199254740992.0); }
};

void self_test() {
  // Exact two-stage vector in rand 0.9.2 src/rngs/std.rs test_stdrng_construction.
  ChaCha12 r(std::array<uint32_t,8>{1,23,456,7890,0,0,0,0});
  if(r.next64()!=10719222850664546238ULL) throw std::runtime_error("StdRng vector 0 mismatch");
  std::array<uint32_t,8> key{}; for(auto& k:key) k=r.next();
  ChaCha12 second(key);
  if(second.next64()!=14064965282130556830ULL) throw std::runtime_error("StdRng vector 1 mismatch");
}

struct Header { char magic[8]; uint32_t users,items; uint64_t train,probe,implicit;
                uint16_t min_day,max_day; double mean; };
struct Probe { uint32_t user; uint16_t item,day; uint8_t rating; char pad[3]; };
static_assert(sizeof(Header)==56 && sizeof(Probe)==12);

template<class T> void write(const std::string& path,const std::vector<T>& x) {
  std::ofstream out(path,std::ios::binary);
  out.write(reinterpret_cast<const char*>(x.data()),x.size()*sizeof(T));
  if(!out) throw std::runtime_error("cannot write "+path);
}

int main(int argc,char** argv) {
 try {
  self_test();
  if(argc==2 && std::string(argv[1])=="--self-test") { std::cout<<"rand 0.9.2 reference vectors pass\n"; return 0; }
  if(argc!=4) throw std::runtime_error("usage: public_split3_map DATA DENSE_TO_RAW_U32 OUTPUT_PREFIX");
  int fd=open(argv[1],O_RDONLY); if(fd<0) throw std::runtime_error("cannot open data");
  struct stat st{}; if(fstat(fd,&st)) throw std::runtime_error("cannot stat data");
  const auto* bytes=static_cast<const uint8_t*>(mmap(nullptr,st.st_size,PROT_READ,MAP_PRIVATE,fd,0));
  if(bytes==MAP_FAILED) throw std::runtime_error("cannot map data");
  const Header& h=*reinterpret_cast<const Header*>(bytes);
  if(std::memcmp(h.magic,"NFPDAT2",7)||h.train>=UINT32_MAX) throw std::runtime_error("invalid dataset");
  const auto* offsets=reinterpret_cast<const uint64_t*>(bytes+sizeof(Header));
  const auto* items=reinterpret_cast<const uint16_t*>(offsets+h.users+1);
  const auto* days=items+h.train;
  const auto* probe=reinterpret_cast<const Probe*>(reinterpret_cast<const uint8_t*>(days+h.train)+h.train);
  std::vector<uint32_t> raw(h.users);
  std::ifstream uf(argv[2],std::ios::binary); uf.read(reinterpret_cast<char*>(raw.data()),raw.size()*4);
  if(!uf||uf.peek()!=EOF) throw std::runtime_error("invalid user map length");
  std::vector<uint32_t> source_order(h.users),source_user(h.users);
  std::iota(source_order.begin(),source_order.end(),0);
  std::sort(source_order.begin(),source_order.end(),[&](uint32_t a,uint32_t b){return raw[a]<raw[b];});
  for(uint32_t i=0;i<h.users;++i) {
    if(i && raw[source_order[i]]==raw[source_order[i-1]]) throw std::runtime_error("duplicate raw user");
    source_user[source_order[i]]=i;
  }
  std::vector<uint8_t> in_probe(h.users,0);
  for(uint64_t j=0;j<h.probe;++j) in_probe[probe[j].user]=1;
  ChaCha12 rng(42);
  std::vector<uint32_t> random(h.train);
  uint64_t draws=0;
  for(uint32_t u:source_order) if(in_probe[u])
    for(uint64_t j=offsets[u];j<offsets[u+1];++j) { random[j]=rng.next(); ++draws; }
  struct Candidate { double key; uint32_t row; };
  std::vector<Candidate> candidates;
  std::vector<uint32_t> local;
  for(uint32_t u:source_order) if(in_probe[u]) {
    local.resize(offsets[u+1]-offsets[u]); std::iota(local.begin(),local.end(),uint32_t(offsets[u]));
    std::sort(local.begin(),local.end(),[&](uint32_t a,uint32_t b) {
      if(days[a]!=days[b]) return days[a]>days[b];
      if(random[a]!=random[b]) return random[a]<random[b];
      return a<b;
    });
    size_t g=0; unsigned day_rank=0;
    while(g<local.size() && day_rank<2) {
      size_t e=g+1; while(e<local.size() && days[local[e]]==days[local[g]]) ++e;
      for(size_t k=g+1;k<e;++k) {
        double x=std::max(rng.uniform(),std::numeric_limits<double>::min());
        candidates.push_back({std::log(x)*double(e-g-1),local[k]});
      }
      ++day_rank; g=e;
    }
  }
  std::vector<uint32_t>().swap(random);
  const size_t added=size_t(std::round(.25*double(h.probe)));
  if(candidates.size()<=added) throw std::runtime_error("not enough candidates");
  std::nth_element(candidates.begin(),candidates.begin()+added,candidates.end(),
                   [](const Candidate& a,const Candidate& b){return a.key>b.key;});
  std::vector<uint32_t> moved; moved.reserve(added);
  for(size_t k=0;k<added;++k) moved.push_back(candidates[k].row);
  std::sort(moved.begin(),moved.end());
  std::cout<<"rng_u32_draws="<<draws<<" candidates="<<candidates.size()<<" moved="<<added<<'\n';
  std::vector<Candidate>().swap(candidates);
  struct Query { uint32_t item,user,original; };
  std::vector<Query> queries; queries.reserve(h.probe+added);
  for(uint32_t j=0;j<h.probe;++j) queries.push_back({probe[j].item,source_user[probe[j].user],j});
  uint32_t u=0;
  for(uint32_t row:moved) {
    while(row>=offsets[u+1]) ++u;
    queries.push_back({items[row],source_user[u],UINT32_MAX});
  }
  std::sort(queries.begin(),queries.end(),[](const Query& a,const Query& b) {
    return a.item!=b.item ? a.item<b.item : a.user<b.user;
  });
  std::vector<uint32_t> mapping(h.probe,UINT32_MAX),source_items(queries.size()),source_users(queries.size());
  for(uint32_t j=0;j<queries.size();++j) {
    const auto& q=queries[j];
    if(j && q.item==queries[j-1].item && q.user==queries[j-1].user) throw std::runtime_error("duplicate query");
    if(q.original!=UINT32_MAX) mapping[q.original]=j;
    source_items[j]=q.item; source_users[j]=q.user;
  }
  for(auto j:mapping) if(j==UINT32_MAX) throw std::runtime_error("unmapped Probe row");
  std::string prefix=argv[3];
  write(prefix+".u32",mapping); write(prefix+".moved.u32",moved);
  write(prefix+".source-items.u32",source_items); write(prefix+".source-users.u32",source_users);
  std::cout<<"mapped="<<mapping.size()<<" source_rows="<<queries.size()<<" rating_values_read=0\n";
  munmap(const_cast<uint8_t*>(bytes),st.st_size); close(fd);
 } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
