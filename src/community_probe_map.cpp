// Map original probe.txt row order to this repository's processed Probe order.
// Reads original training files only for the first-seen user map and label check.
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

struct Header {
  char magic[8]; uint32_t users, items;
  uint64_t train_count, probe_count, implicit_count;
  uint16_t min_day, max_day; double mean;
};
struct Probe { uint32_t user; uint16_t item, day; uint8_t rating; char pad[3]; };
static_assert(sizeof(Header)==56 && sizeof(Probe)==12);

static uint32_t number(const char* p) {
  uint32_t x=0;
  if (*p<'0'||*p>'9') throw std::runtime_error("expected integer");
  while (*p>='0'&&*p<='9') { x=10*x+uint32_t(*p-'0'); ++p; }
  return x;
}
static uint64_t key(uint32_t user,uint32_t item) { return (uint64_t(user)<<16)|item; }

int main(int argc,char** argv) {
  try {
    if (argc!=5) throw std::runtime_error("usage: community_probe_map DATA PROBE_TXT TRAINING_DIR OUTPUT_U32");
    std::unordered_map<uint64_t,uint32_t> original;
    original.reserve(1500000);
    std::vector<std::vector<uint32_t>> probe_groups(17771);
    std::ifstream probe_text(argv[2]);
    if (!probe_text) throw std::runtime_error("cannot read probe.txt");
    std::string line; uint32_t movie=0, rows=0;
    while (std::getline(probe_text,line)) {
      if (line.empty()) continue;
      if (line.back()==':') { movie=number(line.c_str()); continue; }
      if (!movie) throw std::runtime_error("probe row before movie");
      if (movie>17770) throw std::runtime_error("bad probe movie");
      probe_groups[movie].push_back(number(line.c_str()));
      ++rows;
    }
    // Source ingest groups numerically by movie ID. The original probe.txt
    // itself is ordered lexicographically by movie filename (1, 10, 1000...).
    uint32_t source_row=0;
    for (uint32_t i=1;i<=17770;++i)
      for (uint32_t raw:probe_groups[i])
        if (!original.emplace(key(raw,i),source_row++).second)
          throw std::runtime_error("duplicate probe pair");
    if (source_row!=rows) throw std::runtime_error("probe count mismatch");
    std::vector<uint8_t> source_rating(rows,0);
    std::vector<int32_t> raw_to_dense(2700000,-1);
    std::vector<uint32_t> dense_to_raw;
    dense_to_raw.reserve(500000);
    const std::string dir=argv[3];
    for (movie=1;movie<=17770;++movie) {
      std::string id=std::to_string(movie);
      std::string path=dir+"/mv_"+std::string(7-id.size(),'0')+id+".txt";
      std::ifstream f(path);
      if (!f) throw std::runtime_error("cannot read "+path);
      std::getline(f,line); // movie header
      while (std::getline(f,line)) {
        const char* s=line.c_str();
        uint32_t raw=number(s);
        if (raw>=raw_to_dense.size()) throw std::runtime_error("raw user out of range");
        if (raw_to_dense[raw]<0) {
          raw_to_dense[raw]=int32_t(dense_to_raw.size());
          dense_to_raw.push_back(raw);
        }
        auto found=original.find(key(raw,movie));
        if (found!=original.end()) {
          const char* comma=std::strchr(s,',');
          if (!comma) throw std::runtime_error("bad training row");
          uint32_t rating=number(comma+1);
          if (rating<1||rating>5||source_rating[found->second])
            throw std::runtime_error("bad or duplicate source rating");
          source_rating[found->second]=uint8_t(rating);
        }
      }
    }
    std::ifstream bin(argv[1],std::ios::binary);
    Header h{}; bin.read(reinterpret_cast<char*>(&h),sizeof(h));
    if (!bin||std::memcmp(h.magic,"NFPDAT2",7)||h.probe_count!=rows||h.users!=dense_to_raw.size())
      throw std::runtime_error("dataset shape mismatch");
    bin.seekg(sizeof(Header)+sizeof(uint64_t)*(uint64_t(h.users)+1)+5*h.train_count);
    std::vector<uint32_t> mapping(rows);
    std::vector<uint8_t> seen(rows,0);
    for (uint32_t j=0;j<rows;++j) {
      Probe q{}; bin.read(reinterpret_cast<char*>(&q),sizeof(q));
      if (!bin||q.user>=dense_to_raw.size()) throw std::runtime_error("bad processed probe");
      auto found=original.find(key(dense_to_raw[q.user],q.item));
      if (found==original.end()||seen[found->second]++||source_rating[found->second]!=q.rating)
        throw std::runtime_error("pair, uniqueness or rating mismatch");
      mapping[j]=found->second;
    }
    for (uint8_t s:seen) if (s!=1) throw std::runtime_error("mapping not bijective");
    std::ofstream out(argv[4],std::ios::binary);
    out.write(reinterpret_cast<const char*>(mapping.data()),mapping.size()*sizeof(uint32_t));
    if (!out) throw std::runtime_error("cannot write map");
    std::cout<<"mapped "<<rows<<" probe rows, "<<h.users<<" users; pairs and ratings agree\n";
  } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
