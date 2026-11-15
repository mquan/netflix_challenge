// Training-only user preference projection onto frozen movie vectors.
// Usage: content_profile DATA ITEM_VECTORS DIM OUT_PREFIX
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

struct Header {
  char magic[8];
  uint32_t users, items;
  uint64_t train_count, probe_count, implicit_count;
  uint16_t min_day, max_day;
  double mean;
};
struct Probe {
  uint32_t user;
  uint16_t item, day;
  uint8_t rating;
  char pad[3];
};
static_assert(sizeof(Header) == 56);
static_assert(sizeof(Probe) == 12);

template <typename T> void read(std::ifstream& f, std::vector<T>& x, uint64_t n) {
  x.resize(n);
  f.read(reinterpret_cast<char*>(x.data()), n * sizeof(T));
  if (!f) throw std::runtime_error("truncated input");
}
template <typename T> void write(const std::string& path, const std::vector<T>& x) {
  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot write " + path);
  f.write(reinterpret_cast<const char*>(x.data()), x.size() * sizeof(T));
  if (!f) throw std::runtime_error("write failed " + path);
}

int main(int argc, char** argv) {
  try {
    if (argc != 5) throw std::runtime_error("usage: content_profile DATA ITEM_VECTORS DIM OUT_PREFIX");
    const int dim = std::stoi(argv[3]);
    if (dim < 1 || dim > 1024) throw std::runtime_error("invalid dimension");
    std::ifstream f(argv[1], std::ios::binary);
    if (!f) throw std::runtime_error("cannot read dataset");
    Header h{}; f.read(reinterpret_cast<char*>(&h), sizeof(h));
    if (!f || std::memcmp(h.magic, "NFPDAT2", 7)) throw std::runtime_error("invalid dataset");
    std::vector<uint64_t> offsets;
    std::vector<uint16_t> items;
    std::vector<uint8_t> ratings;
    std::vector<Probe> probe;
    read(f, offsets, uint64_t(h.users) + 1);
    read(f, items, h.train_count);
    f.seekg(h.train_count * sizeof(uint16_t), std::ios::cur); // train days
    read(f, ratings, h.train_count);
    read(f, probe, h.probe_count);
    if (offsets.back() != h.train_count) throw std::runtime_error("bad offsets");

    std::ifstream vf(argv[2], std::ios::binary | std::ios::ate);
    if (!vf || vf.tellg() != std::streamoff(uint64_t(h.items) * dim * sizeof(float)))
      throw std::runtime_error("movie vector shape mismatch");
    vf.seekg(0);
    std::vector<float> v; read(vf, v, uint64_t(h.items) * dim);
    for (float x : v) if (!std::isfinite(x)) throw std::runtime_error("nonfinite movie vector");

    // Shrunk user and item means remove rating-scale and popularity effects.
    std::vector<double> item_sum(h.items, 0), user_sum(h.users, 0);
    std::vector<uint32_t> item_n(h.items, 0);
    for (uint32_t u = 0; u < h.users; ++u) {
      for (uint64_t p = offsets[u]; p < offsets[u+1]; ++p) {
        if (items[p] < 1 || items[p] > h.items || ratings[p] < 1 || ratings[p] > 5)
          throw std::runtime_error("invalid training row");
        item_sum[items[p]-1] += ratings[p]; ++item_n[items[p]-1];
        user_sum[u] += ratings[p];
      }
    }
    std::vector<float> item_mean(h.items), user_mean(h.users);
    for (uint32_t i = 0; i < h.items; ++i)
      item_mean[i] = float((item_sum[i] + 25*h.mean) / (item_n[i] + 25));
    for (uint32_t u = 0; u < h.users; ++u)
      user_mean[u] = float((user_sum[u] + 5*h.mean) / (offsets[u+1]-offsets[u] + 5));

    std::vector<float> profile(uint64_t(h.users) * dim, 0), support(h.users, 0);
    for (uint32_t u = 0; u < h.users; ++u) {
      float* row = profile.data() + uint64_t(u)*dim;
      const uint64_t n = offsets[u+1] - offsets[u];
      for (uint64_t p = offsets[u]; p < offsets[u+1]; ++p) {
        const uint32_t i = items[p]-1;
        const float residual = ratings[p] - user_mean[u] - item_mean[i] + float(h.mean);
        const float* movie = v.data() + uint64_t(i)*dim;
        for (int j = 0; j < dim; ++j) row[j] += residual * movie[j];
      }
      const float denom = float(n + 30); // shrink sparse user profiles
      for (int j = 0; j < dim; ++j) row[j] /= denom;
      support[u] = std::log1p(float(n));
    }

    std::vector<float> affinity(h.probe_count), row_support(h.probe_count);
    for (uint64_t p = 0; p < h.probe_count; ++p) {
      const auto& q = probe[p];
      if (q.user >= h.users || q.item < 1 || q.item > h.items)
        throw std::runtime_error("invalid probe pair");
      const float* a = profile.data() + uint64_t(q.user)*dim;
      const float* b = v.data() + uint64_t(q.item-1)*dim;
      double dot = 0;
      for (int j = 0; j < dim; ++j) dot += double(a[j])*b[j];
      affinity[p] = float(dot);
      row_support[p] = support[q.user];
    }
    write(std::string(argv[4])+".affinity.f32", affinity);
    write(std::string(argv[4])+".support.f32", row_support);
    std::cout << "users=" << h.users << " items=" << h.items
              << " probe=" << h.probe_count << " dim=" << dim << '\n';
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n'; return 1;
  }
}
