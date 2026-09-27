#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr uint32_t kNumMovies = 17770;
constexpr char kDatasetMagic[8] = {'N', 'F', 'P', 'D', 'A', 'T', '2', '\0'};
constexpr char kBaselineMagic[8] = {'N', 'F', 'P', 'B', 'A', 'S', '1', '\0'};
constexpr char kTimeBaselineMagic[8] = {'N', 'F', 'P', 'T', 'I', 'M', '1', '\0'};

struct TrainRec {
  uint32_t user;
  uint16_t item;
  uint16_t day;
  uint8_t rating;
};

struct ProbeRec {
  uint32_t user;
  uint16_t item;
  uint16_t day;
  uint8_t rating;
};

struct PairRec {
  uint32_t user;
  uint16_t item;
  uint16_t day;
};

struct DatasetHeader {
  char magic[8];
  uint32_t users;
  uint32_t items;
  uint64_t train_count;
  uint64_t probe_count;
  uint64_t implicit_count;
  uint16_t min_day;
  uint16_t max_day;
  double mean;
};

struct BaselineHeader {
  char magic[8];
  uint32_t users;
  uint32_t items;
  double mean;
  double lambda_user;
  double lambda_item;
  uint32_t iterations;
};

struct TimeBaselineHeader {
  char magic[8];
  uint32_t users;
  uint32_t items;
  uint64_t sessions;
  uint32_t iterations;
  double mean;
};

struct TemporalIndex {
  std::vector<uint64_t> session_offsets;
  std::vector<uint16_t> session_days;
  std::vector<uint16_t> session_counts;
  std::vector<float> mean_day;
};

struct TimeBaselineModel {
  std::vector<float> user_bias;
  std::vector<float> user_alpha;
  std::vector<float> item_bias;
  std::vector<float> item_bin_bias;
  std::vector<float> user_scale;
  std::vector<float> day_bias;
  std::vector<float> day_scale;
  std::vector<float> item_frequency_bias;
};

struct Dataset {
  DatasetHeader header{};
  std::vector<uint64_t> train_offsets;
  std::vector<uint16_t> train_items;
  std::vector<uint16_t> train_days;
  std::vector<uint8_t> train_ratings;
  std::vector<ProbeRec> probe;
  std::vector<uint64_t> implicit_offsets;
  std::vector<uint16_t> implicit_items;
  std::vector<uint16_t> implicit_days;
};

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error(message);
}

template <typename T>
void write_pod(std::ofstream &out, const T &value) {
  out.write(reinterpret_cast<const char *>(&value), sizeof(T));
  if (!out) fail("failed writing output file");
}

template <typename T>
void write_vector(std::ofstream &out, const std::vector<T> &values) {
  if (!values.empty()) {
    out.write(reinterpret_cast<const char *>(values.data()),
              static_cast<std::streamsize>(values.size() * sizeof(T)));
  }
  if (!out) fail("failed writing output vector");
}

template <typename T>
void read_pod(std::ifstream &in, T &value) {
  in.read(reinterpret_cast<char *>(&value), sizeof(T));
  if (!in) fail("failed reading input file");
}

template <typename T>
void read_vector(std::ifstream &in, std::vector<T> &values, size_t size) {
  values.resize(size);
  if (size != 0) {
    in.read(reinterpret_cast<char *>(values.data()),
            static_cast<std::streamsize>(size * sizeof(T)));
  }
  if (!in) fail("failed reading input vector");
}

std::string value_after(const std::vector<std::string> &args,
                        const std::string &flag,
                        const std::string &fallback = "") {
  for (size_t i = 0; i + 1 < args.size(); ++i) {
    if (args[i] == flag) return args[i + 1];
  }
  if (!fallback.empty()) return fallback;
  fail("missing required argument " + flag);
}

uint32_t uint_after(const std::vector<std::string> &args,
                    const std::string &flag, uint32_t fallback) {
  const std::string text = value_after(args, flag, std::to_string(fallback));
  size_t used = 0;
  const auto result = std::stoul(text, &used);
  if (used != text.size() || result > std::numeric_limits<uint32_t>::max()) {
    fail("invalid integer for " + flag + ": " + text);
  }
  return static_cast<uint32_t>(result);
}

double double_after(const std::vector<std::string> &args,
                    const std::string &flag, double fallback) {
  const std::string text = value_after(args, flag, std::to_string(fallback));
  size_t used = 0;
  const double result = std::stod(text, &used);
  if (used != text.size() || !std::isfinite(result)) {
    fail("invalid number for " + flag + ": " + text);
  }
  return result;
}

// Days since 1970-01-01, adapted from Howard Hinnant's civil calendar algorithm.
int64_t days_from_civil(int year, unsigned month, unsigned day) {
  year -= month <= 2;
  const int era = (year >= 0 ? year : year - 399) / 400;
  const unsigned yoe = static_cast<unsigned>(year - era * 400);
  const unsigned adjusted_month =
      static_cast<unsigned>(static_cast<int>(month) + (month > 2 ? -3 : 9));
  const unsigned doy = (153 * adjusted_month + 2) / 5 + day - 1;
  const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
  return era * 146097 + static_cast<int>(doe) - 719468;
}

uint16_t parse_day(std::string_view text) {
  if (text.size() < 10 || text[4] != '-' || text[7] != '-') {
    fail("invalid date: " + std::string(text));
  }
  const int year = std::stoi(std::string(text.substr(0, 4)));
  const unsigned month = static_cast<unsigned>(
      std::stoi(std::string(text.substr(5, 2))));
  const unsigned day = static_cast<unsigned>(
      std::stoi(std::string(text.substr(8, 2))));
  // A few deliberately perturbed Netflix records predate 1999-12-31, so use
  // the beginning of 1999 as a stable non-negative epoch.
  const int64_t epoch = days_from_civil(1999, 1, 1);
  const int64_t value = days_from_civil(year, month, day) - epoch;
  if (value < 0 || value > std::numeric_limits<uint16_t>::max()) {
    fail("date outside supported range: " + std::string(text));
  }
  return static_cast<uint16_t>(value);
}

uint64_t pair_key(uint32_t raw_user, uint32_t movie) {
  return (static_cast<uint64_t>(raw_user) << 16U) | movie;
}

uint32_t parse_u32(std::string_view text) {
  uint64_t value = 0;
  if (text.empty()) fail("empty unsigned integer");
  for (const char c : text) {
    if (c < '0' || c > '9') fail("invalid unsigned integer");
    value = value * 10 + static_cast<unsigned>(c - '0');
    if (value > std::numeric_limits<uint32_t>::max()) {
      fail("unsigned integer overflow");
    }
  }
  return static_cast<uint32_t>(value);
}

std::unordered_set<uint64_t> load_probe_keys(const fs::path &path) {
  std::ifstream in(path);
  if (!in) fail("cannot open probe file: " + path.string());
  std::unordered_set<uint64_t> result;
  result.reserve(1500000);
  std::string line;
  uint32_t movie = 0;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) continue;
    if (line.back() == ':') {
      movie = parse_u32(std::string_view(line).substr(0, line.size() - 1));
      if (movie == 0 || movie > kNumMovies) fail("bad movie in probe file");
    } else {
      if (movie == 0) fail("probe user before movie header");
      result.insert(pair_key(parse_u32(line), movie));
    }
  }
  return result;
}

uint32_t dense_user(uint32_t raw_user, std::vector<int32_t> &mapping,
                    uint32_t &user_count) {
  if (raw_user >= mapping.size()) mapping.resize(raw_user + 1, -1);
  int32_t &slot = mapping[raw_user];
  if (slot < 0) slot = static_cast<int32_t>(user_count++);
  return static_cast<uint32_t>(slot);
}

void parse_training_file(const fs::path &path, uint32_t expected_movie,
                         const std::unordered_set<uint64_t> &probe_keys,
                         std::vector<int32_t> &user_mapping,
                         uint32_t &user_count, std::vector<TrainRec> &train,
                         std::vector<ProbeRec> &probe, uint16_t &min_day,
                         uint16_t &max_day, long double &rating_sum) {
  std::ifstream in(path);
  if (!in) fail("cannot open training file: " + path.string());
  std::string line;
  if (!std::getline(in, line)) fail("empty training file: " + path.string());
  if (!line.empty() && line.back() == '\r') line.pop_back();
  if (line.empty() || line.back() != ':' ||
      parse_u32(std::string_view(line).substr(0, line.size() - 1)) !=
          expected_movie) {
    fail("unexpected movie header in " + path.string());
  }
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    const size_t comma1 = line.find(',');
    const size_t comma2 =
        comma1 == std::string::npos ? comma1 : line.find(',', comma1 + 1);
    if (comma1 == std::string::npos || comma2 == std::string::npos) {
      fail("bad training row in " + path.string());
    }
    const uint32_t raw_user = parse_u32(std::string_view(line).substr(0, comma1));
    const uint32_t rating_value = parse_u32(
        std::string_view(line).substr(comma1 + 1, comma2 - comma1 - 1));
    if (rating_value < 1 || rating_value > 5) fail("rating outside [1,5]");
    const uint16_t day =
        parse_day(std::string_view(line).substr(comma2 + 1));
    min_day = std::min(min_day, day);
    max_day = std::max(max_day, day);
    const uint32_t user = dense_user(raw_user, user_mapping, user_count);
    if (probe_keys.count(pair_key(raw_user, expected_movie)) != 0) {
      probe.push_back({user, static_cast<uint16_t>(expected_movie), day,
                       static_cast<uint8_t>(rating_value)});
    } else {
      train.push_back({user, static_cast<uint16_t>(expected_movie), day,
                       static_cast<uint8_t>(rating_value)});
      rating_sum += rating_value;
    }
  }
}

std::vector<PairRec> load_qualifying(const fs::path &path,
                                     std::vector<int32_t> &user_mapping,
                                     uint32_t &user_count) {
  std::ifstream in(path);
  if (!in) fail("cannot open qualifying file: " + path.string());
  std::vector<PairRec> result;
  result.reserve(3000000);
  std::string line;
  uint32_t movie = 0;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) continue;
    if (line.back() == ':') {
      movie = parse_u32(std::string_view(line).substr(0, line.size() - 1));
      if (movie == 0 || movie > kNumMovies) fail("bad qualifying movie");
      continue;
    }
    const size_t comma = line.find(',');
    if (movie == 0 || comma == std::string::npos) fail("bad qualifying row");
    const uint32_t raw_user = parse_u32(std::string_view(line).substr(0, comma));
    if (raw_user >= user_mapping.size() || user_mapping[raw_user] < 0) {
      fail("qualifying set contains user absent from training set");
    }
    const uint32_t user = dense_user(raw_user, user_mapping, user_count);
    const uint16_t day = parse_day(std::string_view(line).substr(comma + 1));
    result.push_back({user, static_cast<uint16_t>(movie), day});
  }
  return result;
}

std::string movie_filename(uint32_t movie) {
  std::string number = std::to_string(movie);
  return "mv_" + std::string(7 - number.size(), '0') + number + ".txt";
}

void command_ingest(const std::vector<std::string> &args) {
  const fs::path training_dir = value_after(args, "--training-dir");
  const fs::path probe_path = value_after(args, "--probe");
  const fs::path qualifying_path = value_after(args, "--qualifying");
  const fs::path output_path = value_after(args, "--output");

  std::cerr << "loading probe keys...\n";
  const auto probe_keys = load_probe_keys(probe_path);
  std::cerr << "probe pairs: " << probe_keys.size() << '\n';

  std::vector<int32_t> user_mapping;
  std::vector<TrainRec> raw_train;
  std::vector<ProbeRec> probe;
  raw_train.reserve(100000000);
  probe.reserve(probe_keys.size());
  uint32_t user_count = 0;
  uint16_t min_day = std::numeric_limits<uint16_t>::max();
  uint16_t max_day = 0;
  long double rating_sum = 0;

  for (uint32_t movie = 1; movie <= kNumMovies; ++movie) {
    parse_training_file(training_dir / movie_filename(movie), movie, probe_keys,
                        user_mapping, user_count, raw_train, probe, min_day,
                        max_day, rating_sum);
    if (movie % 1000 == 0 || movie == kNumMovies) {
      std::cerr << "parsed movies " << movie << '/' << kNumMovies
                << ", ratings " << (raw_train.size() + probe.size()) << '\n';
    }
  }
  if (probe.size() != probe_keys.size()) {
    fail("found " + std::to_string(probe.size()) + " probe ratings, expected " +
         std::to_string(probe_keys.size()));
  }

  std::cerr << "loading qualifying interactions...\n";
  std::vector<PairRec> qualifying =
      load_qualifying(qualifying_path, user_mapping, user_count);

  std::vector<uint64_t> train_offsets(user_count + 1, 0);
  for (const auto &row : raw_train) ++train_offsets[row.user + 1];
  std::partial_sum(train_offsets.begin(), train_offsets.end(),
                   train_offsets.begin());
  std::vector<uint64_t> train_cursor = train_offsets;
  std::vector<uint16_t> train_items(raw_train.size());
  std::vector<uint16_t> train_days(raw_train.size());
  std::vector<uint8_t> train_ratings(raw_train.size());
  for (const auto &row : raw_train) {
    const uint64_t pos = train_cursor[row.user]++;
    train_items[pos] = row.item;
    train_days[pos] = row.day;
    train_ratings[pos] = row.rating;
  }
  struct LocalTrain {
    uint16_t item;
    uint16_t day;
    uint8_t rating;
  };
  std::vector<LocalTrain> local_train;
  for (uint32_t user = 0; user < user_count; ++user) {
    const uint64_t begin = train_offsets[user];
    const uint64_t end = train_offsets[user + 1];
    local_train.clear();
    local_train.reserve(static_cast<size_t>(end - begin));
    for (uint64_t pos = begin; pos < end; ++pos) {
      local_train.push_back(
          {train_items[pos], train_days[pos], train_ratings[pos]});
    }
    std::sort(local_train.begin(), local_train.end(),
              [](const LocalTrain &a, const LocalTrain &b) {
                return a.day != b.day ? a.day < b.day : a.item < b.item;
              });
    for (size_t j = 0; j < local_train.size(); ++j) {
      train_items[begin + j] = local_train[j].item;
      train_days[begin + j] = local_train[j].day;
      train_ratings[begin + j] = local_train[j].rating;
    }
  }

  std::vector<uint64_t> implicit_offsets(user_count + 1, 0);
  for (const auto &row : raw_train) ++implicit_offsets[row.user + 1];
  for (const auto &row : probe) ++implicit_offsets[row.user + 1];
  for (const auto &row : qualifying) ++implicit_offsets[row.user + 1];
  std::partial_sum(implicit_offsets.begin(), implicit_offsets.end(),
                   implicit_offsets.begin());
  std::vector<uint64_t> implicit_cursor = implicit_offsets;
  std::vector<uint16_t> implicit_items(implicit_offsets.back());
  std::vector<uint16_t> implicit_days(implicit_offsets.back());
  for (const auto &row : raw_train) {
    const uint64_t pos = implicit_cursor[row.user]++;
    implicit_items[pos] = row.item;
    implicit_days[pos] = row.day;
  }
  for (const auto &row : probe) {
    const uint64_t pos = implicit_cursor[row.user]++;
    implicit_items[pos] = row.item;
    implicit_days[pos] = row.day;
  }
  for (const auto &row : qualifying) {
    const uint64_t pos = implicit_cursor[row.user]++;
    implicit_items[pos] = row.item;
    implicit_days[pos] = row.day;
  }
  struct LocalImplicit {
    uint16_t item;
    uint16_t day;
  };
  std::vector<LocalImplicit> local_implicit;
  for (uint32_t user = 0; user < user_count; ++user) {
    const uint64_t begin = implicit_offsets[user];
    const uint64_t end = implicit_offsets[user + 1];
    local_implicit.clear();
    local_implicit.reserve(static_cast<size_t>(end - begin));
    for (uint64_t pos = begin; pos < end; ++pos) {
      local_implicit.push_back({implicit_items[pos], implicit_days[pos]});
    }
    std::sort(local_implicit.begin(), local_implicit.end(),
              [](const LocalImplicit &a, const LocalImplicit &b) {
                return a.day != b.day ? a.day < b.day : a.item < b.item;
              });
    for (size_t j = 0; j < local_implicit.size(); ++j) {
      implicit_items[begin + j] = local_implicit[j].item;
      implicit_days[begin + j] = local_implicit[j].day;
    }
  }

  DatasetHeader header{};
  std::memcpy(header.magic, kDatasetMagic, sizeof(kDatasetMagic));
  header.users = user_count;
  header.items = kNumMovies;
  header.train_count = raw_train.size();
  header.probe_count = probe.size();
  header.implicit_count = implicit_items.size();
  header.min_day = min_day;
  header.max_day = max_day;
  header.mean = static_cast<double>(rating_sum / raw_train.size());

  fs::create_directories(output_path.parent_path());
  const fs::path temporary = output_path.string() + ".tmp";
  std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
  if (!out) fail("cannot create dataset: " + temporary.string());
  write_pod(out, header);
  write_vector(out, train_offsets);
  write_vector(out, train_items);
  write_vector(out, train_days);
  write_vector(out, train_ratings);
  write_vector(out, probe);
  write_vector(out, implicit_offsets);
  write_vector(out, implicit_items);
  write_vector(out, implicit_days);
  out.close();
  if (!out) fail("failed closing dataset file");
  fs::rename(temporary, output_path);

  std::cout << std::fixed << std::setprecision(8)
            << "users=" << header.users << " items=" << header.items
            << " train=" << header.train_count
            << " probe=" << header.probe_count
            << " implicit=" << header.implicit_count
            << " mean=" << header.mean << " days=" << header.min_day << ".."
            << header.max_day << '\n';
}

Dataset load_dataset(const fs::path &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) fail("cannot open dataset: " + path.string());
  Dataset data;
  read_pod(in, data.header);
  if (std::memcmp(data.header.magic, kDatasetMagic, sizeof(kDatasetMagic)) != 0) {
    fail("unrecognized dataset format");
  }
  read_vector(in, data.train_offsets,
              static_cast<size_t>(data.header.users) + 1);
  read_vector(in, data.train_items, data.header.train_count);
  read_vector(in, data.train_days, data.header.train_count);
  read_vector(in, data.train_ratings, data.header.train_count);
  read_vector(in, data.probe, data.header.probe_count);
  read_vector(in, data.implicit_offsets,
              static_cast<size_t>(data.header.users) + 1);
  read_vector(in, data.implicit_items, data.header.implicit_count);
  read_vector(in, data.implicit_days, data.header.implicit_count);
  return data;
}

double clipped(double value) { return std::clamp(value, 1.0, 5.0); }

double baseline_rmse(const Dataset &data, const std::vector<float> &user_bias,
                     const std::vector<float> &item_bias) {
  long double squared_error = 0;
  for (const auto &row : data.probe) {
    const double prediction =
        clipped(data.header.mean + user_bias[row.user] + item_bias[row.item - 1]);
    const double error = static_cast<double>(row.rating) - prediction;
    squared_error += error * error;
  }
  return std::sqrt(static_cast<double>(squared_error / data.probe.size()));
}

void save_baseline(const fs::path &path, const Dataset &data,
                   const std::vector<float> &user_bias,
                   const std::vector<float> &item_bias, double lambda_user,
                   double lambda_item, uint32_t iterations) {
  BaselineHeader header{};
  std::memcpy(header.magic, kBaselineMagic, sizeof(kBaselineMagic));
  header.users = data.header.users;
  header.items = data.header.items;
  header.mean = data.header.mean;
  header.lambda_user = lambda_user;
  header.lambda_item = lambda_item;
  header.iterations = iterations;
  fs::create_directories(path.parent_path());
  const fs::path temporary = path.string() + ".tmp";
  std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
  if (!out) fail("cannot create baseline model");
  write_pod(out, header);
  write_vector(out, user_bias);
  write_vector(out, item_bias);
  out.close();
  if (!out) fail("failed closing baseline model");
  fs::rename(temporary, path);
}

void command_baseline(const std::vector<std::string> &args) {
  const fs::path data_path = value_after(args, "--data");
  const fs::path model_path = value_after(args, "--model");
  const uint32_t iterations = uint_after(args, "--iterations", 10);
  const double lambda_user = double_after(args, "--lambda-user", 10.0);
  const double lambda_item = double_after(args, "--lambda-item", 25.0);
  const Dataset data = load_dataset(data_path);
  std::vector<float> user_bias(data.header.users, 0.0F);
  std::vector<float> item_bias(data.header.items, 0.0F);
  std::vector<double> item_sums(data.header.items);
  std::vector<uint64_t> item_counts(data.header.items);

  for (uint32_t iteration = 1; iteration <= iterations; ++iteration) {
    std::fill(item_sums.begin(), item_sums.end(), 0.0);
    std::fill(item_counts.begin(), item_counts.end(), 0);
    for (uint32_t user = 0; user < data.header.users; ++user) {
      for (uint64_t pos = data.train_offsets[user];
           pos < data.train_offsets[user + 1]; ++pos) {
        const uint32_t item = data.train_items[pos] - 1;
        item_sums[item] += data.train_ratings[pos] - data.header.mean -
                           user_bias[user];
        ++item_counts[item];
      }
    }
    for (uint32_t item = 0; item < data.header.items; ++item) {
      item_bias[item] = static_cast<float>(
          item_sums[item] / (lambda_item + item_counts[item]));
    }
    for (uint32_t user = 0; user < data.header.users; ++user) {
      double sum = 0;
      const uint64_t begin = data.train_offsets[user];
      const uint64_t end = data.train_offsets[user + 1];
      for (uint64_t pos = begin; pos < end; ++pos) {
        sum += data.train_ratings[pos] - data.header.mean -
               item_bias[data.train_items[pos] - 1];
      }
      user_bias[user] =
          static_cast<float>(sum / (lambda_user + end - begin));
    }
    const double rmse = baseline_rmse(data, user_bias, item_bias);
    std::cout << "iteration=" << iteration << " probe_rmse=" << std::fixed
              << std::setprecision(8) << rmse << '\n';
  }
  save_baseline(model_path, data, user_bias, item_bias, lambda_user,
                lambda_item, iterations);
}

void command_eval_baseline(const std::vector<std::string> &args) {
  const fs::path data_path = value_after(args, "--data");
  const fs::path model_path = value_after(args, "--model");
  const Dataset data = load_dataset(data_path);
  std::ifstream in(model_path, std::ios::binary);
  if (!in) fail("cannot open baseline model");
  BaselineHeader header{};
  read_pod(in, header);
  if (std::memcmp(header.magic, kBaselineMagic, sizeof(kBaselineMagic)) != 0 ||
      header.users != data.header.users || header.items != data.header.items) {
    fail("baseline model is incompatible with dataset");
  }
  std::vector<float> user_bias;
  std::vector<float> item_bias;
  read_vector(in, user_bias, header.users);
  read_vector(in, item_bias, header.items);
  const double rmse = baseline_rmse(data, user_bias, item_bias);
  std::cout << std::fixed << std::setprecision(8) << "probe_rmse=" << rmse
            << " count=" << data.probe.size() << '\n';
}

TemporalIndex build_temporal_index(const Dataset &data) {
  TemporalIndex index;
  index.session_offsets.resize(static_cast<size_t>(data.header.users) + 1, 0);
  index.mean_day.resize(data.header.users, 0.0F);
  for (uint32_t user = 0; user < data.header.users; ++user) {
    const uint64_t begin = data.implicit_offsets[user];
    const uint64_t end = data.implicit_offsets[user + 1];
    uint64_t sessions = 0;
    uint64_t day_sum = 0;
    uint16_t previous = std::numeric_limits<uint16_t>::max();
    for (uint64_t pos = begin; pos < end; ++pos) {
      const uint16_t day = data.implicit_days[pos];
      day_sum += day;
      if (pos == begin || day != previous) ++sessions;
      previous = day;
    }
    index.session_offsets[user + 1] = sessions;
    if (end != begin) {
      index.mean_day[user] =
          static_cast<float>(static_cast<double>(day_sum) / (end - begin));
    }
  }
  std::partial_sum(index.session_offsets.begin(), index.session_offsets.end(),
                   index.session_offsets.begin());
  index.session_days.resize(index.session_offsets.back());
  index.session_counts.resize(index.session_offsets.back());
  for (uint32_t user = 0; user < data.header.users; ++user) {
    const uint64_t begin = data.implicit_offsets[user];
    const uint64_t end = data.implicit_offsets[user + 1];
    uint64_t session = index.session_offsets[user];
    uint64_t pos = begin;
    while (pos < end) {
      const uint16_t day = data.implicit_days[pos];
      uint64_t next = pos + 1;
      while (next < end && data.implicit_days[next] == day) ++next;
      index.session_days[session] = day;
      const uint64_t count = next - pos;
      index.session_counts[session] = static_cast<uint16_t>(
          std::min<uint64_t>(count, std::numeric_limits<uint16_t>::max()));
      ++session;
      pos = next;
    }
  }
  return index;
}

constexpr uint32_t kTimeBins = 30;
constexpr uint32_t kFrequencyBins = 8;

uint32_t time_bin(const Dataset &data, uint16_t day) {
  const uint32_t span =
      static_cast<uint32_t>(data.header.max_day - data.header.min_day) + 1;
  const uint32_t shifted = day - data.header.min_day;
  return std::min<uint32_t>(kTimeBins - 1, shifted * kTimeBins / span);
}

uint32_t frequency_bin(uint16_t count) {
  if (count <= 1) return 0;
  const auto value = static_cast<uint32_t>(
      std::floor(std::log(static_cast<double>(count)) / std::log(6.76)));
  return std::min<uint32_t>(kFrequencyBins - 1, value);
}

float time_deviation(uint16_t day, float mean_day) {
  const float delta = static_cast<float>(day) - mean_day;
  if (delta == 0.0F) return 0.0F;
  return std::copysign(std::pow(std::abs(delta), 0.4F), delta);
}

uint64_t find_session(const TemporalIndex &index, uint32_t user,
                      uint16_t day) {
  const uint64_t begin = index.session_offsets[user];
  const uint64_t end = index.session_offsets[user + 1];
  const auto first = index.session_days.begin() + static_cast<ptrdiff_t>(begin);
  const auto last = index.session_days.begin() + static_cast<ptrdiff_t>(end);
  const auto found = std::lower_bound(first, last, day);
  if (found == last || *found != day) fail("rating day absent from implicit set");
  return static_cast<uint64_t>(found - index.session_days.begin());
}

double time_baseline_prediction(const Dataset &data,
                                const TemporalIndex &index,
                                const TimeBaselineModel &model, uint32_t user,
                                uint16_t item_one_based, uint16_t day) {
  const uint32_t item = item_one_based - 1;
  const uint64_t session = find_session(index, user, day);
  const uint32_t tbin = time_bin(data, day);
  const uint32_t fbin = frequency_bin(index.session_counts[session]);
  const float stable_item =
      model.item_bias[item] + model.item_bin_bias[item * kTimeBins + tbin];
  const float scale = model.user_scale[user] + model.day_scale[session];
  return data.header.mean + model.user_bias[user] +
         model.user_alpha[user] * time_deviation(day, index.mean_day[user]) +
         model.day_bias[session] + stable_item * scale +
         model.item_frequency_bias[item * kFrequencyBins + fbin];
}

double time_baseline_rmse(const Dataset &data, const TemporalIndex &index,
                          const TimeBaselineModel &model) {
  long double squared_error = 0;
  for (const auto &row : data.probe) {
    const double prediction = clipped(time_baseline_prediction(
        data, index, model, row.user, row.item, row.day));
    const double error = static_cast<double>(row.rating) - prediction;
    squared_error += error * error;
  }
  return std::sqrt(static_cast<double>(squared_error / data.probe.size()));
}

void save_time_baseline(const fs::path &path, const Dataset &data,
                        const TemporalIndex &index,
                        const TimeBaselineModel &model, uint32_t iterations) {
  TimeBaselineHeader header{};
  std::memcpy(header.magic, kTimeBaselineMagic, sizeof(kTimeBaselineMagic));
  header.users = data.header.users;
  header.items = data.header.items;
  header.sessions = index.session_days.size();
  header.iterations = iterations;
  header.mean = data.header.mean;
  fs::create_directories(path.parent_path());
  const fs::path temporary = path.string() + ".tmp";
  std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
  if (!out) fail("cannot create temporal baseline model");
  write_pod(out, header);
  write_vector(out, model.user_bias);
  write_vector(out, model.user_alpha);
  write_vector(out, model.item_bias);
  write_vector(out, model.item_bin_bias);
  write_vector(out, model.user_scale);
  write_vector(out, model.day_bias);
  write_vector(out, model.day_scale);
  write_vector(out, model.item_frequency_bias);
  out.close();
  if (!out) fail("failed closing temporal baseline model");
  fs::rename(temporary, path);
}

TimeBaselineModel load_time_baseline(const fs::path &path, const Dataset &data,
                                     const TemporalIndex &index,
                                     TimeBaselineHeader &header) {
  std::ifstream in(path, std::ios::binary);
  if (!in) fail("cannot open temporal baseline model");
  read_pod(in, header);
  if (std::memcmp(header.magic, kTimeBaselineMagic,
                  sizeof(kTimeBaselineMagic)) != 0 ||
      header.users != data.header.users || header.items != data.header.items ||
      header.sessions != index.session_days.size()) {
    fail("temporal baseline model is incompatible with dataset");
  }
  TimeBaselineModel model;
  read_vector(in, model.user_bias, header.users);
  read_vector(in, model.user_alpha, header.users);
  read_vector(in, model.item_bias, header.items);
  read_vector(in, model.item_bin_bias,
              static_cast<size_t>(header.items) * kTimeBins);
  read_vector(in, model.user_scale, header.users);
  read_vector(in, model.day_bias, header.sessions);
  read_vector(in, model.day_scale, header.sessions);
  read_vector(in, model.item_frequency_bias,
              static_cast<size_t>(header.items) * kFrequencyBins);
  return model;
}

void command_time_baseline(const std::vector<std::string> &args) {
  const fs::path data_path = value_after(args, "--data");
  const fs::path model_path = value_after(args, "--model");
  const uint32_t iterations = uint_after(args, "--iterations", 40);
  const double rate_scale = double_after(args, "--rate-scale", 1.0);
  const Dataset data = load_dataset(data_path);
  const TemporalIndex index = build_temporal_index(data);
  std::cerr << "sessions=" << index.session_days.size() << '\n';

  TimeBaselineModel model;
  model.user_bias.assign(data.header.users, 0.0F);
  model.user_alpha.assign(data.header.users, 0.0F);
  model.item_bias.assign(data.header.items, 0.0F);
  model.item_bin_bias.assign(
      static_cast<size_t>(data.header.items) * kTimeBins, 0.0F);
  model.user_scale.assign(data.header.users, 1.0F);
  model.day_bias.assign(index.session_days.size(), 0.0F);
  model.day_scale.assign(index.session_days.size(), 0.0F);
  model.item_frequency_bias.assign(
      static_cast<size_t>(data.header.items) * kFrequencyBins, 0.0F);

  const float lr_bu = static_cast<float>(2.67e-3 * rate_scale);
  const float lr_but = static_cast<float>(2.57e-3 * rate_scale);
  const float lr_alpha = static_cast<float>(3.11e-6 * rate_scale);
  const float lr_bi = static_cast<float>(0.488e-3 * rate_scale);
  const float lr_bin = static_cast<float>(0.115e-3 * rate_scale);
  const float lr_cu = static_cast<float>(5.64e-3 * rate_scale);
  const float lr_cut = static_cast<float>(1.03e-3 * rate_scale);
  const float lr_freq = static_cast<float>(2.36e-3 * rate_scale);
  constexpr float reg_bu = 0.0255F;
  constexpr float reg_but = 0.00231F;
  constexpr float reg_alpha = 3.95F;
  constexpr float reg_bi = 0.0255F;
  constexpr float reg_bin = 0.0929F;
  constexpr float reg_cu = 0.0476F;
  constexpr float reg_cut = 0.0190F;
  constexpr float reg_freq = 1.10e-8F;

  std::vector<uint32_t> users(data.header.users);
  std::iota(users.begin(), users.end(), 0);
  std::mt19937 random(2009);
  for (uint32_t iteration = 1; iteration <= iterations; ++iteration) {
    std::shuffle(users.begin(), users.end(), random);
    long double train_squared_error = 0;
    for (const uint32_t user : users) {
      uint64_t session = index.session_offsets[user];
      const uint64_t session_end = index.session_offsets[user + 1];
      for (uint64_t pos = data.train_offsets[user];
           pos < data.train_offsets[user + 1]; ++pos) {
        const uint16_t day = data.train_days[pos];
        while (session < session_end && index.session_days[session] < day) {
          ++session;
        }
        if (session == session_end || index.session_days[session] != day) {
          fail("training day absent from temporal index");
        }
        const uint32_t item = data.train_items[pos] - 1;
        const uint32_t tbin = time_bin(data, day);
        const uint32_t fbin = frequency_bin(index.session_counts[session]);
        const size_t item_bin_pos =
            static_cast<size_t>(item) * kTimeBins + tbin;
        const size_t frequency_pos =
            static_cast<size_t>(item) * kFrequencyBins + fbin;
        const float dev = time_deviation(day, index.mean_day[user]);
        const float stable_item =
            model.item_bias[item] + model.item_bin_bias[item_bin_pos];
        const float scale =
            model.user_scale[user] + model.day_scale[session];
        const float prediction =
            static_cast<float>(data.header.mean) + model.user_bias[user] +
            model.user_alpha[user] * dev + model.day_bias[session] +
            stable_item * scale + model.item_frequency_bias[frequency_pos];
        const float error = static_cast<float>(data.train_ratings[pos]) - prediction;
        train_squared_error += static_cast<double>(error) * error;

        model.user_bias[user] +=
            lr_bu * (error - reg_bu * model.user_bias[user]);
        model.user_alpha[user] +=
            lr_alpha * (error * dev - reg_alpha * model.user_alpha[user]);
        model.day_bias[session] +=
            lr_but * (error - reg_but * model.day_bias[session]);
        model.item_bias[item] +=
            lr_bi * (error * scale - reg_bi * model.item_bias[item]);
        model.item_bin_bias[item_bin_pos] +=
            lr_bin *
            (error * scale - reg_bin * model.item_bin_bias[item_bin_pos]);
        model.user_scale[user] +=
            lr_cu *
            (error * stable_item - reg_cu * (model.user_scale[user] - 1.0F));
        model.day_scale[session] +=
            lr_cut *
            (error * stable_item - reg_cut * model.day_scale[session]);
        model.item_frequency_bias[frequency_pos] +=
            lr_freq * (error - reg_freq * model.item_frequency_bias[frequency_pos]);
      }
    }
    const double train_rmse = std::sqrt(
        static_cast<double>(train_squared_error / data.header.train_count));
    const double probe_rmse = time_baseline_rmse(data, index, model);
    std::cout << "iteration=" << iteration << " train_rmse=" << std::fixed
              << std::setprecision(8) << train_rmse
              << " probe_rmse=" << probe_rmse << '\n';
  }
  save_time_baseline(model_path, data, index, model, iterations);
}

void command_eval_time_baseline(const std::vector<std::string> &args) {
  const fs::path data_path = value_after(args, "--data");
  const fs::path model_path = value_after(args, "--model");
  const Dataset data = load_dataset(data_path);
  const TemporalIndex index = build_temporal_index(data);
  TimeBaselineHeader header{};
  const TimeBaselineModel model =
      load_time_baseline(model_path, data, index, header);
  const double rmse = time_baseline_rmse(data, index, model);
  std::cout << std::fixed << std::setprecision(8) << "probe_rmse=" << rmse
            << " count=" << data.probe.size()
            << " iterations=" << header.iterations << '\n';
}

void usage() {
  std::cerr
      << "usage:\n"
      << "  netflix ingest --training-dir DIR --probe FILE --qualifying FILE "
         "--output FILE\n"
      << "  netflix baseline --data FILE --model FILE [--iterations N] "
         "[--lambda-user X] [--lambda-item X]\n"
      << "  netflix eval-baseline --data FILE --model FILE\n"
      << "  netflix time-baseline --data FILE --model FILE "
         "[--iterations N] [--rate-scale X]\n"
      << "  netflix eval-time-baseline --data FILE --model FILE\n";
}

}  // namespace

int main(int argc, char **argv) {
  try {
    if (argc < 2) {
      usage();
      return 2;
    }
    const std::string command = argv[1];
    std::vector<std::string> args;
    for (int i = 2; i < argc; ++i) args.emplace_back(argv[i]);
    if (command == "ingest") {
      command_ingest(args);
    } else if (command == "baseline") {
      command_baseline(args);
    } else if (command == "eval-baseline") {
      command_eval_baseline(args);
    } else if (command == "time-baseline") {
      command_time_baseline(args);
    } else if (command == "eval-time-baseline") {
      command_eval_time_baseline(args);
    } else {
      usage();
      return 2;
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
