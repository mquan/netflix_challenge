// Compile with NETFLIX_NATIVE_SOURCE set to an isolated frozen netflix.cpp.
// The training entry point is renamed; this executable only inspects data.
#define main netflix_unused_training_main
#include NETFLIX_NATIVE_SOURCE
#undef main

int main(int argc, char **argv) {
  try {
    if (argc != 2) throw std::runtime_error("expected dataset path");
    const auto data = load_dataset(argv[1]);
    const auto temporal = build_temporal_index(data);
    std::cout << "{\"fingerprint\":" << joint_fingerprint(data)
              << ",\"users\":" << data.header.users
              << ",\"items\":" << data.header.items
              << ",\"train_count\":" << data.header.train_count
              << ",\"query_count\":" << data.header.probe_count
              << ",\"sessions\":" << temporal.session_days.size() << "}\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
