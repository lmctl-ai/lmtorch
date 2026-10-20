#pragma once
#include <lmcpp/lmcpp.hpp>
#include <cstdint>
#include <string>
#include <map>
#include <vector>
namespace torchcli {
struct TensorData { std::string dtype; std::vector<int64_t> shape; std::vector<uint8_t> bytes; };
struct TensorBundle { std::map<std::string, TensorData> tensors; std::map<std::string, std::string> metadata; };
struct TensorRef { std::string path, key; };
TensorRef parse_ref(const std::string& ref);
TensorData read_tensor(const TensorRef& ref);
TensorBundle read_bundle(const std::string& path);
// replace=false installs with link(2) and refuses an existing path (the lmapi operations); true renames over it (DSL saves).
std::string write_bundle(const std::string& path, const TensorBundle& bundle, bool replace = false);
std::string write_tensor(const std::string& path, const TensorData& tensor, bool replace = false);
// Throws INVALID_PATH unless the canonical target stays inside the working directory.
void check_contained(const std::string& path);
lmcpp::Json descriptor(const TensorRef& ref, const TensorData& tensor);
}
