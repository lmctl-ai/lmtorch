#pragma once
#include <lmcpp/lmcpp.hpp>
#include <cstdint>
#include <string>
#include <vector>
namespace torchcli {
struct TensorData { std::string dtype; std::vector<int64_t> shape; std::vector<uint8_t> bytes; };
struct TensorRef { std::string path, key; };
TensorRef parse_ref(const std::string& ref);
TensorData read_tensor(const TensorRef& ref);
std::string write_tensor(const std::string& path, const TensorData& tensor);
lmcpp::Json descriptor(const TensorRef& ref, const TensorData& tensor);
}
