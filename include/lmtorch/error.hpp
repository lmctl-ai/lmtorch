#pragma once

#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>

namespace lmtorch {

enum class ErrorCode { CommandFailed, Conflict, NotFound, ValidationError };

class Error final : public std::runtime_error {
 public:
  ErrorCode code;
  nlohmann::ordered_json data;

  Error(ErrorCode code, std::string message, nlohmann::ordered_json data = nlohmann::ordered_json::object())
      : std::runtime_error(message), code(code), data(std::move(data)) {}
};

}  // namespace lmtorch
