// The `_lmcode_` bytecode host: binds schema/lmtorch.dsl.json (embedded byte for byte: schema/lmtorch.host.json merged with
// the lmcode std schema) to libtorch through the vendored lmcode-cpp VM, whose default Registry installs the std catalog.
// One process runs one or more chunks against one Torch/Out context; stdout carries exactly one JSON line.
#include "lmtorch/dsl_host.hpp"
#include "lmtorch/safetensors.hpp"
#include "dslvm.hpp"
#include "dsl_schema_embedded.hpp"
#include <torch/torch.h>
#include <torch/version.h>
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#ifndef _WIN32
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace lmtorch::dsl {
namespace {
using dslvm::Args;
using dslvm::Value;
using OJson = nlohmann::ordered_json;

[[noreturn]] void fail(const std::string& message) { throw dslvm::Error(message); }

// ---------------------------------------------------------------- host objects
// Handles never move or die during a run (deques keep addresses stable); torch.release() empties a tensor handle instead of
// freeing it, so a stale local fails with a clear message rather than reading freed memory.
struct TensorBox { torch::Tensor t; bool kept = false; };
struct BundleBox { std::map<std::string, torch::Tensor> tensors; std::map<std::string, std::string> meta; };
struct ModelBox { torch::nn::Sequential seq; std::vector<std::string> layers; };
struct OptBox { std::unique_ptr<torch::optim::Optimizer> opt; std::string kind; };
struct Out { OJson result = OJson::object(); };

struct Host {
  std::deque<TensorBox> tensors;
  std::deque<std::vector<torch::Tensor>> lists;
  std::deque<BundleBox> bundles;
  std::deque<ModelBox> models;
  std::deque<OptBox> optimizers;
  Out out;
  Value box(torch::Tensor t) {
    if (!t.defined()) fail("internal: undefined tensor result");
    tensors.push_back(TensorBox{std::move(t), false});
    return Value::object("Tensor", &tensors.back());
  }
  template <class T> Value hold(std::deque<T>& arena, const char* type, T value) {
    arena.push_back(std::move(value));
    return Value::object(type, &arena.back());
  }
};

template <class T> T* self(void* p) { return static_cast<T*>(p); }
torch::Tensor& tensor_of(void* p) {
  auto* b = self<TensorBox>(p);
  if (!b || !b->t.defined()) fail("tensor was released by torch.release(); call keep() on tensors that must outlive a release");
  return b->t;
}
torch::Tensor& T(const Value& v) { return tensor_of(v.obj); }

// ---------------------------------------------------------------- dtypes, shapes, text
const std::vector<std::string> kDTypes = {"F32", "F64", "I64", "BOOL"};
torch::ScalarType scalar_type(const std::string& d) {
  if (d == "F32") return torch::kFloat32;
  if (d == "F64") return torch::kFloat64;
  if (d == "I64") return torch::kInt64;
  if (d == "BOOL") return torch::kBool;
  fail("dtype must be one of F32, F64, I64, BOOL; found " + d);
}
std::string dtype_name(torch::ScalarType t) {
  if (t == torch::kFloat32) return "F32";
  if (t == torch::kFloat64) return "F64";
  if (t == torch::kInt64) return "I64";
  if (t == torch::kBool) return "BOOL";
  fail(std::string("unsupported tensor dtype ") + c10::toString(t));
}
std::string shape_text(const torch::Tensor& t) {
  std::string s = "[";
  for (int64_t i = 0; i < t.dim(); ++i) s += (i ? ", " : "") + std::to_string(t.size(i));
  return s + "]";
}
constexpr int64_t kMaxElements = int64_t(1) << 31;
std::vector<int64_t> dims_of(const Args& a, size_t from, size_t count, const char* who) {
  std::vector<int64_t> d;
  int64_t n = 1;
  for (size_t i = from; i < from + count; ++i) {
    if (a[i].i < 0) fail(std::string(who) + ": dimensions must be nonnegative; found " + std::to_string(a[i].i));
    if (a[i].i > 0 && n > kMaxElements / a[i].i) fail(std::string(who) + ": tensor exceeds 2^31 elements");
    n *= std::max<int64_t>(a[i].i, 1);
    d.push_back(a[i].i);
  }
  return d;
}
std::string number_text(double v, int digits) {
  if (std::isnan(v)) return "nan";
  if (std::isinf(v)) return v > 0 ? "inf" : "-inf";
  char buf[40];
  std::snprintf(buf, sizeof buf, "%.*g", digits, v);
  std::string s = buf;
  if (s.find_first_of(".eEn") == std::string::npos) s += ".0";
  return s;
}
// JSON-shaped nested lists; json=true keeps round-trip precision and writes nonfinite floats as null.
void nested_text(const torch::Tensor& t, std::string& s, bool json) {
  if (t.dim() == 0) {
    auto st = t.scalar_type();
    if (st == torch::kBool) s += t.item<bool>() ? "true" : "false";
    else if (st == torch::kInt64) s += std::to_string(t.item<int64_t>());
    else {
      double v = t.item<double>();
      if (json && !std::isfinite(v)) s += "null";
      else s += number_text(v, json ? (st == torch::kFloat32 ? 9 : 17) : 6);
    }
    return;
  }
  s += "[";
  for (int64_t i = 0; i < t.size(0); ++i) {
    if (i) s += ", ";
    nested_text(t[i], s, json);
  }
  s += "]";
}
std::string tensor_text(const torch::Tensor& t, bool json) {
  torch::NoGradGuard g;
  std::string s;
  nested_text(t.detach().cpu(), s, json);
  return s;
}

// ---------------------------------------------------------------- files (confined to the working directory)
std::string read_text(const std::string& path) {
  lmtorch::check_contained(path);
  std::error_code ec;
  if (!std::filesystem::is_regular_file(path, ec)) fail("cannot read " + path);
  if (std::filesystem::file_size(path, ec) > (uint64_t(256) << 20)) fail("cannot read " + path + ": larger than 256 MiB");
  std::ifstream in(path, std::ios::binary);
  std::ostringstream data;
  data << in.rdbuf();
  if (!in && !in.eof()) fail("cannot read " + path);
  return data.str();
}
void write_text_atomic(const std::string& path, const std::string& text) {
  if (path.empty()) fail("output path must not be empty");
  lmtorch::check_contained(path);
#ifdef _WIN32
  std::string detail;
  if (lmtorch::win_atomic_write(path, text.data(), text.size(), true, detail) != lmtorch::WinWrite::Ok) fail("cannot write " + path + ": " + detail);
#else
  std::string tmp = path + ".tmpXXXXXX";
  std::vector<char> name(tmp.begin(), tmp.end());
  name.push_back('\0');
  int fd = ::mkstemp(name.data());
  if (fd < 0) fail("cannot write " + path + ": " + std::strerror(errno));
  size_t off = 0;
  while (off < text.size()) {
    ssize_t w = ::write(fd, text.data() + off, text.size() - off);
    if (w < 0) { if (errno == EINTR) continue; ::close(fd); ::unlink(name.data()); fail("cannot write " + path); }
    off += size_t(w);
  }
  mode_t mask = ::umask(0);
  ::umask(mask);
  if (::fchmod(fd, 0644 & ~mask) != 0 || ::fsync(fd) != 0 || ::close(fd) != 0) { ::unlink(name.data()); fail("cannot write " + path); }
  if (::rename(name.data(), path.c_str()) != 0) { ::unlink(name.data()); fail("cannot write " + path + ": " + std::strerror(errno)); }
#endif
}

lmtorch::TensorData to_data(const torch::Tensor& input) {
  torch::NoGradGuard g;
  auto t = input.detach().contiguous().cpu();
  lmtorch::TensorData d;
  d.dtype = dtype_name(t.scalar_type());
  d.shape.assign(t.sizes().begin(), t.sizes().end());
  d.bytes.resize(t.nbytes());
  if (!d.bytes.empty()) std::memcpy(d.bytes.data(), t.data_ptr(), d.bytes.size());
  return d;
}
torch::Tensor from_data(const lmtorch::TensorData& d) {
  auto t = torch::empty(d.shape, torch::TensorOptions().dtype(scalar_type(d.dtype)));
  if (t.nbytes() != d.bytes.size()) fail("tensor payload has invalid size");
  if (!d.bytes.empty()) std::memcpy(t.data_ptr(), d.bytes.data(), d.bytes.size());
  return t;
}
// lmcpp errors from the safetensors layer carry a code and message; surface the message.
template <class F> auto io(F&& f) -> decltype(f()) {
  try { return f(); }
  catch (const dslvm::Error&) { throw; }
  catch (const lmcpp::Error& e) { fail(e.message); }
}
void save_bundle(const std::string& path, const std::map<std::string, torch::Tensor>& tensors, const std::map<std::string, std::string>& meta) {
  if (path.empty()) fail("output path must not be empty");
  if (tensors.empty()) fail("cannot save an empty bundle: put() at least one tensor");
  lmtorch::TensorBundle b;
  for (const auto& [name, t] : tensors) b.tensors.emplace(name, to_data(t));
  b.metadata = meta;
  io([&] { return lmtorch::write_bundle(path, b, true); });
}
lmtorch::TensorBundle load_bundle(const std::string& path) { return io([&] { return lmtorch::read_bundle(path); }); }

// ---------------------------------------------------------------- tensor literals and CSV
// Every scalar must sit at the same, final depth: "[3, [1, 2]]" and "[[1], [[2]]]" are ragged, not reshaped.
void infer(const OJson& j, size_t depth, std::vector<int64_t>& shape, std::vector<const OJson*>& leaves, size_t& leaf_depth) {
  if (j.is_array()) {
    if (leaf_depth != SIZE_MAX && depth >= leaf_depth) fail("torch.tensor: ragged nested list (a list appears where a number was expected)");
    if (depth == shape.size()) shape.push_back(int64_t(j.size()));
    else if (shape[depth] != int64_t(j.size())) fail("torch.tensor: ragged nested list (dimension " + std::to_string(depth) + " has lengths " + std::to_string(shape[depth]) + " and " + std::to_string(j.size()) + ")");
    for (const auto& x : j) infer(x, depth + 1, shape, leaves, leaf_depth);
    return;
  }
  if (leaf_depth == SIZE_MAX) leaf_depth = depth;
  if (depth != leaf_depth || depth != shape.size()) fail("torch.tensor: ragged nested list (a scalar appears where a list was expected)");
  if (!j.is_number() && !j.is_boolean()) fail("torch.tensor: elements must be numbers or booleans");
  if (j.is_number_float() && !std::isfinite(j.get<double>())) fail("torch.tensor: elements must be finite");
  leaves.push_back(&j);
}
torch::Tensor parse_tensor(const std::string& text, const std::string* dtype) {
  OJson j;
  try { j = OJson::parse(text); } catch (const std::exception& e) { fail(std::string("torch.tensor: not a JSON number or nested list: ") + e.what()); }
  std::vector<int64_t> shape;
  std::vector<const OJson*> leaves;
  size_t leaf_depth = SIZE_MAX;
  if (j.is_array() && j.empty()) shape.push_back(0);
  else infer(j, 0, shape, leaves, leaf_depth);
  int64_t expected = 1;
  for (auto d : shape) expected *= d;
  if (expected != int64_t(leaves.size())) fail("torch.tensor: ragged nested list");
  bool any_bool = false, any_float = false, any_number = false;
  for (auto* x : leaves) { if (x->is_boolean()) any_bool = true; else { any_number = true; if (x->is_number_float()) any_float = true; } }
  if (any_bool && any_number) fail("torch.tensor: cannot mix booleans and numbers");
  std::string dt = dtype ? *dtype : any_bool ? "BOOL" : (any_float || leaves.empty()) ? "F32" : "I64";
  auto st = scalar_type(dt);
  if (any_bool && st != torch::kBool) fail("torch.tensor: boolean elements need dtype BOOL");
  if (any_number && st == torch::kBool) fail("torch.tensor: numeric elements cannot be BOOL");
  auto t = torch::empty(shape, torch::TensorOptions().dtype(st));
  int64_t i = 0;
  if (st == torch::kBool) { auto* p = t.data_ptr<bool>(); for (auto* x : leaves) p[i++] = x->get<bool>(); }
  else if (st == torch::kInt64) {
    auto* p = t.data_ptr<int64_t>();
    for (auto* x : leaves) {
      if (x->is_number_float()) { double v = x->get<double>(); if (std::floor(v) != v || std::abs(v) > 9007199254740991.0) fail("torch.tensor: I64 elements must be integers"); p[i++] = int64_t(v); }
      else if (x->is_number_unsigned() && x->get<uint64_t>() > uint64_t(INT64_MAX)) fail("torch.tensor: I64 element out of range");
      else p[i++] = x->get<int64_t>();
    }
  } else if (st == torch::kFloat64) { auto* p = t.data_ptr<double>(); for (auto* x : leaves) p[i++] = x->get<double>(); }
  else {
    auto* p = t.data_ptr<float>();
    for (auto* x : leaves) { double v = x->get<double>(); if (std::abs(v) > 3.4028234663852886e38) fail("torch.tensor: F32 element out of range"); p[i++] = float(v); }
  }
  return t;
}
std::string trim(const std::string& s) {
  size_t a = s.find_first_not_of(" \t\r"), b = s.find_last_not_of(" \t\r");
  return a == std::string::npos ? "" : s.substr(a, b - a + 1);
}
torch::Tensor parse_csv(const std::string& path, bool header) {
  std::string text = read_text(path);
  if (text.rfind("\xEF\xBB\xBF", 0) == 0) text.erase(0, 3);  // a UTF-8 byte-order mark (spreadsheet exports)
  std::istringstream in(text);
  std::string line;
  std::vector<float> values;
  int64_t cols = -1, rows = 0, lineno = 0;
  bool skipped = !header;
  while (std::getline(in, line)) {
    ++lineno;
    if (trim(line).empty()) continue;
    if (!skipped) { skipped = true; continue; }
    std::vector<std::string> cells;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) cells.push_back(trim(cell));
    if (!line.empty() && line.back() == ',') cells.push_back("");
    if (cols < 0) cols = int64_t(cells.size());
    else if (int64_t(cells.size()) != cols) fail(path + ":" + std::to_string(lineno) + ": expected " + std::to_string(cols) + " columns, found " + std::to_string(cells.size()));
    for (const auto& c : cells) {
      char* end = nullptr;
      errno = 0;
      double v = c.empty() ? 0 : std::strtod(c.c_str(), &end);
      if (c.empty() || end != c.c_str() + c.size() || errno == ERANGE || !std::isfinite(v)) fail(path + ":" + std::to_string(lineno) + ": not a finite number: '" + c + "'");
      values.push_back(float(v));
    }
    ++rows;
  }
  if (rows == 0) fail(path + ": no data rows");
  return torch::from_blob(values.data(), {rows, cols}, torch::kFloat32).clone();
}
std::string csv_text(const torch::Tensor& input) {
  torch::NoGradGuard g;
  auto t = input.detach().cpu();
  if (t.dim() == 0 || t.dim() > 2) fail("write_csv needs a 1-D or 2-D tensor; found shape " + shape_text(t));
  if (t.dim() == 1) t = t.unsqueeze(1);
  std::string s;
  bool json_like = t.scalar_type() == torch::kFloat32 || t.scalar_type() == torch::kFloat64;
  for (int64_t r = 0; r < t.size(0); ++r) {
    for (int64_t c = 0; c < t.size(1); ++c) {
      if (c) s += ",";
      auto v = t[r][c];
      if (t.scalar_type() == torch::kBool) s += v.item<bool>() ? "1" : "0";
      else if (t.scalar_type() == torch::kInt64) s += std::to_string(v.item<int64_t>());
      else s += number_text(v.item<double>(), json_like && t.scalar_type() == torch::kFloat64 ? 17 : 9);
    }
    s += "\n";
  }
  return s;
}

// ---------------------------------------------------------------- registration helpers
const std::set<std::string> kEnums = {"DType"};
bool kind_ok(const std::string& expected, const Value& v) {
  if (expected == "float") return v.kind == Value::FLOAT;
  if (expected == "int") return v.kind == Value::INT;
  if (expected == "string") return v.kind == Value::STR;
  if (expected == "bool") return v.kind == Value::BOOL;
  if (kEnums.count(expected)) return v.kind == Value::STR && std::find(kDTypes.begin(), kDTypes.end(), v.s) != kDTypes.end();
  return v.kind == Value::OBJ && v.type == expected && v.obj;
}
// Layer specs are the text Model.summary() prints ("linear(2, 8)", "relu", "conv2d(1, 4, 3, stride 1, padding 0)"); a saved
// model stores them in its "layers" metadata so torch.load_model() can rebuild the architecture before loading weights.
std::string add_layer(ModelBox& m, const std::string& kind, const std::vector<int64_t>& n, double f) {
  static const std::set<std::string> known = {"linear", "conv2d", "max_pool2d", "layer_norm", "softmax", "log_softmax", "leaky_relu", "dropout", "flatten", "relu", "tanh", "sigmoid", "gelu"};
  if (!known.count(kind)) fail("unknown layer '" + kind + "'");
  auto need = [&](size_t count) { if (n.size() != count) fail("layer " + kind + " expects " + std::to_string(count) + " integer argument(s)"); };
  auto positive = [](int64_t v, const char* what) { if (v < 1 || v > 1000000) fail(std::string(what) + " must be between 1 and 1000000"); return v; };
  auto ints = [&](size_t count) { std::string s; for (size_t i = 0; i < count; ++i) s += (i ? ", " : "") + std::to_string(n[i]); return s; };
  if (kind == "linear") { need(2); m.seq->push_back(torch::nn::Linear(positive(n[0], "linear in_features"), positive(n[1], "linear out_features"))); return "linear(" + ints(2) + ")"; }
  if (kind == "conv2d") {
    if (n.size() == 3) { m.seq->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(positive(n[0], "conv2d in_channels"), positive(n[1], "conv2d out_channels"), positive(n[2], "conv2d kernel")))); return "conv2d(" + ints(3) + ")"; }
    need(5);
    if (n[4] < 0 || n[4] > 1000) fail("conv2d padding must be between 0 and 1000");
    m.seq->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(positive(n[0], "conv2d in_channels"), positive(n[1], "conv2d out_channels"), positive(n[2], "conv2d kernel")).stride(positive(n[3], "conv2d stride")).padding(n[4])));
    return "conv2d(" + ints(5) + ")";
  }
  if (kind == "max_pool2d") { need(1); m.seq->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(positive(n[0], "max_pool2d kernel")))); return "max_pool2d(" + ints(1) + ")"; }
  if (kind == "layer_norm") { need(1); m.seq->push_back(torch::nn::LayerNorm(torch::nn::LayerNormOptions({positive(n[0], "layer_norm size")}))); return "layer_norm(" + ints(1) + ")"; }
  if (kind == "softmax") { need(1); m.seq->push_back(torch::nn::Softmax(torch::nn::SoftmaxOptions(n[0]))); return "softmax(" + ints(1) + ")"; }
  if (kind == "log_softmax") { need(1); m.seq->push_back(torch::nn::LogSoftmax(torch::nn::LogSoftmaxOptions(n[0]))); return "log_softmax(" + ints(1) + ")"; }
  if (kind == "leaky_relu") { need(0); if (!std::isfinite(f)) fail("negative slope must be finite"); m.seq->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(f))); return "leaky_relu(" + number_text(f, 17) + ")"; }
  if (kind == "dropout") { need(0); if (!(f >= 0 && f < 1)) fail("dropout probability must be in [0, 1)"); m.seq->push_back(torch::nn::Dropout(torch::nn::DropoutOptions(f))); return "dropout(" + number_text(f, 17) + ")"; }
  need(0);
  if (kind == "flatten") m.seq->push_back(torch::nn::Flatten());
  else if (kind == "relu") m.seq->push_back(torch::nn::ReLU());
  else if (kind == "tanh") m.seq->push_back(torch::nn::Tanh());
  else if (kind == "sigmoid") m.seq->push_back(torch::nn::Sigmoid());
  else if (kind == "gelu") m.seq->push_back(torch::nn::GELU());
  else fail("unknown layer '" + kind + "'");
  return kind;
}
void load_weights(ModelBox& m, const lmtorch::TensorBundle& b, const std::string& path) {
  std::map<std::string, torch::Tensor> mine;
  for (const auto& kv : m.seq->named_parameters()) mine[kv.key()] = kv.value();
  for (const auto& kv : m.seq->named_buffers()) mine[kv.key()] = kv.value();
  for (const auto& kv : b.tensors)
    if (!mine.count(kv.first)) fail(path + " has tensor '" + kv.first + "' that this model does not have");
  torch::NoGradGuard g;
  for (auto& [name, t] : mine) {
    auto it = b.tensors.find(name);
    if (it == b.tensors.end()) fail(path + " is missing model tensor '" + name + "'");
    auto src = from_data(it->second);
    if (src.sizes() != t.sizes() || src.scalar_type() != t.scalar_type()) fail(path + ": tensor '" + name + "' is " + dtype_name(src.scalar_type()) + shape_text(src) + ", the model expects " + dtype_name(t.scalar_type()) + shape_text(t));
    t.copy_(src);
  }
}
// Parses "a(1, 2) -> b -> c(0.5)"; float arguments belong to leaky_relu and dropout only.
void replay_layers(ModelBox& m, const std::string& spec, const std::string& where) {
  if (spec.empty() || spec == "(empty)") fail(where + ": saved model has no layers");
  size_t pos = 0;
  while (pos <= spec.size()) {
    size_t next = spec.find(" -> ", pos);
    std::string item = spec.substr(pos, next == std::string::npos ? std::string::npos : next - pos);
    std::string kind = item;
    std::vector<int64_t> ints;
    double f = std::nan("");
    auto open = item.find('(');
    if (open != std::string::npos) {
      if (item.back() != ')') fail(where + ": malformed layer spec '" + item + "'");
      kind = item.substr(0, open);
      std::stringstream args(item.substr(open + 1, item.size() - open - 2));
      std::string a;
      while (std::getline(args, a, ',')) {
        a = trim(a);
        if (kind == "leaky_relu" || kind == "dropout") { char* end = nullptr; f = std::strtod(a.c_str(), &end); if (a.empty() || *end) fail(where + ": malformed layer spec '" + item + "'"); continue; }
        auto sp = a.rfind(' ');
        if (sp != std::string::npos) a = a.substr(sp + 1);  // "stride 1" -> "1"
        char* end = nullptr;
        long long v = std::strtoll(a.c_str(), &end, 10);
        if (a.empty() || *end) fail(where + ": malformed layer spec '" + item + "'");
        ints.push_back(v);
      }
    }
    m.layers.push_back(add_layer(m, kind, ints, f));
    if (next == std::string::npos) break;
    pos = next + 4;
  }
}
std::string first_line(std::string m) {
  auto pos = m.find('\n');
  if (pos != std::string::npos) m.resize(pos);
  if (m.size() > 300) m.resize(300);
  return m;
}
// Every entry checks its argument kinds (the VM does not) and maps libtorch and C++ exceptions to one-line host errors.
void method(dslvm::Registry& r, const std::string& type, const std::string& name, std::vector<std::string> params, std::string ret, dslvm::Method fn) {
  std::string who = type + "." + name;
  r.method(type, name, params, ret, [params, fn, who](void* obj, Args& a) {
    if (a.size() != params.size()) fail(who + ": argument count mismatch");
    for (size_t i = 0; i < params.size(); ++i)
      if (!kind_ok(params[i], a[i])) fail(who + " argument " + std::to_string(i + 1) + " must be " + params[i]);
    try { return fn(obj, a); }
    catch (const dslvm::Error&) { throw; }
    catch (const c10::Error& e) { fail(who + ": " + first_line(e.what_without_backtrace())); }
    catch (const lmcpp::Error& e) { fail(who + ": " + e.message); }
    catch (const std::bad_alloc&) { fail(who + ": out of memory"); }
    catch (const std::exception& e) { fail(who + ": " + first_line(e.what())); }
    catch (...) { fail(who + ": unknown failure"); }
  });
}
double finite(double v, const char* what) { if (!std::isfinite(v)) fail(std::string(what) + " must be finite"); return v; }
int64_t check_dim(const torch::Tensor& t, int64_t d) {
  int64_t n = std::max<int64_t>(t.dim(), 1);
  if (d < -n || d >= n) fail("dimension " + std::to_string(d) + " is out of range for a tensor of shape " + shape_text(t));
  return d;
}
void same_shape(const torch::Tensor& a, const torch::Tensor& b, const char* who) {
  if (a.sizes() != b.sizes()) fail(std::string(who) + ": target shape " + shape_text(b) + " differs from input shape " + shape_text(a));
}
torch::Tensor one_element(const torch::Tensor& t, const char* who) {
  if (t.numel() != 1) fail(std::string(who) + " needs a one-element tensor; found shape " + shape_text(t));
  return t.detach().reshape({});
}
int64_t flat_index(const torch::Tensor& t, int64_t i) {
  if (i < 0 || i >= t.numel()) fail("flat index " + std::to_string(i) + " is out of range for " + std::to_string(t.numel()) + " elements");
  return i;
}

// ---------------------------------------------------------------- the registry
void register_torch(dslvm::Registry& r, Host& h) {
  r.context("torch", "Torch");
  r.context("out", "Out");
  const std::vector<std::string> I1 = {"int"}, I2 = {"int", "int"}, I3 = {"int", "int", "int"}, I4 = {"int", "int", "int", "int"};
  const std::vector<std::vector<std::string>> dims = {I1, I2, I3, I4};
  auto tensor_ret = [&h](torch::Tensor t) { return h.box(std::move(t)); };

  // ---- Torch: process settings
  method(r, "Torch", "version", {}, "string", [](void*, Args&) {
    return Value::string(std::string("lmtorch " LMTORCH_VERSION " (libtorch ") + std::to_string(TORCH_VERSION_MAJOR) + "." + std::to_string(TORCH_VERSION_MINOR) + "." + std::to_string(TORCH_VERSION_PATCH) + ", CPU)");
  });
  method(r, "Torch", "manual_seed", {"int"}, "void", [](void*, Args& a) {
    if (a[0].i < 0) fail("seed must be nonnegative");
    torch::manual_seed(uint64_t(a[0].i));
    return Value::none();
  });
  method(r, "Torch", "set_num_threads", {"int"}, "void", [](void*, Args& a) {
    if (a[0].i < 1 || a[0].i > 256) fail("thread count must be between 1 and 256");
    at::set_num_threads(int(a[0].i));
    return Value::none();
  });
  method(r, "Torch", "grad_enabled", {"bool"}, "void", [](void*, Args& a) { at::GradMode::set_enabled(a[0].b); return Value::none(); });
  method(r, "Torch", "is_grad_enabled", {}, "bool", [](void*, Args&) { return Value::boolean(at::GradMode::is_enabled()); });

  // ---- Torch: creation (shapes are 1 to 4 int dimensions)
  for (size_t n = 1; n <= 4; ++n) {
    method(r, "Torch", "zeros", dims[n - 1], "Tensor", [n, tensor_ret](void*, Args& a) { return tensor_ret(torch::zeros(dims_of(a, 0, n, "zeros"))); });
    method(r, "Torch", "ones", dims[n - 1], "Tensor", [n, tensor_ret](void*, Args& a) { return tensor_ret(torch::ones(dims_of(a, 0, n, "ones"))); });
    method(r, "Torch", "rand", dims[n - 1], "Tensor", [n, tensor_ret](void*, Args& a) { return tensor_ret(torch::rand(dims_of(a, 0, n, "rand"))); });
    method(r, "Torch", "randn", dims[n - 1], "Tensor", [n, tensor_ret](void*, Args& a) { return tensor_ret(torch::randn(dims_of(a, 0, n, "randn"))); });
    auto full_params = dims[n - 1];
    full_params.push_back("float");
    method(r, "Torch", "full", full_params, "Tensor", [n, tensor_ret](void*, Args& a) { return tensor_ret(torch::full(dims_of(a, 0, n, "full"), a[n].f)); });
  }
  for (size_t n = 1; n <= 2; ++n) {
    auto p = dims[n - 1];
    p.insert(p.begin(), {"int", "int"});
    method(r, "Torch", "randint", p, "Tensor", [n, tensor_ret](void*, Args& a) {
      if (a[0].i >= a[1].i) fail("randint needs low < high");
      return tensor_ret(torch::randint(a[0].i, a[1].i, dims_of(a, 2, n, "randint"), torch::kInt64));
    });
  }
  method(r, "Torch", "randperm", {"int"}, "Tensor", [tensor_ret](void*, Args& a) {
    if (a[0].i < 0 || a[0].i > kMaxElements) fail("randperm size must be between 0 and 2^31");
    return tensor_ret(torch::randperm(a[0].i, torch::kInt64));
  });
  method(r, "Torch", "arange", {"int"}, "Tensor", [tensor_ret](void*, Args& a) {
    if (a[0].i < 0 || a[0].i > kMaxElements) fail("arange end must be between 0 and 2^31");
    return tensor_ret(torch::arange(a[0].i, torch::kInt64));
  });
  method(r, "Torch", "arange", {"float", "float", "float"}, "Tensor", [tensor_ret](void*, Args& a) {
    for (int i = 0; i < 3; ++i) finite(a[i].f, "arange argument");
    if (a[2].f == 0) fail("arange step must not be zero");
    if ((a[1].f - a[0].f) / a[2].f > double(kMaxElements)) fail("arange exceeds 2^31 elements");
    return tensor_ret(torch::arange(a[0].f, a[1].f, a[2].f, torch::kFloat32));
  });
  method(r, "Torch", "linspace", {"float", "float", "int"}, "Tensor", [tensor_ret](void*, Args& a) {
    finite(a[0].f, "linspace start"); finite(a[1].f, "linspace end");
    if (a[2].i < 0 || a[2].i > kMaxElements) fail("linspace steps must be between 0 and 2^31");
    return tensor_ret(torch::linspace(a[0].f, a[1].f, a[2].i, torch::kFloat32));
  });
  method(r, "Torch", "eye", {"int"}, "Tensor", [tensor_ret](void*, Args& a) {
    if (a[0].i < 0 || a[0].i > 46340) fail("eye size must be between 0 and 46340");
    return tensor_ret(torch::eye(a[0].i));
  });
  method(r, "Torch", "scalar", {"float"}, "Tensor", [tensor_ret](void*, Args& a) { return tensor_ret(torch::scalar_tensor(a[0].f, torch::kFloat32)); });
  method(r, "Torch", "tensor", {"string"}, "Tensor", [tensor_ret](void*, Args& a) { return tensor_ret(parse_tensor(a[0].s, nullptr)); });
  method(r, "Torch", "tensor", {"string", "DType"}, "Tensor", [tensor_ret](void*, Args& a) { return tensor_ret(parse_tensor(a[0].s, &a[1].s)); });
  method(r, "Torch", "where", {"Tensor", "Tensor", "Tensor"}, "Tensor", [tensor_ret](void*, Args& a) {
    if (T(a[0]).scalar_type() != torch::kBool) fail("where condition must be a BOOL tensor (use a comparison such as x.gt(0.0))");
    return tensor_ret(torch::where(T(a[0]), T(a[1]), T(a[2])));
  });
  method(r, "Torch", "list", {}, "TensorList", [&h](void*, Args&) { return h.hold(h.lists, "TensorList", std::vector<torch::Tensor>{}); });
  auto joined = [](void* list, const char* who) {
    auto& items = *self<std::vector<torch::Tensor>>(list);
    if (items.empty()) fail(std::string(who) + " needs a non-empty list");
    return items;
  };
  method(r, "Torch", "cat", {"TensorList", "int"}, "Tensor", [tensor_ret, joined](void*, Args& a) { return tensor_ret(torch::cat(joined(a[0].obj, "cat"), a[1].i)); });
  method(r, "Torch", "stack", {"TensorList", "int"}, "Tensor", [tensor_ret, joined](void*, Args& a) { return tensor_ret(torch::stack(joined(a[0].obj, "stack"), a[1].i)); });

  // ---- Torch: files
  method(r, "Torch", "load", {"string"}, "Tensor", [tensor_ret](void*, Args& a) {
    auto b = load_bundle(a[0].s);
    auto it = b.tensors.find("tensor");
    if (it == b.tensors.end()) fail(a[0].s + " has no tensor named 'tensor' (use torch.load(path, key) or torch.open(path).keys())");
    return tensor_ret(from_data(it->second));
  });
  method(r, "Torch", "load", {"string", "string"}, "Tensor", [tensor_ret](void*, Args& a) {
    auto b = load_bundle(a[0].s);
    auto it = b.tensors.find(a[1].s);
    if (it == b.tensors.end()) fail(a[0].s + " has no tensor named '" + a[1].s + "'");
    return tensor_ret(from_data(it->second));
  });
  method(r, "Torch", "save", {"string", "Tensor"}, "void", [](void*, Args& a) { save_bundle(a[0].s, {{"tensor", T(a[1])}}, {}); return Value::none(); });
  method(r, "Torch", "bundle", {}, "Bundle", [&h](void*, Args&) { return h.hold(h.bundles, "Bundle", BundleBox{}); });
  method(r, "Torch", "open", {"string"}, "Bundle", [&h](void*, Args& a) {
    auto b = load_bundle(a[0].s);
    BundleBox box;
    for (const auto& [name, data] : b.tensors) box.tensors.emplace(name, from_data(data));
    box.meta = b.metadata;
    return h.hold(h.bundles, "Bundle", std::move(box));
  });
  method(r, "Torch", "read_csv", {"string", "bool"}, "Tensor", [tensor_ret](void*, Args& a) { return tensor_ret(parse_csv(a[0].s, a[1].b)); });

  // ---- Torch: models and optimizers
  method(r, "Torch", "sequential", {}, "Model", [&h](void*, Args&) { return h.hold(h.models, "Model", ModelBox{torch::nn::Sequential(), {}}); });
  method(r, "Torch", "load_model", {"string"}, "Model", [&h](void*, Args& a) {
    auto b = load_bundle(a[0].s);
    auto fmt = b.metadata.find("torchcli_format"), spec = b.metadata.find("layers");
    if (fmt == b.metadata.end() || fmt->second != "model-1" || spec == b.metadata.end()) fail(a[0].s + " is not a model saved by Model.save (metadata torchcli_format=model-1 and layers are required)");
    ModelBox m{torch::nn::Sequential(), {}};
    replay_layers(m, spec->second, a[0].s);
    load_weights(m, b, a[0].s);
    return h.hold(h.models, "Model", std::move(m));
  });
  auto lr_ok = [](double lr) { if (!std::isfinite(lr) || lr <= 0) fail("learning rate must be positive and finite"); return lr; };
  auto model_params = [](void* m) {
    auto params = self<ModelBox>(m)->seq->parameters();
    if (params.empty()) fail("model has no parameters (add a linear, conv2d or layer_norm layer first)");
    return params;
  };
  auto list_params = [](void* l) {
    auto params = *self<std::vector<torch::Tensor>>(l);
    if (params.empty()) fail("optimizer needs at least one parameter");
    for (auto& p : params)
      if (!p.is_leaf() || !p.requires_grad()) fail("optimizer parameters must be leaf tensors with requires_grad(true)");
    return params;
  };
  auto make_opt = [&h](std::unique_ptr<torch::optim::Optimizer> o, std::string kind) { return h.hold(h.optimizers, "Optimizer", OptBox{std::move(o), std::move(kind)}); };
  method(r, "Torch", "sgd", {"Model", "float"}, "Optimizer", [=](void*, Args& a) { return make_opt(std::make_unique<torch::optim::SGD>(model_params(a[0].obj), torch::optim::SGDOptions(lr_ok(a[1].f))), "sgd"); });
  method(r, "Torch", "sgd", {"Model", "float", "float"}, "Optimizer", [=](void*, Args& a) {
    if (!std::isfinite(a[2].f) || a[2].f < 0 || a[2].f >= 1) fail("momentum must be in [0, 1)");
    return make_opt(std::make_unique<torch::optim::SGD>(model_params(a[0].obj), torch::optim::SGDOptions(lr_ok(a[1].f)).momentum(a[2].f)), "sgd");
  });
  method(r, "Torch", "sgd", {"TensorList", "float"}, "Optimizer", [=](void*, Args& a) { return make_opt(std::make_unique<torch::optim::SGD>(list_params(a[0].obj), torch::optim::SGDOptions(lr_ok(a[1].f))), "sgd"); });
  method(r, "Torch", "adam", {"Model", "float"}, "Optimizer", [=](void*, Args& a) { return make_opt(std::make_unique<torch::optim::Adam>(model_params(a[0].obj), torch::optim::AdamOptions(lr_ok(a[1].f))), "adam"); });
  method(r, "Torch", "adam", {"TensorList", "float"}, "Optimizer", [=](void*, Args& a) { return make_opt(std::make_unique<torch::optim::Adam>(list_params(a[0].obj), torch::optim::AdamOptions(lr_ok(a[1].f))), "adam"); });
  method(r, "Torch", "adamw", {"Model", "float", "float"}, "Optimizer", [=](void*, Args& a) {
    if (!std::isfinite(a[2].f) || a[2].f < 0) fail("weight decay must be nonnegative and finite");
    return make_opt(std::make_unique<torch::optim::AdamW>(model_params(a[0].obj), torch::optim::AdamWOptions(lr_ok(a[1].f)).weight_decay(a[2].f)), "adamw");
  });

  // ---- Torch: handle lifetime
  method(r, "Torch", "mark", {}, "int", [&h](void*, Args&) { return Value::integer(int64_t(h.tensors.size())); });
  method(r, "Torch", "release", {"int"}, "int", [&h](void*, Args& a) {
    if (a[0].i < 0 || a[0].i > int64_t(h.tensors.size())) fail("release mark " + std::to_string(a[0].i) + " is not a value returned by torch.mark()");
    int64_t n = 0;
    for (size_t i = size_t(a[0].i); i < h.tensors.size(); ++i)
      if (!h.tensors[i].kept && h.tensors[i].t.defined()) { h.tensors[i].t = torch::Tensor(); ++n; }
    return Value::integer(n);
  });
  method(r, "Torch", "live_tensors", {}, "int", [&h](void*, Args&) {
    int64_t n = 0;
    for (const auto& b : h.tensors) n += b.t.defined();
    return Value::integer(n);
  });

  // ---- Tensor: information
  method(r, "Tensor", "numel", {}, "int", [](void* p, Args&) { return Value::integer(tensor_of(p).numel()); });
  method(r, "Tensor", "dim", {}, "int", [](void* p, Args&) { return Value::integer(tensor_of(p).dim()); });
  method(r, "Tensor", "size", {"int"}, "int", [](void* p, Args& a) {
    auto& t = tensor_of(p);
    if (t.dim() == 0) fail("size() of a zero-dimensional tensor");
    return Value::integer(t.size(check_dim(t, a[0].i)));
  });
  method(r, "Tensor", "shape", {}, "string", [](void* p, Args&) { return Value::string(shape_text(tensor_of(p))); });
  method(r, "Tensor", "dtype", {}, "DType", [](void* p, Args&) { return Value::string(dtype_name(tensor_of(p).scalar_type())); });
  method(r, "Tensor", "item", {}, "float", [](void* p, Args&) { return Value::real(one_element(tensor_of(p), "item()").item<double>()); });
  method(r, "Tensor", "item_int", {}, "int", [](void* p, Args&) {
    auto t = one_element(tensor_of(p), "item_int()");
    if (t.is_floating_point()) fail("item_int() needs an I64 or BOOL tensor; use item() for floats or to(DType.I64)");
    return Value::integer(t.item<int64_t>());
  });
  method(r, "Tensor", "value", {"int"}, "float", [](void* p, Args& a) { auto& t = tensor_of(p); return Value::real(t.detach().reshape(-1)[flat_index(t, a[0].i)].item<double>()); });
  method(r, "Tensor", "value_int", {"int"}, "int", [](void* p, Args& a) {
    auto& t = tensor_of(p);
    if (t.is_floating_point()) fail("value_int() needs an I64 or BOOL tensor; use value() for floats");
    return Value::integer(t.detach().reshape(-1)[flat_index(t, a[0].i)].item<int64_t>());
  });
  method(r, "Tensor", "set", {"int", "float"}, "Tensor", [](void* p, Args& a) {
    auto& t = tensor_of(p);
    if (!t.is_contiguous()) fail("set() needs a contiguous tensor; clone() a view first");
    torch::NoGradGuard g;
    t.view(-1)[flat_index(t, a[0].i)].fill_(a[1].f);
    return Value::object("Tensor", p);
  });
  method(r, "Tensor", "to_text", {}, "string", [](void* p, Args&) { return Value::string(tensor_text(tensor_of(p), false)); });
  method(r, "Tensor", "to_json", {}, "string", [](void* p, Args&) { return Value::string(tensor_text(tensor_of(p), true)); });
  method(r, "Tensor", "to", {"DType"}, "Tensor", [tensor_ret](void* p, Args& a) { return tensor_ret(tensor_of(p).to(scalar_type(a[0].s))); });

  // ---- Tensor: arithmetic (operators + - * neg and their reversed float forms map here)
  auto binary = [&](const char* name, torch::Tensor (*tt)(const torch::Tensor&, const torch::Tensor&), torch::Tensor (*ts)(const torch::Tensor&, double)) {
    method(r, "Tensor", name, {"Tensor"}, "Tensor", [tensor_ret, tt](void* p, Args& a) { return tensor_ret(tt(tensor_of(p), T(a[0]))); });
    if (ts) method(r, "Tensor", name, {"float"}, "Tensor", [tensor_ret, ts](void* p, Args& a) { return tensor_ret(ts(tensor_of(p), a[0].f)); });
  };
  binary("add", [](const torch::Tensor& x, const torch::Tensor& y) { return x + y; }, [](const torch::Tensor& x, double y) { return x + y; });
  binary("sub", [](const torch::Tensor& x, const torch::Tensor& y) { return x - y; }, [](const torch::Tensor& x, double y) { return x - y; });
  binary("mul", [](const torch::Tensor& x, const torch::Tensor& y) { return x * y; }, [](const torch::Tensor& x, double y) { return x * y; });
  binary("div", [](const torch::Tensor& x, const torch::Tensor& y) { return x / y; }, [](const torch::Tensor& x, double y) { return x / y; });
  binary("matmul", [](const torch::Tensor& x, const torch::Tensor& y) { return x.matmul(y); }, nullptr);
  binary("maximum", [](const torch::Tensor& x, const torch::Tensor& y) { return torch::maximum(x, y); }, nullptr);
  binary("minimum", [](const torch::Tensor& x, const torch::Tensor& y) { return torch::minimum(x, y); }, nullptr);
  binary("gt", [](const torch::Tensor& x, const torch::Tensor& y) { return x.gt(y); }, [](const torch::Tensor& x, double y) { return x.gt(y); });
  binary("lt", [](const torch::Tensor& x, const torch::Tensor& y) { return x.lt(y); }, [](const torch::Tensor& x, double y) { return x.lt(y); });
  binary("eq", [](const torch::Tensor& x, const torch::Tensor& y) { return x.eq(y); }, [](const torch::Tensor& x, double y) { return x.eq(y); });
  method(r, "Tensor", "ge", {"float"}, "Tensor", [tensor_ret](void* p, Args& a) { return tensor_ret(tensor_of(p).ge(a[0].f)); });
  method(r, "Tensor", "le", {"float"}, "Tensor", [tensor_ret](void* p, Args& a) { return tensor_ret(tensor_of(p).le(a[0].f)); });
  method(r, "Tensor", "rsub", {"float"}, "Tensor", [tensor_ret](void* p, Args& a) { return tensor_ret(torch::rsub(tensor_of(p), a[0].f)); });
  method(r, "Tensor", "pow", {"float"}, "Tensor", [tensor_ret](void* p, Args& a) { return tensor_ret(tensor_of(p).pow(a[0].f)); });
  method(r, "Tensor", "clamp", {"float", "float"}, "Tensor", [tensor_ret](void* p, Args& a) {
    if (!(a[0].f <= a[1].f)) fail("clamp needs low <= high");
    return tensor_ret(tensor_of(p).clamp(a[0].f, a[1].f));
  });
  auto unary = [&](const char* name, torch::Tensor (*fn)(const torch::Tensor&)) {
    method(r, "Tensor", name, {}, "Tensor", [tensor_ret, fn](void* p, Args&) { return tensor_ret(fn(tensor_of(p))); });
  };
  unary("neg", [](const torch::Tensor& x) { return -x; });
  unary("abs", [](const torch::Tensor& x) { return x.abs(); });
  unary("exp", [](const torch::Tensor& x) { return x.exp(); });
  unary("log", [](const torch::Tensor& x) { return x.log(); });
  unary("sqrt", [](const torch::Tensor& x) { return x.sqrt(); });
  unary("square", [](const torch::Tensor& x) { return x.square(); });
  unary("sin", [](const torch::Tensor& x) { return x.sin(); });
  unary("cos", [](const torch::Tensor& x) { return x.cos(); });
  unary("sign", [](const torch::Tensor& x) { return x.sign(); });
  unary("floor", [](const torch::Tensor& x) { return x.floor(); });
  unary("round", [](const torch::Tensor& x) { return x.round(); });
  unary("relu", [](const torch::Tensor& x) { return x.relu(); });
  unary("sigmoid", [](const torch::Tensor& x) { return x.sigmoid(); });
  unary("tanh", [](const torch::Tensor& x) { return x.tanh(); });
  unary("gelu", [](const torch::Tensor& x) { return torch::gelu(x); });
  method(r, "Tensor", "leaky_relu", {"float"}, "Tensor", [tensor_ret](void* p, Args& a) { return tensor_ret(torch::leaky_relu(tensor_of(p), finite(a[0].f, "negative slope"))); });
  method(r, "Tensor", "softmax", {"int"}, "Tensor", [tensor_ret](void* p, Args& a) { auto& t = tensor_of(p); return tensor_ret(t.softmax(check_dim(t, a[0].i))); });
  method(r, "Tensor", "log_softmax", {"int"}, "Tensor", [tensor_ret](void* p, Args& a) { auto& t = tensor_of(p); return tensor_ret(t.log_softmax(check_dim(t, a[0].i))); });

  // ---- Tensor: reductions
  unary("sum", [](const torch::Tensor& x) { return x.sum(); });
  unary("mean", [](const torch::Tensor& x) { return x.mean(); });
  unary("std", [](const torch::Tensor& x) { return x.std(); });
  unary("var", [](const torch::Tensor& x) { return x.var(); });
  unary("norm", [](const torch::Tensor& x) { return x.norm(); });
  unary("max", [](const torch::Tensor& x) { return x.max(); });
  unary("min", [](const torch::Tensor& x) { return x.min(); });
  unary("argmax", [](const torch::Tensor& x) { return x.argmax(); });
  method(r, "Tensor", "sum", {"int"}, "Tensor", [tensor_ret](void* p, Args& a) { auto& t = tensor_of(p); return tensor_ret(t.sum(at::IntArrayRef{check_dim(t, a[0].i)})); });
  method(r, "Tensor", "sum", {"int", "bool"}, "Tensor", [tensor_ret](void* p, Args& a) { auto& t = tensor_of(p); return tensor_ret(t.sum(at::IntArrayRef{check_dim(t, a[0].i)}, a[1].b)); });
  method(r, "Tensor", "mean", {"int"}, "Tensor", [tensor_ret](void* p, Args& a) { auto& t = tensor_of(p); return tensor_ret(t.mean(at::IntArrayRef{check_dim(t, a[0].i)})); });
  method(r, "Tensor", "mean", {"int", "bool"}, "Tensor", [tensor_ret](void* p, Args& a) { auto& t = tensor_of(p); return tensor_ret(t.mean(at::IntArrayRef{check_dim(t, a[0].i)}, a[1].b)); });
  method(r, "Tensor", "std", {"int", "bool"}, "Tensor", [tensor_ret](void* p, Args& a) { auto& t = tensor_of(p); return tensor_ret(torch::std(t, at::IntArrayRef{check_dim(t, a[0].i)}, true, a[1].b)); });
  method(r, "Tensor", "amax", {"int"}, "Tensor", [tensor_ret](void* p, Args& a) { auto& t = tensor_of(p); return tensor_ret(t.amax(at::IntArrayRef{check_dim(t, a[0].i)})); });
  method(r, "Tensor", "amin", {"int"}, "Tensor", [tensor_ret](void* p, Args& a) { auto& t = tensor_of(p); return tensor_ret(t.amin(at::IntArrayRef{check_dim(t, a[0].i)})); });
  method(r, "Tensor", "argmax", {"int"}, "Tensor", [tensor_ret](void* p, Args& a) { auto& t = tensor_of(p); return tensor_ret(t.argmax(check_dim(t, a[0].i))); });
  method(r, "Tensor", "argmin", {"int"}, "Tensor", [tensor_ret](void* p, Args& a) { auto& t = tensor_of(p); return tensor_ret(t.argmin(check_dim(t, a[0].i))); });

  // ---- Tensor: shape
  for (size_t n = 1; n <= 4; ++n)
    method(r, "Tensor", "reshape", dims[n - 1], "Tensor", [n, tensor_ret](void* p, Args& a) {
      std::vector<int64_t> d;
      for (size_t i = 0; i < n; ++i) { if (a[i].i < -1) fail("reshape dimensions must be nonnegative or -1"); d.push_back(a[i].i); }
      return tensor_ret(tensor_of(p).reshape(d));
    });
  unary("flatten", [](const torch::Tensor& x) { return x.flatten(); });
  unary("t", [](const torch::Tensor& x) { return x.t(); });
  unary("squeeze", [](const torch::Tensor& x) { return x.squeeze(); });
  method(r, "Tensor", "transpose", {"int", "int"}, "Tensor", [tensor_ret](void* p, Args& a) { auto& t = tensor_of(p); return tensor_ret(t.transpose(check_dim(t, a[0].i), check_dim(t, a[1].i))); });
  method(r, "Tensor", "unsqueeze", {"int"}, "Tensor", [tensor_ret](void* p, Args& a) {
    auto& t = tensor_of(p);
    if (a[0].i < -t.dim() - 1 || a[0].i > t.dim()) fail("unsqueeze dimension " + std::to_string(a[0].i) + " is out of range for shape " + shape_text(t));
    return tensor_ret(t.unsqueeze(a[0].i));
  });
  method(r, "Tensor", "squeeze", {"int"}, "Tensor", [tensor_ret](void* p, Args& a) { auto& t = tensor_of(p); return tensor_ret(t.squeeze(check_dim(t, a[0].i))); });
  method(r, "Tensor", "slice", {"int", "int", "int"}, "Tensor", [tensor_ret](void* p, Args& a) {
    auto& t = tensor_of(p);
    if (t.dim() == 0) fail("slice() of a zero-dimensional tensor");
    return tensor_ret(t.slice(check_dim(t, a[0].i), a[1].i, a[2].i));
  });
  method(r, "Tensor", "select", {"int", "int"}, "Tensor", [tensor_ret](void* p, Args& a) {
    auto& t = tensor_of(p);
    if (t.dim() == 0) fail("select() of a zero-dimensional tensor");
    int64_t d = check_dim(t, a[0].i), n = t.size(d);
    if (a[1].i < -n || a[1].i >= n) fail("select index " + std::to_string(a[1].i) + " is out of range for dimension of size " + std::to_string(n));
    return tensor_ret(t.select(d, a[1].i));
  });
  method(r, "Tensor", "index_select", {"int", "Tensor"}, "Tensor", [tensor_ret](void* p, Args& a) {
    auto& t = tensor_of(p);
    if (T(a[1]).scalar_type() != torch::kInt64 || T(a[1]).dim() != 1) fail("index_select indices must be a 1-D I64 tensor");
    return tensor_ret(t.index_select(check_dim(t, a[0].i), T(a[1])));
  });

  // ---- Tensor: neural-network functions and losses
  method(r, "Tensor", "linear", {"Tensor"}, "Tensor", [tensor_ret](void* p, Args& a) { return tensor_ret(torch::linear(tensor_of(p), T(a[0]))); });
  method(r, "Tensor", "linear", {"Tensor", "Tensor"}, "Tensor", [tensor_ret](void* p, Args& a) { return tensor_ret(torch::linear(tensor_of(p), T(a[0]), T(a[1]))); });
  auto loss = [&](const char* name, torch::Tensor (*fn)(const torch::Tensor&, const torch::Tensor&), bool shapes) {
    method(r, "Tensor", name, {"Tensor"}, "Tensor", [tensor_ret, fn, shapes, name](void* p, Args& a) {
      if (shapes) same_shape(tensor_of(p), T(a[0]), name);
      return tensor_ret(fn(tensor_of(p), T(a[0])));
    });
  };
  loss("mse_loss", [](const torch::Tensor& x, const torch::Tensor& y) { return torch::mse_loss(x, y); }, true);
  loss("l1_loss", [](const torch::Tensor& x, const torch::Tensor& y) { return torch::l1_loss(x, y); }, true);
  loss("smooth_l1_loss", [](const torch::Tensor& x, const torch::Tensor& y) { return torch::smooth_l1_loss(x, y); }, true);
  loss("bce_with_logits", [](const torch::Tensor& x, const torch::Tensor& y) { return torch::binary_cross_entropy_with_logits(x, y); }, true);
  loss("cross_entropy", [](const torch::Tensor& x, const torch::Tensor& y) { return torch::nn::functional::cross_entropy(x, y); }, false);
  loss("nll_loss", [](const torch::Tensor& x, const torch::Tensor& y) { return torch::nll_loss(x, y); }, false);

  // ---- Tensor: predicates
  method(r, "Tensor", "allclose", {"Tensor", "float", "float"}, "bool", [](void* p, Args& a) {
    if (!(a[1].f >= 0) || !(a[2].f >= 0)) fail("allclose tolerances must be nonnegative");
    if (tensor_of(p).sizes() != T(a[0]).sizes()) return Value::boolean(false);
    return Value::boolean(torch::allclose(tensor_of(p).detach().to(torch::kFloat64), T(a[0]).detach().to(torch::kFloat64), a[1].f, a[2].f));
  });
  method(r, "Tensor", "equal", {"Tensor"}, "bool", [](void* p, Args& a) { return Value::boolean(tensor_of(p).scalar_type() == T(a[0]).scalar_type() && torch::equal(tensor_of(p), T(a[0]))); });
  method(r, "Tensor", "is_finite", {}, "bool", [](void* p, Args&) { return Value::boolean(torch::isfinite(tensor_of(p).detach()).all().item<bool>()); });
  method(r, "Tensor", "any", {}, "bool", [](void* p, Args&) { return Value::boolean(tensor_of(p).detach().any().item<bool>()); });
  method(r, "Tensor", "all", {}, "bool", [](void* p, Args&) { return Value::boolean(tensor_of(p).detach().all().item<bool>()); });

  // ---- Tensor: autograd and lifetime
  method(r, "Tensor", "requires_grad", {}, "bool", [](void* p, Args&) { return Value::boolean(tensor_of(p).requires_grad()); });
  method(r, "Tensor", "requires_grad", {"bool"}, "Tensor", [](void* p, Args& a) {
    auto& t = tensor_of(p);
    if (a[0].b && !t.is_floating_point()) fail("only F32 and F64 tensors can require gradients");
    if (!t.is_leaf()) fail("requires_grad(bool) applies to leaf tensors; detach() a computed tensor first");
    t.requires_grad_(a[0].b);
    return Value::object("Tensor", p);
  });
  method(r, "Tensor", "backward", {}, "void", [](void* p, Args&) {
    auto& t = tensor_of(p);
    if (!t.requires_grad()) fail("backward() on a tensor that does not depend on any requires_grad(true) leaf");
    if (t.numel() != 1) fail("backward() needs a one-element (scalar) result; found shape " + shape_text(t) + " (reduce it with sum() or mean())");
    t.backward();
    return Value::none();
  });
  method(r, "Tensor", "grad", {}, "Tensor", [tensor_ret](void* p, Args&) {
    auto& t = tensor_of(p);
    if (!t.grad().defined()) fail("tensor has no gradient: mark the leaf with requires_grad(true) and call backward() on a result that depends on it");
    return tensor_ret(t.grad().clone());
  });
  method(r, "Tensor", "has_grad", {}, "bool", [](void* p, Args&) { return Value::boolean(tensor_of(p).grad().defined()); });
  method(r, "Tensor", "zero_grad", {}, "Tensor", [](void* p, Args&) { tensor_of(p).mutable_grad().reset(); return Value::object("Tensor", p); });
  unary("detach", [](const torch::Tensor& x) { return x.detach(); });
  unary("clone", [](const torch::Tensor& x) { return x.clone(); });
  method(r, "Tensor", "keep", {}, "Tensor", [](void* p, Args&) { tensor_of(p); self<TensorBox>(p)->kept = true; return Value::object("Tensor", p); });
  method(r, "Tensor", "save", {"string"}, "Tensor", [](void* p, Args& a) { save_bundle(a[0].s, {{"tensor", tensor_of(p)}}, {}); return Value::object("Tensor", p); });
  method(r, "Tensor", "write_csv", {"string"}, "Tensor", [](void* p, Args& a) { write_text_atomic(a[0].s, csv_text(tensor_of(p))); return Value::object("Tensor", p); });

  // ---- TensorList (len/at make it a foreach collection); StrList is the lmcode std type
  method(r, "TensorList", "add", {"Tensor"}, "TensorList", [](void* p, Args& a) { self<std::vector<torch::Tensor>>(p)->push_back(T(a[0])); return Value::object("TensorList", p); });
  method(r, "TensorList", "len", {}, "int", [](void* p, Args&) { return Value::integer(int64_t(self<std::vector<torch::Tensor>>(p)->size())); });
  method(r, "TensorList", "at", {"int"}, "Tensor", [tensor_ret](void* p, Args& a) {
    auto& v = *self<std::vector<torch::Tensor>>(p);
    if (a[0].i < 0 || a[0].i >= int64_t(v.size())) fail("list index " + std::to_string(a[0].i) + " is out of range for " + std::to_string(v.size()) + " tensors");
    return tensor_ret(v[size_t(a[0].i)]);
  });

  // ---- Bundle: named tensors plus string metadata in one safetensors file
  method(r, "Bundle", "put", {"string", "Tensor"}, "Bundle", [](void* p, Args& a) {
    if (a[0].s.empty() || a[0].s == "__metadata__") fail("invalid tensor name '" + a[0].s + "'");
    self<BundleBox>(p)->tensors[a[0].s] = T(a[1]);
    return Value::object("Bundle", p);
  });
  method(r, "Bundle", "get", {"string"}, "Tensor", [tensor_ret](void* p, Args& a) {
    auto& b = *self<BundleBox>(p);
    auto it = b.tensors.find(a[0].s);
    if (it == b.tensors.end()) fail("bundle has no tensor named '" + a[0].s + "'");
    return tensor_ret(it->second);
  });
  method(r, "Bundle", "has", {"string"}, "bool", [](void* p, Args& a) { return Value::boolean(self<BundleBox>(p)->tensors.count(a[0].s) > 0); });
  method(r, "Bundle", "keys", {}, "StrList", [&r](void* p, Args&) {
    std::vector<std::string> keys;
    for (const auto& kv : self<BundleBox>(p)->tensors) keys.push_back(kv.first);
    return Value::object("StrList", r.std_collections().make_str_list(std::move(keys)));
  });
  method(r, "Bundle", "meta", {"string"}, "string", [](void* p, Args& a) {
    auto& m = self<BundleBox>(p)->meta;
    auto it = m.find(a[0].s);
    if (it == m.end()) fail("bundle has no metadata key '" + a[0].s + "'");
    return Value::string(it->second);
  });
  method(r, "Bundle", "meta", {"string", "string"}, "Bundle", [](void* p, Args& a) {
    if (a[0].s.empty()) fail("metadata key must not be empty");
    self<BundleBox>(p)->meta[a[0].s] = a[1].s;
    return Value::object("Bundle", p);
  });
  method(r, "Bundle", "has_meta", {"string"}, "bool", [](void* p, Args& a) { return Value::boolean(self<BundleBox>(p)->meta.count(a[0].s) > 0); });
  method(r, "Bundle", "save", {"string"}, "void", [](void* p, Args& a) { auto& b = *self<BundleBox>(p); save_bundle(a[0].s, b.tensors, b.meta); return Value::none(); });

  // ---- Model: a torch::nn::Sequential built layer by layer; parameter names match PyTorch's nn.Sequential ("0.weight")
  auto layer = [&](const char* name, std::vector<std::string> params) {
    std::vector<std::string> types = params;
    method(r, "Model", name, std::move(params), "Model", [name, types](void* p, Args& a) {
      std::vector<int64_t> ints;
      double f = std::nan("");
      for (size_t i = 0; i < types.size(); ++i) { if (types[i] == "int") ints.push_back(a[i].i); else f = a[i].f; }
      auto& m = *self<ModelBox>(p);
      m.layers.push_back(add_layer(m, name, ints, f));
      return Value::object("Model", p);
    });
  };
  layer("linear", {"int", "int"});
  layer("conv2d", {"int", "int", "int"});
  layer("conv2d", {"int", "int", "int", "int", "int"});
  layer("max_pool2d", {"int"});
  layer("layer_norm", {"int"});
  layer("softmax", {"int"});
  layer("log_softmax", {"int"});
  layer("leaky_relu", {"float"});
  layer("dropout", {"float"});
  for (const char* name : {"flatten", "relu", "tanh", "sigmoid", "gelu"}) layer(name, {});
  method(r, "Model", "forward", {"Tensor"}, "Tensor", [tensor_ret](void* p, Args& a) {
    auto& m = *self<ModelBox>(p);
    if (m.seq->is_empty()) fail("model has no layers");
    return tensor_ret(m.seq->forward(T(a[0])));
  });
  method(r, "Model", "parameters", {}, "TensorList", [&h](void* p, Args&) { return h.hold(h.lists, "TensorList", self<ModelBox>(p)->seq->parameters()); });
  method(r, "Model", "param", {"string"}, "Tensor", [tensor_ret](void* p, Args& a) {
    auto named = self<ModelBox>(p)->seq->named_parameters();
    if (auto* t = named.find(a[0].s)) return tensor_ret(*t);
    std::string names;
    for (const auto& kv : named) names += (names.empty() ? "" : ", ") + kv.key();
    fail("model has no parameter '" + a[0].s + "' (parameters: " + (names.empty() ? "none" : names) + ")");
  });
  method(r, "Model", "num_parameters", {}, "int", [](void* p, Args&) {
    int64_t n = 0;
    for (const auto& t : self<ModelBox>(p)->seq->parameters()) n += t.numel();
    return Value::integer(n);
  });
  method(r, "Model", "layers", {}, "int", [](void* p, Args&) { return Value::integer(int64_t(self<ModelBox>(p)->seq->size())); });
  method(r, "Model", "summary", {}, "string", [](void* p, Args&) {
    std::string s;
    for (const auto& l : self<ModelBox>(p)->layers) s += (s.empty() ? "" : " -> ") + l;
    return Value::string(s.empty() ? "(empty)" : s);
  });
  method(r, "Model", "train", {}, "Model", [](void* p, Args&) { self<ModelBox>(p)->seq->train(true); return Value::object("Model", p); });
  method(r, "Model", "eval", {}, "Model", [](void* p, Args&) { self<ModelBox>(p)->seq->eval(); return Value::object("Model", p); });
  method(r, "Model", "zero_grad", {}, "Model", [](void* p, Args&) { self<ModelBox>(p)->seq->zero_grad(); return Value::object("Model", p); });
  method(r, "Model", "save", {"string"}, "void", [](void* p, Args& a) {
    auto& m = *self<ModelBox>(p);
    std::map<std::string, torch::Tensor> tensors;
    for (const auto& kv : m.seq->named_parameters()) tensors[kv.key()] = kv.value();
    for (const auto& kv : m.seq->named_buffers()) tensors[kv.key()] = kv.value();
    if (tensors.empty()) fail("model has no parameters to save (add a linear, conv2d or layer_norm layer)");
    std::string summary;
    for (const auto& l : m.layers) summary += (summary.empty() ? "" : " -> ") + l;
    save_bundle(a[0].s, tensors, {{"torchcli_format", "model-1"}, {"layers", summary}});
    return Value::none();
  });
  method(r, "Model", "load", {"string"}, "Model", [](void* p, Args& a) { load_weights(*self<ModelBox>(p), load_bundle(a[0].s), a[0].s); return Value::object("Model", p); });

  // ---- Optimizer
  method(r, "Optimizer", "zero_grad", {}, "void", [](void* p, Args&) { self<OptBox>(p)->opt->zero_grad(); return Value::none(); });
  method(r, "Optimizer", "step", {}, "void", [](void* p, Args&) { self<OptBox>(p)->opt->step(); return Value::none(); });
  method(r, "Optimizer", "lr", {}, "float", [](void* p, Args&) { return Value::real(self<OptBox>(p)->opt->param_groups().at(0).options().get_lr()); });
  method(r, "Optimizer", "set_lr", {"float"}, "Optimizer", [lr_ok](void* p, Args& a) {
    double lr = lr_ok(a[0].f);
    for (auto& g : self<OptBox>(p)->opt->param_groups()) g.options().set_lr(lr);
    return Value::object("Optimizer", p);
  });

  // ---- Out: the JSON result object, logs and application failures
  auto key_ok = [](const std::string& k) { if (k.empty()) fail("result key must not be empty"); return k; };
  method(r, "Out", "set", {"string", "float"}, "Out", [key_ok](void* p, Args& a) { self<Out>(p)->result[key_ok(a[0].s)] = std::isfinite(a[1].f) ? OJson(a[1].f) : OJson(nullptr); return Value::object("Out", p); });
  method(r, "Out", "set", {"string", "int"}, "Out", [key_ok](void* p, Args& a) { self<Out>(p)->result[key_ok(a[0].s)] = a[1].i; return Value::object("Out", p); });
  method(r, "Out", "set", {"string", "string"}, "Out", [key_ok](void* p, Args& a) { self<Out>(p)->result[key_ok(a[0].s)] = a[1].s; return Value::object("Out", p); });
  method(r, "Out", "set", {"string", "bool"}, "Out", [key_ok](void* p, Args& a) { self<Out>(p)->result[key_ok(a[0].s)] = a[1].b; return Value::object("Out", p); });
  method(r, "Out", "set", {"string", "Tensor"}, "Out", [key_ok](void* p, Args& a) {
    if (T(a[1]).numel() > 1000000) fail("out.set: tensor has more than 1000000 elements; save it to a file instead");
    self<Out>(p)->result[key_ok(a[0].s)] = OJson::parse(tensor_text(T(a[1]), true));
    return Value::object("Out", p);
  });
  auto append = [key_ok](void* p, const std::string& key, OJson v) {
    auto& slot = self<Out>(p)->result[key_ok(key)];
    if (slot.is_null()) slot = OJson::array();
    if (!slot.is_array()) fail("out.append: result '" + key + "' is not a list (it was set with out.set)");
    slot.push_back(std::move(v));
    return Value::object("Out", p);
  };
  method(r, "Out", "append", {"string", "float"}, "Out", [append](void* p, Args& a) { return append(p, a[0].s, std::isfinite(a[1].f) ? OJson(a[1].f) : OJson(nullptr)); });
  method(r, "Out", "append", {"string", "int"}, "Out", [append](void* p, Args& a) { return append(p, a[0].s, OJson(a[1].i)); });
  method(r, "Out", "log", {"string"}, "void", [](void*, Args& a) { std::cerr << a[0].s << "\n"; return Value::none(); });
  method(r, "Out", "check", {"bool", "string"}, "void", [](void*, Args& a) { if (!a[0].b) throw dslvm::Error("APP_FAIL", "check failed: " + a[1].s); return Value::none(); });
  method(r, "Out", "fail", {"string"}, "void", [](void*, Args& a) -> Value { throw dslvm::Error("APP_FAIL", a[0].s); });
  method(r, "Out", "write_text", {"string", "string"}, "void", [](void*, Args& a) { write_text_atomic(a[0].s, a[1].s); return Value::none(); });
}

void print_summary(dslvm::Registry& reg) {
  std::vector<std::string> order, functions;
  std::map<std::string, std::vector<std::string>> members;
  std::cout << "lmtorch DSL host: compile with lmscript against `lmtorch --dsl-schema --json`, run with `lmtorch --dsl FILE.dslb`\n";
  std::cout << "enum DType: F32, F64, I64, BOOL\n";
  for (const auto& sig : reg.signatures()) {
    if (sig.rfind("context ", 0) == 0) std::cout << sig << "\n";
    else if (sig.rfind("function ", 0) == 0) functions.push_back(sig.substr(9));
    else if (sig.rfind("method ", 0) == 0) {
      std::string body = sig.substr(7), type = body.substr(0, body.find('.'));
      if (!members.count(type)) order.push_back(type);
      members[type].push_back(body.substr(type.size() + 1));
    }
  }
  for (const auto& type : order) {
    std::cout << type << ":\n";
    for (const auto& m : members[type]) std::cout << "  " << m << "\n";
  }
  std::cout << "lmcode std functions:\n";
  for (const auto& f : functions) std::cout << "  " << f << "\n";
}
}  // namespace

bool selects_dsl(int argc, char** argv) {
  if (argc < 2) return false;
  std::string a = argv[1];
  return a == "--dsl" || a == "--dsl-schema" || a == "--dsl-signatures";
}

int run(int argc, char** argv) {
  const std::string usage = "usage: lmtorch --dsl-schema [--json] | --dsl-signatures | --dsl FILE.dslb [FILE.dslb ...]\n";
  Host host;
  dslvm::Registry reg;
  register_torch(reg, host);
  std::string first = argv[1];
  if (first == "--dsl-schema") {
    if (argc == 2) { print_summary(reg); return std::cout ? 0 : 4; }
    if (argc == 3 && std::string(argv[2]) == "--json") {
      std::cout.write(reinterpret_cast<const char*>(embedded_dsl_schema), std::streamsize(embedded_dsl_schema_size));
      return std::cout ? 0 : 4;
    }
    std::cerr << usage;
    return 2;
  }
  if (first == "--dsl-signatures") {
    if (argc != 2) { std::cerr << usage; return 2; }
    for (const auto& s : reg.signatures()) std::cout << s << "\n";
    return 0;
  }
  if (first != "--dsl" || argc < 3) { std::cerr << usage; return 2; }
  for (int i = 2; i < argc; ++i) {
    std::string a = argv[i];
    if (a.empty() || a.rfind("--", 0) == 0) { std::cerr << usage; return 2; }
  }
  const std::string schema(reinterpret_cast<const char*>(embedded_dsl_schema), embedded_dsl_schema_size);
  // Every file is read and verified before any chunk runs, so a bad later file never leaves a half-done run.
  std::vector<dslvm::Program> programs;
  for (int i = 2; i < argc; ++i) {
    try {
      programs.push_back(dslvm::Program::load(dslvm::read_file(argv[i]), schema, reg));
    } catch (const dslvm::Error& e) {
      return dslvm::report_error("load", dslvm::Error(e.code, std::string(argv[i]) + ": " + e.what()));
    }
  }
  dslvm::Context ctx;
  ctx.set("torch", &host);
  ctx.set("out", &host.out);
  for (auto& program : programs) {
    // Autograd recording is per file: torch.grad_enabled(false) in one file must not silently disable training in the next.
    at::GradMode::set_enabled(true);
    try {
      program.run(ctx, reg);
    } catch (const dslvm::Error& e) {
      return dslvm::report_error(e.code == "APP_FAIL" ? "app" : "run", e);
    } catch (const std::exception& e) {
      return dslvm::report_error("run", dslvm::Error("RUN_HOST", e.what()));
    } catch (...) {
      return dslvm::report_error("run", dslvm::Error("RUN_HOST", "unknown failure"));
    }
  }
  return dslvm::report_ok(host.out.result.dump());
}
}  // namespace lmtorch::dsl
