#pragma once
#include <cstring>
#include <deque>
#include <fstream>
#include <map>
#include <nlohmann/json.hpp>
#include <random>
#include <sstream>

#include "dslvm.hpp"
#include "lmtorch/safetensors.hpp"
#include "tensor.hpp"
namespace lmtorch_lite {
using Json = nlohmann::ordered_json;
using dslvm::Args;
using dslvm::Value;
struct Box {
  Tensor t;
  bool kept = false;
};
struct Layer {
  std::string kind;
  std::vector<int64_t> args;
  double value = 0;
  Tensor w, b;
};
struct Model {
  std::vector<Layer> layers;
  bool training = true;
};
struct Optimizer {
  std::vector<Tensor> params;
  std::string kind;
  double lr = .01, momentum = 0, decay = 0;
  size_t step = 0;
  std::vector<std::vector<double>> m, v;
};
struct Bundle {
  std::map<std::string, Tensor> tensors;
  std::map<std::string, std::string> meta;
};
struct Host {
  std::deque<Box> tensors;
  std::deque<std::vector<Tensor>> lists;
  std::deque<Model> models;
  std::deque<Optimizer> optimizers;
  std::deque<Bundle> bundles;
  Json output = Json::object();
  std::mt19937 rng{0};
  bool spare_valid = false;
  double spare = 0;
  Value box(Tensor t) {
    tensors.push_back({t, false});
    return Value::object("Tensor", &tensors.back());
  }
  double uniform() { return double(float(rng() & 0xffffff) * (1.0f / 16777216.0f)); }
  uint64_t random64() {
    uint64_t a = rng();
    return (a << 32) | rng();
  }
  double uniform64() { return (random64() & ((1ULL << 53) - 1)) * (1.0 / 9007199254740992.0); }
  Tensor random(Shape s, bool normal = false) {
    auto t = make(s);
    if (!normal) {
      for (size_t i = 0; i < t->n(); ++i) t->set(i, uniform());
      return t;
    }
    // ATen CPU normal fill: 16-value transform for contiguous F32, cached double pairs for short draws.
    if (t->n() >= 16) {
      for (size_t i = 0; i < t->n(); ++i) t->set(i, uniform());
      auto block = [&](size_t at) {
        for (int j = 0; j < 8; ++j) {
          float u1 = 1 - float(t->get(at + j)), u2 = t->get(at + j + 8);
          float radius = std::sqrt(-2 * std::log(u1));
          float theta = float(2 * 3.141592653589793 * u2);
          t->set(at + j, float(radius * std::cos(theta)));
          t->set(at + j + 8, float(radius * std::sin(theta)));
        }
      };
      for (size_t i = 0; i + 15 < t->n(); i += 16) block(i);
      if (t->n() % 16) {
        size_t at = t->n() - 16;
        for (size_t j = 0; j < 16; ++j) t->set(at + j, uniform());
        block(at);
      }
    } else
      for (size_t i = 0; i < t->n(); ++i) {
        double z;
        if (spare_valid) {
          z = spare;
          spare_valid = false;
        } else {
          double u1 = uniform64(), u2 = uniform64();
          double radius = std::sqrt(-2 * std::log1p(-u2)), theta = 2 * 3.141592653589793 * u1;
          z = radius * std::cos(theta);
          spare = radius * std::sin(theta);
          spare_valid = true;
        }
        t->set(i, z);
      }
    return t;
  }
  void seed(int64_t s) {
    require(s >= 0, "seed must be nonnegative");
    rng.seed(uint32_t(s));
    spare_valid = false;
  }
};
inline Tensor tensor(const Value& v) {
  auto b = static_cast<Box*>(v.obj);
  require(b && bool(b->t), "tensor released by torch.release()");
  return b->t;
}
inline Tensor self_tensor(void* p) { return tensor(Value::object("Tensor", p)); }
inline Json tensor_json(Tensor t) {
  require(t->n() <= 1000000, "output tensor too large");
  size_t at = 0;
  std::function<Json(size_t)> rec = [&](size_t d) -> Json {
    if (d == t->shape.size()) {
      size_t i = at++;
      if (t->dtype() == "I64") return t->integer(i);
      if (t->dtype() == "BOOL") return bool(t->get(i));
      double x = t->get(i);
      return std::isfinite(x) ? Json(x) : Json(nullptr);
    }
    auto a = Json::array();
    for (int64_t i = 0; i < t->shape[d]; ++i) a.push_back(rec(d + 1));
    return a;
  };
  return rec(0);
}
inline Tensor parse_tensor(const std::string& text, const std::string& requested = "") {
  auto j = Json::parse(text);
  Shape s;
  bool floats = false, booleans = true;
  std::function<void(const Json&, size_t)> scan = [&](const Json& x, size_t depth) {
    if (x.is_array()) {
      if (s.size() <= depth) s.push_back(x.size());
      require(s[depth] == int64_t(x.size()), "ragged tensor");
      for (auto& e : x) scan(e, depth + 1);
    } else {
      require(x.is_number() || x.is_boolean(), "tensor requires numeric JSON");
      floats |= x.is_number_float();
      booleans &= x.is_boolean();
      require(depth == s.size(), "ragged tensor rank");
    }
  };
  scan(j, 0);
  auto t = make(s, requested.empty() ? (floats ? "F32" : booleans ? "BOOL" : "I64") : requested);
  size_t i = 0;
  std::function<void(const Json&)> fill = [&](const Json& x) {
    if (x.is_array())
      for (auto& e : x) fill(e);
    else {
      if (t->dtype() == "I64" && x.is_number_integer())
        t->set_int(i++, x.get<int64_t>());
      else
        t->set(i++, x.is_boolean() ? double(x.get<bool>()) : x.get<double>());
    }
  };
  fill(j);
  return t;
}
inline lmtorch::TensorData encode(Tensor t) {
  lmtorch::TensorData data;
  data.dtype = t->dtype();
  data.shape = t->shape;
  std::visit(
      [&](const auto& values) {
        using T = typename std::decay_t<decltype(values)>::value_type;
        data.bytes.resize(t->n() * sizeof(T));
        for (size_t i = 0; i < t->n(); ++i) {
          T v = values[t->index(i)];
          std::memcpy(data.bytes.data() + i * sizeof(T), &v, sizeof(T));
        }
      },
      t->storage->data);
  return data;
}
inline Tensor decode(const lmtorch::TensorData& data) {
  auto t = make(data.shape, data.dtype);
  std::visit(
      [&](auto& values) {
        require(values.size() * sizeof(typename std::decay_t<decltype(values)>::value_type) == data.bytes.size(),
                "invalid tensor bytes");
        std::memcpy(values.data(), data.bytes.data(), data.bytes.size());
      },
      t->storage->data);
  return t;
}
inline void save(const Bundle& b, const std::string& path) {
  lmtorch::TensorBundle v;
  v.metadata = b.meta;
  for (auto& [key, t] : b.tensors) v.tensors[key] = encode(t);
  lmtorch::write_bundle(path, v, true);
}
inline Bundle open(const std::string& path) {
  auto v = lmtorch::read_bundle(path);
  Bundle b;
  b.meta = v.metadata;
  for (auto& [k, t] : v.tensors) b.tensors[k] = decode(t);
  return b;
}
inline std::vector<Tensor> parameters(Model& m) {
  std::vector<Tensor> p;
  for (auto& l : m.layers) {
    if (l.w) p.push_back(l.w);
    if (l.b) p.push_back(l.b);
  }
  return p;
}
inline std::string summary(Model& m) {
  std::string s;
  for (auto& l : m.layers) {
    if (!s.empty()) s += " -> ";
    s += l.kind;
    if (!l.args.empty() || l.kind == "dropout" || l.kind == "leaky_relu") {
      s += "(";
      for (size_t i = 0; i < l.args.size(); ++i) s += (i ? ", " : "") + std::to_string(l.args[i]);
      if (l.kind == "dropout" || l.kind == "leaky_relu") {
        std::ostringstream number;
        number.precision(17);
        number << l.value;
        s += number.str();
      }
      s += ")";
    }
  }
  return s;
}
inline void add_layer(Host& h, Model& m, std::string kind, std::vector<int64_t> args = {}, double value = 0) {
  Layer l{kind, args, value, nullptr, nullptr};
  if (kind == "linear" || kind == "conv2d") {
    require(args[0] > 0 && args[1] > 0, "layer dimensions must be positive");
    int64_t fan = args[0];
    Shape shape = {args[1], args[0]};
    if (kind == "conv2d") {
      require(args[2] > 0, "kernel must be positive");
      shape.push_back(args[2]);
      shape.push_back(args[2]);
      fan *= args[2] * args[2];
    }
    float bound = 1 / std::sqrt(float(fan));
    l.w = h.random(shape);
    l.b = h.random({args[1]});
    for (auto t : {l.w, l.b}) {
      for (size_t i = 0; i < t->n(); ++i) t->set(i, float(float(t->get(i)) * float(2 * bound) - bound));
      t->requires = true;
    }
  }
  if (kind == "layer_norm") {
    l.w = make({args[0]}, "F32", 1);
    l.b = make({args[0]});
    l.w->requires = l.b->requires = true;
  }
  m.layers.push_back(std::move(l));
}
inline Tensor forward(Host& h, Model& m, Tensor x) {
  for (auto& l : m.layers) {
    if (l.kind == "linear")
      x = binary(matmul(x, transpose(l.w, 0, 1)), l.b, "add");
    else if (l.kind == "conv2d")
      x = conv2d(x, l.w, l.b, l.args.size() > 3 ? l.args[3] : 1, l.args.size() > 4 ? l.args[4] : 0);
    else if (l.kind == "max_pool2d")
      x = maxpool(x, l.args[0]);
    else if (l.kind == "flatten")
      x = reshape(x, {x->shape[0], int64_t(x->n() / x->shape[0])});
    else if (l.kind == "softmax" || l.kind == "log_softmax")
      x = softmax(x, l.args[0], l.kind == "log_softmax");
    else if (l.kind == "layer_norm") {
      int d = x->shape.size() - 1;
      auto mean = reduce(x, d, true, "mean");
      auto centered = binary(x, mean, "sub");
      auto variance = reduce(unary(centered, "square"), d, true, "mean");
      x = binary(binary(binary(centered, unary(binary(variance, scalar(1e-5), "add"), "sqrt"), "div"), l.w, "mul"), l.b,
                 "add");
    } else if (l.kind == "dropout") {
      if (m.training) {
        auto mask = h.random(x->shape);
        for (size_t i = 0; i < mask->n(); ++i) mask->set(i, mask->get(i) >= l.value ? 1 / (1 - l.value) : 0);
        x = binary(x, mask, "mul");
      }
    } else
      x = unary(x, l.kind, l.value);
  }
  return x;
}
inline Bundle model_bundle(Model& m) {
  Bundle b;
  b.meta["torchcli_format"] = "model-1";
  b.meta["layers"] = summary(m);
  for (size_t i = 0; i < m.layers.size(); ++i) {
    auto& l = m.layers[i];
    if (l.w) b.tensors[std::to_string(i) + ".weight"] = l.w;
    if (l.b) b.tensors[std::to_string(i) + ".bias"] = l.b;
  }
  return b;
}
inline void load_weights(Model& m, const Bundle& b) {
  auto expected = model_bundle(m);
  require(expected.tensors.size() == b.tensors.size(), "model key mismatch");
  for (auto& [key, t] : expected.tensors) {
    auto it = b.tensors.find(key);
    require(it != b.tensors.end() && it->second->shape == t->shape, "model weight shape/key mismatch");
    for (size_t i = 0; i < t->n(); ++i) t->set(i, it->second->get(i));
  }
}
inline Model load_model(Host& h, const std::string& path) {
  auto b = open(path);
  require(b.meta["torchcli_format"] == "model-1", "invalid model metadata");
  Model m;
  auto text = b.meta.at("layers");
  size_t at = 0;
  while (at < text.size()) {
    auto end = text.find(" -> ", at);
    std::string spec = text.substr(at, end == std::string::npos ? std::string::npos : end - at);
    auto p = spec.find('(');
    std::string kind = spec.substr(0, p);
    std::vector<int64_t> args;
    double value = 0;
    if (p != std::string::npos) {
      require(spec.back() == ')', "invalid architecture");
      std::stringstream ss(spec.substr(p + 1, spec.size() - p - 2));
      std::string token;
      while (std::getline(ss, token, ',')) {
        if (kind == "dropout" || kind == "leaky_relu")
          value = std::stod(token);
        else
          args.push_back(std::stoll(token));
      }
    }
    add_layer(h, m, kind, args, value);
    if (end == std::string::npos) break;
    at = end + 4;
  }
  load_weights(m, b);
  return m;
}
inline void optimizer_step(Optimizer& o) {
  ++o.step;
  if (o.m.empty()) {
    for (auto p : o.params) {
      o.m.emplace_back(p->n(), 0);
      o.v.emplace_back(p->n(), 0);
    }
  }
  for (size_t k = 0; k < o.params.size(); ++k) {
    auto p = o.params[k];
    if (p->grad.empty()) continue;
    for (size_t i = 0; i < p->n(); ++i) {
      double g = p->grad[i], update = g;
      if (o.kind == "sgd") {
        if (o.momentum) {
          o.m[k][i] = p->dtype() == "F32" ? double(float(float(o.momentum) * float(o.m[k][i])) + float(g))
                                          : o.momentum * o.m[k][i] + g;
          update = o.m[k][i];
        }
      } else {
        double m = .9 * o.m[k][i] + .1 * g, v = .999 * o.v[k][i] + .001 * g * g;
        if (p->dtype() == "F32") {
          m = float(m);
          v = float(v);
        }
        o.m[k][i] = m;
        o.v[k][i] = v;
        update =
            (m / (1 - std::pow(.9, double(o.step)))) / (std::sqrt(v / (1 - std::pow(.999, double(o.step)))) + 1e-8);
      }
      double value = p->get(i);
      if (o.kind == "adamw") value *= 1 - o.lr * o.decay;
      p->set(i, p->dtype() == "F32" && o.kind == "sgd" ? double(float(value) + float(-float(o.lr) * float(update)))
                                                       : value - o.lr * update);
    }
  }
}
inline Json json_value(const Value& v) {
  switch (v.kind) {
    case Value::INT:
      return v.i;
    case Value::FLOAT:
      return std::isfinite(v.f) ? Json(v.f) : Json(nullptr);
    case Value::STR:
      return v.s;
    case Value::BOOL:
      return v.b;
    case Value::OBJ:
      if (v.type == "Tensor") return tensor_json(tensor(v));
    default:
      break;
  }
  throw std::runtime_error("unsupported output value");
}
inline Value invoke(Host& h, dslvm::Registry& r, const std::string& type, const std::string& name, void* p, Args& a) {
  auto ret = [&](Tensor t) { return h.box(t); };
  auto same = [&] { return Value::object(type, p); };
  if (type == "Torch") {
    if (name == "version") return Value::string("lmtorch lite (portable CPU)");
    if (name == "manual_seed") {
      h.seed(a[0].i);
      return Value::none();
    }
    if (name == "set_num_threads") {
      require(a[0].i >= 1 && a[0].i <= 256, "invalid thread count");
      return Value::none();
    }
    if (name == "grad_enabled") {
      grad_enabled = a[0].b;
      return Value::none();
    }
    if (name == "is_grad_enabled") return Value::boolean(grad_enabled);
    if (name == "tensor") return ret(parse_tensor(a[0].s, a.size() > 1 ? a[1].s : ""));
    if (name == "scalar") return ret(scalar(a[0].f));
    if (name == "zeros" || name == "ones" || name == "full" || name == "rand" || name == "randn") {
      Shape shape;
      for (size_t i = 0; i < a.size() - (name == "full"); ++i) shape.push_back(a[i].i);
      return ret(name == "rand" || name == "randn" ? h.random(shape, name == "randn")
                                                   : make(shape, "F32",
                                                          name == "ones"   ? 1
                                                          : name == "full" ? a.back().f
                                                                           : 0));
    }
    if (name == "randint") {
      require(a[1].i > a[0].i, "randint needs low < high");
      Shape s;
      for (size_t i = 2; i < a.size(); ++i) s.push_back(a[i].i);
      auto t = make(s, "I64");
      uint64_t range = a[1].i - a[0].i;
      for (size_t i = 0; i < t->n(); ++i)
        t->set_int(i, (range >= (1ULL << 28) ? h.random64() : h.rng()) % range + a[0].i);
      return ret(t);
    }
    if (name == "randperm") {
      auto t = make({a[0].i}, "I64");
      for (size_t i = 0; i < t->n(); ++i) t->set_int(i, i);
      for (size_t i = 0; i + 1 < t->n(); ++i) {
        size_t j = i + h.rng() % (t->n() - i);
        auto v = t->integer(i);
        t->set_int(i, t->integer(j));
        t->set_int(j, v);
      }
      return ret(t);
    }
    if (name == "arange") {
      double start = a.size() == 1 ? 0 : a[0].f, end = a.size() == 1 ? double(a[0].i) : a[1].f,
             step = a.size() == 1 ? 1 : a[2].f;
      require(step != 0 && (end - start) / step >= 0, "invalid arange");
      auto t = make({int64_t(std::ceil((end - start) / step))}, a.size() == 1 ? "I64" : "F32");
      for (size_t i = 0; i < t->n(); ++i) t->set(i, start + step * i);
      return ret(t);
    }
    if (name == "linspace") {
      auto t = make({a[2].i});
      for (size_t i = 0; i < t->n(); ++i)
        t->set(i, t->n() < 2 ? a[0].f : a[0].f + (a[1].f - a[0].f) * i / (t->n() - 1));
      return ret(t);
    }
    if (name == "eye") {
      auto t = make({a[0].i, a[0].i});
      for (int64_t i = 0; i < a[0].i; ++i) t->set(i * a[0].i + i, 1);
      return ret(t);
    }
    if (name == "list") {
      h.lists.emplace_back();
      return Value::object("TensorList", &h.lists.back());
    }
    if (name == "cat" || name == "stack")
      return ret(concat(*static_cast<std::vector<Tensor>*>(a[0].obj), a[1].i, name == "stack"));
    if (name == "where") {
      auto cond = tensor(a[0]), x = tensor(a[1]), y = tensor(a[2]);
      require(cond->dtype() == "BOOL", "where needs BOOL");
      auto mask = make(cond->shape, x->dtype());
      for (size_t i = 0; i < mask->n(); ++i) mask->set(i, cond->get(i));
      return ret(binary(binary(x, mask, "mul"), binary(y, binary(scalar(1), mask, "sub"), "mul"), "add"));
    }
    if (name == "mark") return Value::integer(h.tensors.size());
    if (name == "release") {
      require(a[0].i >= 0 && size_t(a[0].i) <= h.tensors.size(), "invalid release mark");
      int64_t n = 0;
      for (size_t i = a[0].i; i < h.tensors.size(); ++i)
        if (!h.tensors[i].kept && h.tensors[i].t) {
          h.tensors[i].t.reset();
          ++n;
        }
      return Value::integer(n);
    }
    if (name == "live_tensors") {
      int64_t n = 0;
      for (auto& b : h.tensors) n += bool(b.t);
      return Value::integer(n);
    }
    if (name == "sequential") {
      h.models.emplace_back();
      return Value::object("Model", &h.models.back());
    }
    if (name == "load_model") {
      h.models.push_back(load_model(h, a[0].s));
      return Value::object("Model", &h.models.back());
    }
    if (name == "bundle" || name == "open") {
      h.bundles.push_back(name == "open" ? open(a[0].s) : Bundle{});
      return Value::object("Bundle", &h.bundles.back());
    }
    if (name == "load") {
      auto b = open(a[0].s);
      return ret(b.tensors.at(a.size() > 1 ? a[1].s : "tensor"));
    }
    if (name == "save") {
      Bundle b;
      b.tensors["tensor"] = tensor(a[1]);
      save(b, a[0].s);
      return Value::none();
    }
    if (name == "read_csv") {
      lmtorch::check_contained(a[0].s);
      std::ifstream file(a[0].s);
      require(bool(file), "cannot read CSV");
      std::string line;
      Json rows = Json::array();
      if (a[1].b) std::getline(file, line);
      size_t cols = 0;
      while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string cell;
        Json row = Json::array();
        while (std::getline(ss, cell, ',')) {
          size_t used = 0;
          double v = std::stod(cell, &used);
          require(std::isfinite(v) && cell.find_first_not_of(" \t\r", used) == std::string::npos, "invalid CSV cell");
          row.push_back(v);
        }
        if (!cols) cols = row.size();
        require(row.size() == cols, "CSV column count");
        rows.push_back(row);
      }
      return ret(parse_tensor(rows.dump(), "F32"));
    }
    if (name == "sgd" || name == "adam" || name == "adamw") {
      Optimizer o;
      o.kind = name;
      o.lr = a[1].f;
      require(o.lr >= 0, "invalid learning rate");
      o.params = a[0].type == "Model" ? parameters(*static_cast<Model*>(a[0].obj))
                                      : *static_cast<std::vector<Tensor>*>(a[0].obj);
      if (a.size() > 2) {
        if (name == "sgd")
          o.momentum = a[2].f;
        else
          o.decay = a[2].f;
      }
      h.optimizers.push_back(std::move(o));
      return Value::object("Optimizer", &h.optimizers.back());
    }
  }
  if (type == "Tensor") {
    auto t = self_tensor(p);
    if (name == "numel") return Value::integer(t->n());
    if (name == "dim") return Value::integer(t->shape.size());
    if (name == "size") return Value::integer(t->shape[axis(a[0].i, t->shape.size())]);
    if (name == "shape") {
      std::string s = "[";
      for (size_t i = 0; i < t->shape.size(); ++i) s += (i ? ", " : "") + std::to_string(t->shape[i]);
      return Value::string(s + "]");
    }
    if (name == "dtype") return Value::string(t->dtype());
    if (name == "item" || name == "item_int") {
      require(t->n() == 1, "item needs one element");
      return name == "item" ? Value::real(t->get(0)) : Value::integer(t->integer(0));
    }
    if (name == "value" || name == "value_int") {
      require(a[0].i >= 0, "negative flat index");
      return name == "value" ? Value::real(t->get(a[0].i)) : Value::integer(t->integer(a[0].i));
    }
    if (name == "to_json" || name == "to_text") return Value::string(tensor_json(t).dump());
    if (name == "to") {
      auto v = make(t->shape, a[0].s);
      for (size_t i = 0; i < t->n(); ++i)
        if (t->dtype() == "I64" && v->dtype() == "I64")
          v->set_int(i, t->integer(i));
        else
          v->set(i, t->get(i));
      return ret(graph(v, {t}, [t](const auto& g) {
        for (size_t i = 0; i < g.size(); ++i) accumulate(t, i, g[i]);
      }));
    }
    if (name == "set") {
      t->set(a[0].i, a[1].f);
      return same();
    }
    if (name == "keep") {
      static_cast<Box*>(p)->kept = true;
      return same();
    }
    if (name == "requires_grad") {
      if (a.empty()) return Value::boolean(t->requires);
      require((t->dtype() == "F32" || t->dtype() == "F64") && t->parents.empty(),
              "requires_grad needs a floating leaf");
      t->requires = a[0].b;
      return same();
    }
    if (name == "backward") {
      backward(t);
      return Value::none();
    }
    if (name == "zero_grad") {
      t->grad.clear();
      return same();
    }
    if (name == "has_grad") return Value::boolean(!t->grad.empty());
    if (name == "grad") {
      require(!t->grad.empty(), "tensor has no gradient");
      auto v = make(t->shape, t->dtype());
      for (size_t i = 0; i < t->n(); ++i) v->set(i, t->grad[i]);
      return ret(v);
    }
    if (name == "detach") {
      auto v = make(t->shape, t->dtype());
      v->storage = t->storage;
      v->offsets = t->offsets;
      return ret(v);
    }
    if (name == "clone") {
      std::vector<size_t> map(t->n());
      std::iota(map.begin(), map.end(), 0);
      return ret(mapped(t, t->shape, map));
    }
    if (name == "add" || name == "sub" || name == "mul" || name == "div" || name == "maximum" || name == "minimum" ||
        name == "eq" || name == "gt" || name == "lt" || name == "ge" || name == "le") {
      auto rhs = a[0].kind == Value::OBJ ? tensor(a[0]) : scalar(a[0].f, t->dtype() == "F64" ? "F64" : "F32");
      return ret(binary(t, rhs, name));
    }
    if (name == "rsub") return ret(binary(scalar(a[0].f, t->dtype() == "F64" ? "F64" : "F32"), t, "sub"));
    if (name == "matmul") return ret(matmul(t, tensor(a[0])));
    if (name == "linear") {
      auto x = matmul(t, transpose(tensor(a[0]), 0, 1));
      if (a.size() > 1) x = binary(x, tensor(a[1]), "add");
      return ret(x);
    }
    if (name == "sum" || name == "mean" || name == "std" || name == "var" || name == "max" || name == "min" ||
        name == "argmax" || name == "argmin" || name == "amax" || name == "amin")
      return ret(reduce(t, a.empty() ? 999 : int(a[0].i), a.size() > 1 && a[1].b,
                        name == "amax"   ? "max"
                        : name == "amin" ? "min"
                                         : name));
    if (name == "reshape") {
      Shape s;
      for (auto v : a) s.push_back(v.i);
      return ret(reshape(t, s));
    }
    if (name == "flatten") return ret(reshape(t, {int64_t(t->n())}));
    if (name == "t") {
      require(t->shape.size() == 2, "t needs rank 2");
      return ret(transpose(t, 0, 1));
    }
    if (name == "transpose") return ret(transpose(t, a[0].i, a[1].i));
    if (name == "unsqueeze") {
      Shape s = t->shape;
      int d = a[0].i;
      if (d < 0) d += s.size() + 1;
      require(d >= 0 && d <= int(s.size()), "unsqueeze dimension");
      s.insert(s.begin() + d, 1);
      return ret(reshape(t, s));
    }
    if (name == "squeeze") {
      Shape s = t->shape;
      if (a.empty())
        s.erase(std::remove(s.begin(), s.end(), 1), s.end());
      else {
        int d = axis(a[0].i, s.size());
        if (s[d] == 1) s.erase(s.begin() + d);
      }
      return ret(reshape(t, s));
    }
    if (name == "slice") {
      int d = axis(a[0].i, t->shape.size());
      auto n = t->shape[d];
      int64_t start = a[1].i, end = a[2].i;
      if (start < 0) start += n;
      if (end < 0) end += n;
      start = std::clamp(start, int64_t(0), n);
      end = std::clamp(end, int64_t(0), n);
      std::vector<int64_t> ids;
      for (auto i = start; i < end; ++i) ids.push_back(i);
      return ret(select(t, d, ids));
    }
    if (name == "select") return ret(select(t, a[0].i, {a[1].i}, true));
    if (name == "index_select") {
      auto ids = tensor(a[1]);
      require(ids->dtype() == "I64" && ids->shape.size() == 1, "index_select needs I64 vector");
      std::vector<int64_t> v;
      for (size_t i = 0; i < ids->n(); ++i) v.push_back(ids->integer(i));
      return ret(select(t, a[0].i, v));
    }
    if (name == "softmax" || name == "log_softmax") return ret(softmax(t, a[0].i, name == "log_softmax"));
    if (name == "cross_entropy") return ret(cross_entropy(t, tensor(a[0])));
    if (name == "mse_loss" || name == "l1_loss" || name == "smooth_l1_loss") {
      auto d = binary(t, tensor(a[0]), "sub");
      if (name == "smooth_l1_loss") throw std::runtime_error("lite unsupported Tensor.smooth_l1_loss");
      return ret(reduce(unary(d, name == "mse_loss" ? "square" : "abs"), 999, false, "mean"));
    }
    if (name == "bce_with_logits") {
      auto y = tensor(a[0]);
      require(t->shape == y->shape, "BCE target shape");
      auto v = make(t->shape, t->dtype());
      for (size_t i = 0; i < t->n(); ++i) {
        double x = t->get(i);
        v->set(i, std::max(0., x) - x * y->get(i) + std::log1p(std::exp(-std::abs(x))));
      }
      auto g = graph(v, {t}, [t, y](const auto& g) {
        for (size_t i = 0; i < g.size(); ++i) accumulate(t, i, g[i] * (1 / (1 + std::exp(-t->get(i))) - y->get(i)));
      });
      return ret(reduce(g, 999, false, "mean"));
    }
    if (name == "allclose" || name == "equal") {
      auto b = tensor(a[0]);
      bool ok = t->shape == b->shape && (name != "equal" || t->dtype() == b->dtype());
      if (ok)
        for (size_t i = 0; i < t->n(); ++i) {
          double x = t->get(i), y = b->get(i);
          if (name == "equal" && t->dtype() == "I64")
            ok &= t->integer(i) == b->integer(i);
          else
            ok &= std::isfinite(x) && std::isfinite(y) &&
                  std::abs(x - y) <= (name == "equal" ? 0 : a[2].f + a[1].f * std::abs(y));
        }
      return Value::boolean(ok);
    }
    if (name == "is_finite") {
      bool finite = true;
      for (size_t i = 0; i < t->n(); ++i) finite &= std::isfinite(t->get(i));
      return Value::boolean(finite);
    }
    if (name == "any" || name == "all") {
      bool result = name == "all";
      for (size_t i = 0; i < t->n(); ++i) {
        if (name == "all")
          result &= bool(t->get(i));
        else
          result |= bool(t->get(i));
      }
      return Value::boolean(result);
    }
    if (name == "norm") return ret(unary(reduce(unary(t, "square"), 999, false, "sum"), "sqrt"));
    if (name == "clamp")
      return ret(binary(binary(t, scalar(a[0].f, t->dtype()), "maximum"), scalar(a[1].f, t->dtype()), "minimum"));
    if (name == "save") {
      Bundle b;
      b.tensors["tensor"] = t;
      save(b, a[0].s);
      return same();
    }
    if (name == "write_csv") {
      require(t->shape.size() == 1 || t->shape.size() == 2, "CSV needs rank 1 or 2");
      lmtorch::check_contained(a[0].s);
      std::ofstream f(a[0].s);
      require(bool(f), "cannot write CSV");
      f.precision(17);
      size_t cols = t->shape.size() == 1 ? 1 : t->shape[1];
      for (size_t i = 0; i < t->n(); ++i) f << t->get(i) << ((i + 1) % cols ? ',' : '\n');
      return same();
    }
    static const std::set<std::string> ops = {"neg",  "square",     "pow",   "abs",   "exp",     "log",
                                              "sqrt", "sin",        "cos",   "relu",  "sigmoid", "tanh",
                                              "gelu", "leaky_relu", "floor", "round", "sign"};
    if (ops.count(name)) return ret(unary(t, name, a.empty() ? 0 : a[0].f));
  }
  if (type == "TensorList") {
    auto& xs = *static_cast<std::vector<Tensor>*>(p);
    if (name == "add") {
      xs.push_back(tensor(a[0]));
      return same();
    }
    if (name == "len") return Value::integer(xs.size());
    if (name == "at") return ret(xs.at(a[0].i));
  }
  if (type == "Bundle") {
    auto& b = *static_cast<Bundle*>(p);
    if (name == "put") {
      b.tensors[a[0].s] = tensor(a[1]);
      return same();
    }
    if (name == "get") return ret(b.tensors.at(a[0].s));
    if (name == "has") return Value::boolean(b.tensors.count(a[0].s));
    if (name == "has_meta") return Value::boolean(b.meta.count(a[0].s));
    if (name == "meta") {
      if (a.size() == 1) return Value::string(b.meta.at(a[0].s));
      b.meta[a[0].s] = a[1].s;
      return same();
    }
    if (name == "keys") {
      std::vector<std::string> keys;
      for (auto& [k, v] : b.tensors) keys.push_back(k);
      return Value::object("StrList", r.std_collections().make_str_list(keys));
    }
    if (name == "save") {
      save(b, a[0].s);
      return Value::none();
    }
  }
  if (type == "Model") {
    auto& m = *static_cast<Model*>(p);
    if (name == "forward") return ret(forward(h, m, tensor(a[0])));
    if (name == "summary") return Value::string(summary(m));
    if (name == "layers") return Value::integer(m.layers.size());
    if (name == "num_parameters") {
      size_t n = 0;
      for (auto t : parameters(m)) n += t->n();
      return Value::integer(n);
    }
    if (name == "parameters") {
      h.lists.push_back(parameters(m));
      return Value::object("TensorList", &h.lists.back());
    }
    if (name == "param") {
      auto b = model_bundle(m);
      return ret(b.tensors.at(a[0].s));
    }
    if (name == "zero_grad") {
      for (auto t : parameters(m)) t->grad.clear();
      return same();
    }
    if (name == "train" || name == "eval") {
      m.training = name == "train";
      return same();
    }
    if (name == "save") {
      save(model_bundle(m), a[0].s);
      return Value::none();
    }
    if (name == "load") {
      load_weights(m, open(a[0].s));
      return same();
    }
    static const std::set<std::string> kinds = {"linear",      "conv2d",  "max_pool2d", "flatten",    "relu",
                                                "sigmoid",     "tanh",    "gelu",       "leaky_relu", "softmax",
                                                "log_softmax", "dropout", "layer_norm"};
    if (kinds.count(name)) {
      std::vector<int64_t> args;
      double value = 0;
      for (auto& v : a) {
        if (v.kind == Value::INT)
          args.push_back(v.i);
        else
          value = v.f;
      }
      add_layer(h, m, name, args, value);
      return same();
    }
  }
  if (type == "Optimizer") {
    auto& o = *static_cast<Optimizer*>(p);
    if (name == "lr") return Value::real(o.lr);
    if (name == "set_lr") {
      require(a[0].f >= 0, "invalid learning rate");
      o.lr = a[0].f;
      return same();
    }
    if (name == "zero_grad") {
      for (auto t : o.params) t->grad.clear();
      return Value::none();
    }
    if (name == "step") {
      optimizer_step(o);
      return Value::none();
    }
  }
  if (type == "Out") {
    if (name == "set") {
      h.output[a[0].s] = json_value(a[1]);
      return same();
    }
    if (name == "append") {
      auto& v = h.output[a[0].s];
      if (v.is_null()) v = Json::array();
      v.push_back(json_value(a[1]));
      return same();
    }
    if (name == "log") return Value::none();
    if (name == "fail" || (name == "check" && !a[0].b)) throw dslvm::Error("APP_FAIL", a[name == "fail" ? 0 : 1].s);
    if (name == "check") return Value::none();
    if (name == "write_text") {
      lmtorch::check_contained(a[0].s);
      std::ofstream f(a[0].s);
      require(bool(f), "cannot write text");
      f << a[1].s;
      return Value::none();
    }
  }
  throw dslvm::Error("RUN_UNSUPPORTED", "lite unsupported " + type + "." + name);
}
inline void register_bindings(dslvm::Registry& r, Host& h, const std::string& schema) {
  auto j = Json::parse(schema);
  r.context("torch", "Torch");
  r.context("out", "Out");
  for (auto it = j["types"].begin(); it != j["types"].end(); ++it) {
    std::string type = it.key();
    if (type == "Experience") continue;
    if (!it.value().contains("methods")) continue;
    for (auto m = it.value()["methods"].begin(); m != it.value()["methods"].end(); ++m) {
      std::string name = m.key();
      for (auto& ov : m.value()) {
        auto params = ov["params"].get<std::vector<std::string>>();
        std::string returns = ov["returns"].get<std::string>();
        if (returns == "self") returns = type;
        r.method(type, name, params, returns, [&h, &r, type, name, returns](void* p, Args& a) {
          try {
            auto value = invoke(h, r, type, name, p, a);
            bool ok = returns == "void"                           ? value.kind == Value::VOID
                      : returns == "bool"                         ? value.kind == Value::BOOL
                      : returns == "int"                          ? value.kind == Value::INT
                      : returns == "float"                        ? value.kind == Value::FLOAT
                      : returns == "string" || returns == "DType" ? value.kind == Value::STR
                                                                  : value.kind == Value::OBJ && value.type == returns;
            if (!ok) throw dslvm::Error("RUN_HOST", "lite result type mismatch: " + type + "." + name);
            return value;
          } catch (const dslvm::Error&) {
            throw;
          } catch (const std::exception& e) {
            throw dslvm::Error("RUN_HOST", e.what());
          }
        });
      }
    }
  }
}
}  // namespace lmtorch_lite
