// Small, portable CPU tensor graph. No libtorch headers or external numerical library.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>
namespace lmtorch_lite {
using Shape = std::vector<int64_t>;
inline void require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
inline void require(bool ok, const std::string& message) {
  if (!ok) throw std::runtime_error(message);
}
inline size_t elements(const Shape& s) {
  size_t n = 1;
  for (auto d : s) {
    require(d >= 0, "negative dimension");
    require(!d || n <= 16000000 / size_t(d), "tensor exceeds 16 million elements");
    n *= d;
  }
  return n;
}
inline int axis(int d, int rank) {
  if (d < 0) d += rank;
  require(d >= 0 && d < rank, "dimension out of range");
  return d;
}
struct Storage {
  std::variant<std::vector<float>, std::vector<double>, std::vector<int64_t>, std::vector<uint8_t>> data;
  std::string dtype;
  uint64_t version = 0;
  Storage(size_t n, const std::string& dt) : dtype(dt) {
    if (dt == "F32")
      data = std::vector<float>(n);
    else if (dt == "F64")
      data = std::vector<double>(n);
    else if (dt == "I64")
      data = std::vector<int64_t>(n);
    else if (dt == "BOOL")
      data = std::vector<uint8_t>(n);
    else
      require(false, "unknown dtype");
  }
  double get(size_t i) const {
    return std::visit([i](const auto& x) -> double { return double(x.at(i)); }, data);
  }
  int64_t integer(size_t i) const {
    if (dtype == "I64") return std::get<std::vector<int64_t>>(data).at(i);
    return int64_t(get(i));
  }
  void set(size_t i, double v) {
    ++version;
    std::visit(
        [&](auto& x) {
          using T = typename std::decay_t<decltype(x)>::value_type;
          x.at(i) = static_cast<T>(v);
        },
        data);
  }
  void set_int(size_t i, int64_t v) {
    ++version;
    if (dtype == "I64")
      std::get<std::vector<int64_t>>(data).at(i) = v;
    else
      set(i, double(v));
  }
};
struct Node;
using Tensor = std::shared_ptr<Node>;
struct Node {
  Shape shape;
  size_t count_;
  std::shared_ptr<Storage> storage;
  std::vector<size_t> offsets;
  bool requires = false;
  std::vector<double> grad;
  std::vector<Tensor> parents;
  std::function<void(const std::vector<double>&)> backward;
  Node(Shape s, const std::string& dt)
      : shape(std::move(s)), count_(elements(shape)), storage(std::make_shared<Storage>(count_, dt)) {}
  size_t n() const { return count_; }
  const std::string& dtype() const { return storage->dtype; }
  size_t index(size_t i) const {
    require(i < n(), "flat index out of range");
    return offsets.empty() ? i : offsets.at(i);
  }
  double get(size_t i) const { return storage->get(index(i)); }
  int64_t integer(size_t i) const { return storage->integer(index(i)); }
  void set(size_t i, double v) { storage->set(index(i), v); }
  void set_int(size_t i, int64_t v) { storage->set_int(index(i), v); }
};
inline bool grad_enabled = true;
inline Tensor make(Shape s, const std::string& dt = "F32", double fill = 0) {
  auto t = std::make_shared<Node>(s, dt);
  if (fill != 0)
    for (size_t i = 0; i < t->n(); ++i) t->set(i, fill);
  return t;
}
inline Tensor scalar(double x, const std::string& dt = "F32") {
  auto t = make({}, dt);
  t->set(0, x);
  return t;
}
inline void accumulate(const Tensor& t, size_t i, double v) {
  if (!t->requires) return;
  if (t->grad.empty()) t->grad.resize(t->n());
  double x = t->grad[i] + v;
  t->grad[i] = t->dtype() == "F32" ? double(float(x)) : x;
}
inline Tensor graph(Tensor r, std::vector<Tensor> parents, std::function<void(const std::vector<double>&)> back) {
  r->requires =
      grad_enabled && std::any_of(parents.begin(), parents.end(), [](const Tensor& t) { return t->requires; });
  if (r->requires) {
    std::vector<uint64_t> versions;
    for (auto& parent : parents) versions.push_back(parent->storage->version);
    r->parents = parents;
    r->backward = [parents, versions, back = std::move(back)](const auto& g) {
      for (size_t i = 0; i < parents.size(); ++i)
        require(parents[i]->storage->version == versions[i], "tensor used by backward was modified in place");
      back(g);
    };
  }
  return r;
}
inline void backward(const Tensor& loss) {
  require(loss->n() == 1 && loss->requires, "backward needs one differentiable value");
  std::set<Node*> seen;
  std::vector<Tensor> order;
  std::function<void(Tensor)> visit = [&](Tensor t) {
    if (!seen.insert(t.get()).second) return;
    for (auto p : t->parents) visit(p);
    order.push_back(t);
  };
  visit(loss);
  for (auto t : order)
    if (!t->parents.empty()) t->grad.clear();
  accumulate(loss, 0, 1);
  for (auto it = order.rbegin(); it != order.rend(); ++it)
    if ((*it)->backward) (*it)->backward((*it)->grad);
}
inline Shape broadcast(Shape a, Shape b) {
  size_t n = std::max(a.size(), b.size());
  a.insert(a.begin(), n - a.size(), 1);
  b.insert(b.begin(), n - b.size(), 1);
  for (size_t i = 0; i < n; ++i) {
    require(a[i] == b[i] || a[i] == 1 || b[i] == 1, "incompatible broadcast");
    a[i] = a[i] == 1 ? b[i] : a[i];
  }
  return a;
}
inline size_t broadcast_index(size_t i, const Shape& out, const Shape& in) {
  size_t result = 0, stride = 1;
  for (int j = int(out.size()) - 1, k = int(in.size()) - 1; j >= 0; --j, --k) {
    size_t coord = out[j] ? i % out[j] : 0;
    i = out[j] ? i / out[j] : 0;
    if (k >= 0) {
      if (in[k] != 1) result += coord * stride;
      stride *= in[k];
    }
  }
  return result;
}
inline std::string promote(const Tensor& a, const Tensor& b) {
  if (a->dtype() == "F64" || b->dtype() == "F64") return "F64";
  if (a->dtype() == "F32" || b->dtype() == "F32") return "F32";
  return a->dtype();
}
inline Tensor binary(Tensor a, Tensor b, const std::string& op) {
  Shape s = broadcast(a->shape, b->shape);
  bool compare = op == "eq" || op == "gt" || op == "lt" || op == "ge" || op == "le";
  auto r = make(s, compare ? "BOOL" : (op == "div" && promote(a, b) == "I64" ? "F32" : promote(a, b)));
  for (size_t i = 0; i < r->n(); ++i) {
    double x = a->get(broadcast_index(i, s, a->shape)), y = b->get(broadcast_index(i, s, b->shape));
    double z = 0;
    if (op == "add")
      z = x + y;
    else if (op == "sub")
      z = x - y;
    else if (op == "mul")
      z = x * y;
    else if (op == "div")
      z = x / y;
    else if (op == "maximum")
      z = std::max(x, y);
    else if (op == "minimum")
      z = std::min(x, y);
    else if (op == "eq")
      z = x == y;
    else if (op == "gt")
      z = x > y;
    else if (op == "lt")
      z = x < y;
    else if (op == "ge")
      z = x >= y;
    else if (op == "le")
      z = x <= y;
    else
      require(false, "unknown binary op");
    r->set(i, z);
  }
  if (compare) return r;
  return graph(r, {a, b}, [a, b, s, op](const auto& g) {
    for (size_t i = 0; i < g.size(); ++i) {
      auto ai = broadcast_index(i, s, a->shape), bi = broadcast_index(i, s, b->shape);
      double x = a->get(ai), y = b->get(bi), da = 1, db = 1;
      if (op == "sub") db = -1;
      if (op == "mul") {
        da = y;
        db = x;
      }
      if (op == "div") {
        da = 1 / y;
        db = -x / (y * y);
      }
      if (op == "maximum") {
        da = x > y ? 1 : (x == y ? .5 : 0);
        db = 1 - da;
      }
      if (op == "minimum") {
        da = x < y ? 1 : (x == y ? .5 : 0);
        db = 1 - da;
      }
      accumulate(a, ai, g[i] * da);
      accumulate(b, bi, g[i] * db);
    }
  });
}
inline Tensor unary(Tensor a, const std::string& op, double arg = 0) {
  auto r = make(a->shape, a->dtype());
  std::vector<double> derivative(a->n());
  for (size_t i = 0; i < a->n(); ++i) {
    double x = a->get(i), y = 0, d = 0;
    if (op == "neg") {
      y = -x;
      d = -1;
    } else if (op == "square") {
      y = x * x;
      d = 2 * x;
    } else if (op == "pow") {
      y = std::pow(x, arg);
      d = arg * std::pow(x, arg - 1);
    } else if (op == "abs") {
      y = std::abs(x);
      d = (x > 0) - (x < 0);
    } else if (op == "exp") {
      y = std::exp(x);
      d = y;
    } else if (op == "log") {
      y = std::log(x);
      d = 1 / x;
    } else if (op == "sqrt") {
      y = std::sqrt(x);
      d = .5 / y;
    } else if (op == "sin") {
      y = std::sin(x);
      d = std::cos(x);
    } else if (op == "cos") {
      y = std::cos(x);
      d = -std::sin(x);
    } else if (op == "relu") {
      y = std::max(0., x);
      d = x > 0;
    } else if (op == "sigmoid") {
      y = 1 / (1 + std::exp(-x));
      d = y * (1 - y);
    } else if (op == "tanh") {
      y = std::tanh(x);
      d = 1 - y * y;
    } else if (op == "leaky_relu") {
      y = x >= 0 ? x : x * arg;
      d = x >= 0 ? 1 : arg;
    } else if (op == "gelu") {
      y = x * .5 * (1 + std::erf(x / std::sqrt(2.)));
      d = .5 * (1 + std::erf(x / std::sqrt(2.))) + x * std::exp(-x * x / 2) / std::sqrt(2 * 3.141592653589793);
    } else if (op == "floor")
      y = std::floor(x);
    else if (op == "round")
      y = std::nearbyint(x);
    else if (op == "sign")
      y = (x > 0) - (x < 0);
    else
      require(false, "unknown unary op");
    r->set(i, y);
    if (a->dtype() == "F32") {
      float fy = r->get(i);
      if (op == "tanh")
        d = float(1.0f - float(fy * fy));
      else if (op == "sigmoid")
        d = float(fy * float(1.0f - fy));
      else
        d = float(d);
    }
    derivative[i] = d;
  }
  return graph(r, {a}, [a, derivative](const auto& g) {
    for (size_t i = 0; i < g.size(); ++i) accumulate(a, i, g[i] * derivative[i]);
  });
}
inline Tensor mapped(Tensor a, Shape shape, std::vector<size_t> map, bool share = false) {
  auto r = make(shape, a->dtype());
  if (share) {
    r->storage = a->storage;
    r->offsets.resize(map.size());
    for (size_t i = 0; i < map.size(); ++i) r->offsets[i] = a->index(map[i]);
  } else
    for (size_t i = 0; i < map.size(); ++i) {
      if (a->dtype() == "I64")
        r->set_int(i, a->integer(map[i]));
      else
        r->set(i, a->get(map[i]));
    }
  return graph(r, {a}, [a, map](const auto& g) {
    for (size_t i = 0; i < g.size(); ++i) accumulate(a, map[i], g[i]);
  });
}
inline Tensor reshape(Tensor a, Shape s) {
  int unknown = -1;
  size_t n = 1;
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] == -1) {
      require(unknown < 0, "multiple inferred dimensions");
      unknown = i;
    } else {
      require(s[i] >= 0, "invalid dimension");
      n *= s[i];
    }
  }
  if (unknown >= 0) {
    require(n && a->n() % n == 0, "invalid inferred reshape");
    s[unknown] = a->n() / n;
  }
  require(elements(s) == a->n(), "reshape element count differs");
  std::vector<size_t> map(a->n());
  std::iota(map.begin(), map.end(), 0);
  return mapped(a, s, map, true);
}
inline std::vector<int64_t> coordinates(size_t i, const Shape& s) {
  Shape c(s.size());
  for (int j = int(s.size()) - 1; j >= 0; --j) {
    c[j] = i % s[j];
    i /= s[j];
  }
  return c;
}
inline size_t flat(const Shape& c, const Shape& s) {
  size_t i = 0;
  for (size_t j = 0; j < s.size(); ++j) i = i * s[j] + c[j];
  return i;
}
inline Tensor transpose(Tensor a, int x, int y) {
  x = axis(x, a->shape.size());
  y = axis(y, a->shape.size());
  Shape s = a->shape;
  std::swap(s[x], s[y]);
  std::vector<size_t> map(elements(s));
  for (size_t i = 0; i < map.size(); ++i) {
    auto c = coordinates(i, s);
    std::swap(c[x], c[y]);
    map[i] = flat(c, a->shape);
  }
  return mapped(a, s, map, true);
}
inline Tensor select(Tensor a, int d, const std::vector<int64_t>& ids, bool remove = false) {
  d = axis(d, a->shape.size());
  Shape s = a->shape;
  s[d] = ids.size();
  std::vector<size_t> map(elements(s));
  for (size_t i = 0; i < map.size(); ++i) {
    auto c = coordinates(i, s);
    auto v = ids[c[d]];
    if (v < 0) v += a->shape[d];
    require(v >= 0 && v < a->shape[d], "index out of range");
    c[d] = v;
    map[i] = flat(c, a->shape);
  }
  if (remove) s.erase(s.begin() + d);
  return mapped(a, s, map, true);
}
inline Tensor reduce(Tensor a, int dim, bool keep, const std::string& op) {
  bool all = dim == 999;
  int d = all ? 0 : axis(dim, a->shape.size());
  Shape s = all ? Shape{} : a->shape;
  if (!all) {
    if (keep)
      s[d] = 1;
    else
      s.erase(s.begin() + d);
  }
  auto dt = (op == "argmax" || op == "argmin")              ? "I64"
            : (a->dtype() == "BOOL" || a->dtype() == "I64") ? "I64"
                                                            : a->dtype();
  auto r = make(s, dt);
  std::vector<size_t> count(r->n()), winner(r->n());
  std::vector<size_t> mapping(a->n());
  std::vector<double> sums(r->n()), squares(r->n());
  for (size_t i = 0; i < a->n(); ++i) {
    auto c = coordinates(i, a->shape);
    int64_t coord = all ? i : c[d];
    if (!all) {
      if (keep)
        c[d] = 0;
      else
        c.erase(c.begin() + d);
    }
    auto j = all ? 0 : flat(c, s);
    mapping[i] = j;
    double x = a->get(i);
    if (!count[j] || ((op == "max" || op == "argmax") && x > a->get(winner[j])) ||
        ((op == "min" || op == "argmin") && x < a->get(winner[j])))
      winner[j] = i;
    sums[j] += x;
    squares[j] += x * x;
    ++count[j];
    if (op == "argmax" || op == "argmin")
      r->set_int(j, all ? winner[j] : coordinates(winner[j], a->shape)[d]);
    else if (op == "max" || op == "min")
      r->set(j, a->get(winner[j]));
    (void)coord;
  }
  for (size_t j = 0; j < r->n(); ++j) {
    if (op == "sum") r->set(j, sums[j]);
    if (op == "mean") r->set(j, count[j] ? sums[j] / count[j] : NAN);
    if (op == "var" || op == "std") {
      double v = count[j] > 1 ? (squares[j] - sums[j] * sums[j] / count[j]) / (count[j] - 1) : NAN;
      r->set(j, op == "std" ? std::sqrt(std::max(0., v)) : v);
    }
  }
  auto output = r->storage;
  return graph(r, {a}, [a, output, op, mapping, count, winner, sums](const auto& g) {
    for (size_t i = 0; i < a->n(); ++i) {
      auto j = mapping[i];
      double z = 0;
      if (op == "sum")
        z = 1;
      else if (op == "mean")
        z = 1. / count[j];
      else if (op == "max" || op == "min")
        z = i == winner[j];
      else if (op == "var")
        z = 2 * (a->get(i) - sums[j] / count[j]) / (count[j] - 1);
      else if (op == "std")
        z = (a->get(i) - sums[j] / count[j]) / ((count[j] - 1) * output->get(j));
      accumulate(a, i, g[j] * z);
    }
  });
}
inline Tensor matmul(Tensor a, Tensor b) {
  require(a->shape.size() == 2 && b->shape.size() == 2 && a->shape[1] == b->shape[0],
          "matmul requires compatible rank-2 matrices");
  auto m = a->shape[0], k = a->shape[1], n = b->shape[1];
  auto r = make({m, n}, promote(a, b));
  for (int64_t i = 0; i < m; ++i)
    for (int64_t j = 0; j < n; ++j) {
      double z = 0;
      for (int64_t q = 0; q < k; ++q) {
        double p = a->get(i * k + q) * b->get(q * n + j);
        z = r->dtype() == "F32" ? double(float(float(z) + float(p))) : z + p;
      }
      r->set(i * n + j, z);
    }
  return graph(r, {a, b}, [a, b, m, k, n](const auto& g) {
    for (int64_t i = 0; i < m; ++i)
      for (int64_t q = 0; q < k; ++q) {
        double z = 0;
        for (int64_t j = 0; j < n; ++j) z += g[i * n + j] * b->get(q * n + j);
        accumulate(a, i * k + q, z);
      }
    for (int64_t q = 0; q < k; ++q)
      for (int64_t j = 0; j < n; ++j) {
        double z = 0;
        for (int64_t i = 0; i < m; ++i) z += a->get(i * k + q) * g[i * n + j];
        accumulate(b, q * n + j, z);
      }
  });
}
inline Tensor softmax(Tensor a, int dim, bool log = false) {
  dim = axis(dim, a->shape.size());
  size_t inner = 1;
  for (size_t j = dim + 1; j < a->shape.size(); ++j) inner *= a->shape[j];
  size_t width = a->shape[dim], outer = width ? a->n() / width / inner : 0;
  auto r = make(a->shape, a->dtype());
  for (size_t o = 0; o < outer; ++o)
    for (size_t i = 0; i < inner; ++i) {
      double mx = -INFINITY, total = 0;
      for (size_t j = 0; j < width; ++j) mx = std::max(mx, a->get((o * width + j) * inner + i));
      for (size_t j = 0; j < width; ++j) total += std::exp(a->get((o * width + j) * inner + i) - mx);
      for (size_t j = 0; j < width; ++j) {
        size_t p = (o * width + j) * inner + i;
        double z = a->get(p) - mx - std::log(total);
        r->set(p, log ? z : std::exp(z));
      }
    }
  auto output = r->storage;
  return graph(r, {a}, [a, output, inner, width, outer, log](const auto& g) {
    for (size_t o = 0; o < outer; ++o)
      for (size_t i = 0; i < inner; ++i) {
        double sum = 0;
        for (size_t j = 0; j < width; ++j) {
          size_t p = (o * width + j) * inner + i;
          sum += log ? g[p] : g[p] * output->get(p);
        }
        for (size_t j = 0; j < width; ++j) {
          size_t p = (o * width + j) * inner + i;
          accumulate(a, p, log ? g[p] - std::exp(output->get(p)) * sum : output->get(p) * (g[p] - sum));
        }
      }
  });
}
inline Tensor concat(const std::vector<Tensor>& v, int d, bool stack) {
  require(!v.empty(), "empty tensor list");
  std::vector<Tensor> xs = v;
  if (stack) {
    for (auto& t : xs) {
      auto s = t->shape;
      int ax = d < 0 ? d + int(s.size()) + 1 : d;
      require(ax >= 0 && ax <= int(s.size()), "stack dimension");
      s.insert(s.begin() + ax, 1);
      t = reshape(t, s);
    }
  }
  d = axis(d, xs[0]->shape.size());
  Shape s = xs[0]->shape;
  s[d] = 0;
  for (auto t : xs) {
    require(t->shape.size() == s.size(), "cat rank mismatch");
    for (size_t j = 0; j < s.size(); ++j)
      if (int(j) != d) require(t->shape[j] == s[j], "cat shape mismatch");
    s[d] += t->shape[d];
  }
  auto r = make(s, xs[0]->dtype());
  std::vector<std::pair<size_t, size_t>> map(r->n());
  int64_t offset = 0;
  for (size_t t = 0; t < xs.size(); ++t) {
    for (size_t i = 0; i < xs[t]->n(); ++i) {
      auto c = coordinates(i, xs[t]->shape);
      c[d] += offset;
      auto j = flat(c, s);
      map[j] = {t, i};
      if (r->dtype() == "I64")
        r->set_int(j, xs[t]->integer(i));
      else
        r->set(j, xs[t]->get(i));
    }
    offset += xs[t]->shape[d];
  }
  return graph(r, xs, [xs, map](const auto& g) {
    for (size_t i = 0; i < g.size(); ++i) accumulate(xs[map[i].first], map[i].second, g[i]);
  });
}
inline Tensor cross_entropy(Tensor x, Tensor y) {
  require(x->shape.size() == 2, "cross_entropy needs rank 2");
  auto n = x->shape[0], c = x->shape[1];
  require(y->n() == size_t(n) || y->shape == x->shape, "cross_entropy target shape");
  auto lp = softmax(x, 1, true);
  auto r = scalar(0, x->dtype());
  double z = 0;
  for (int64_t i = 0; i < n; ++i)
    for (int64_t j = 0; j < c; ++j) {
      double target = y->shape == x->shape ? y->get(i * c + j) : double(y->integer(i) == j);
      z -= target * lp->get(i * c + j) / n;
    }
  r->set(0, z);
  return graph(r, {lp}, [lp, y, n, c](const auto& g) {
    for (int64_t i = 0; i < n; ++i)
      for (int64_t j = 0; j < c; ++j) {
        double target = y->shape == lp->shape ? y->get(i * c + j) : double(y->integer(i) == j);
        accumulate(lp, i * c + j, -g[0] * target / n);
      }
  });
}
inline Tensor conv2d(Tensor x, Tensor w, Tensor b, int stride, int pad) {
  require(x->shape.size() == 4 && w->shape.size() == 4, "conv2d expects NCHW");
  auto n = x->shape[0], ic = x->shape[1], h = x->shape[2], width = x->shape[3], oc = w->shape[0], k = w->shape[2];
  require(w->shape[1] == ic && w->shape[3] == k && stride > 0, "conv2d shapes");
  auto oh = (h + 2 * pad - k) / stride + 1, ow = (width + 2 * pad - k) / stride + 1;
  require(oh > 0 && ow > 0, "kernel exceeds input");
  auto r = make({n, oc, oh, ow}, x->dtype());
  auto each = [=](auto fn) {
    for (int64_t a = 0; a < n; ++a)
      for (int64_t c = 0; c < oc; ++c)
        for (int64_t i = 0; i < oh; ++i)
          for (int64_t j = 0; j < ow; ++j) {
            size_t ri = ((a * oc + c) * oh + i) * ow + j;
            for (int64_t q = 0; q < ic; ++q)
              for (int64_t u = 0; u < k; ++u)
                for (int64_t v = 0; v < k; ++v) {
                  int64_t yy = i * stride + u - pad, xx = j * stride + v - pad;
                  if (yy >= 0 && yy < h && xx >= 0 && xx < width)
                    fn(ri, ((a * ic + q) * h + yy) * width + xx, ((c * ic + q) * k + u) * k + v, c);
                }
          }
  };
  for (size_t i = 0; i < r->n(); ++i) r->set(i, b->get((i / (oh * ow)) % oc));
  each([&](size_t ri, size_t xi, size_t wi, int64_t) { r->set(ri, r->get(ri) + x->get(xi) * w->get(wi)); });
  return graph(r, {x, w, b}, [=](const auto& g) {
    each([&](size_t ri, size_t xi, size_t wi, int64_t) {
      accumulate(x, xi, g[ri] * w->get(wi));
      accumulate(w, wi, g[ri] * x->get(xi));
    });
    for (size_t i = 0; i < g.size(); ++i) accumulate(b, (i / (oh * ow)) % oc, g[i]);
  });
}
inline Tensor maxpool(Tensor x, int k) {
  require(x->shape.size() == 4 && k > 0, "max_pool2d needs NCHW and positive kernel");
  auto n = x->shape[0], c = x->shape[1], h = x->shape[2], w = x->shape[3], oh = h / k, ow = w / k;
  auto r = make({n, c, oh, ow}, x->dtype());
  std::vector<size_t> map(r->n());
  for (size_t i = 0; i < r->n(); ++i) {
    auto coords = coordinates(i, r->shape);
    double mx = -INFINITY;
    for (int u = 0; u < k; ++u)
      for (int v = 0; v < k; ++v) {
        size_t j = ((coords[0] * c + coords[1]) * h + coords[2] * k + u) * w + coords[3] * k + v;
        if (x->get(j) > mx) {
          mx = x->get(j);
          map[i] = j;
        }
      }
    r->set(i, mx);
  }
  return graph(r, {x}, [x, map](const auto& g) {
    for (size_t i = 0; i < g.size(); ++i) accumulate(x, map[i], g[i]);
  });
}
}  // namespace lmtorch_lite
