#include <cassert>
#include <iostream>

#include "tensor.hpp"
using namespace lmtorch_lite;
int main() {
  auto x = make({2, 3}, "F64");
  for (size_t i = 0; i < x->n(); ++i) x->set(i, i + 1);
  x->requires = true;
  auto b = make({3}, "F64", 2);
  b->requires = true;
  auto loss = reduce(unary(binary(x, b, "mul"), "square"), 999, false, "mean");
  backward(loss);
  assert(std::abs(loss->get(0) - 60.666666666666664) < 1e-12);
  for (size_t i = 0; i < x->n(); ++i) assert(std::abs(x->grad[i] - 4 * (i + 1) / 3.) < 1e-12);
  for (size_t i = 0; i < 3; ++i)
    assert(std::abs(b->grad[i] - 2. * 2 * ((i + 1) * (i + 1) + (i + 4) * (i + 4)) / 6) < 1e-12);
  auto w = make({3, 2}, "F64", .5);
  w->requires = true;
  auto y = matmul(x, w);
  assert(y->shape == Shape({2, 2}));
  auto view = select(x, 1, {0, 2});
  assert(view->get(2) == 4);
  view->set(2, 10);
  assert(x->get(3) == 10);
  auto f32 = scalar(16777217.0);
  assert(f32->get(0) == 16777216.0);
  auto i64 = make({1}, "I64");
  i64->set_int(0, INT64_C(9007199254740993));
  assert(i64->integer(0) == INT64_C(9007199254740993));
  bool bad = false;
  try {
    make({INT64_MAX, 2});
  } catch (const std::exception&) {
    bad = true;
  }
  assert(bad);
  // Graph must release its own result: captured storage may survive, captured result nodes may not.
  std::weak_ptr<Node> old;
  {
    auto a = make({2}, "F64", 1);
    a->requires = true;
    auto z = softmax(a, 0);
    old = z;
    backward(reduce(z, 999, false, "sum"));
  }
  assert(old.expired());
  auto changed = make({1}, "F64", 2); changed->requires = true;
  auto stale = unary(changed, "square"); changed->set(0, 3);
  bool mutation_rejected = false;
  try { backward(stale); } catch (const std::exception&) { mutation_rejected = true; }
  assert(mutation_rejected);
  std::cout << "core: broadcast gradients, F64, non-square matmul, view mutation, F32 rounding, exact I64, bounds and "
               "graph lifetime passed\n";
}
