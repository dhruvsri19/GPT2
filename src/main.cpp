#include "gpt2/ops.hpp"
#include "gpt2/tensor.hpp"
#include <iostream>
using namespace std;
int main() {
  Tensor t({2, 2});
  t.set(0, 0, 1.0f);
  t.set(0, 1, 2.0f);
  t.set(1, 0, 3.0f);
  t.set(1, 1, 4.0f);

  Tensor res = matmul(t, t);
  std::cout << "t x t = " << std::endl;
  std::cout << res.get(0, 0) << " " << res.get(0, 1) << std::endl;
  std::cout << res.get(1, 0) << " " << res.get(1, 1) << std::endl;

  Tensor result = softmax(t);
  std::cout << "softmax of t = " << std::endl;
  std::cout << result.get(0, 0) << " " << result.get(0, 1) << std::endl;
  std::cout << result.get(1, 0) << " " << result.get(1, 1) << std::endl;

  return 0;
}