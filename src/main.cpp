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

  Tensor gamma({1, 2});
  gamma.set(0, 0, 1.0f);
  gamma.set(0, 1, 1.0f);
  Tensor beta({1, 2});
  beta.set(0, 0, 0.0f);
  beta.set(0, 1, 0.0f);
  Tensor layer_normalized_t = layernorm(t, gamma, beta);
  std::cout << "layer_normalized_t = " << std::endl;
  std::cout << layer_normalized_t.get(0, 0) << " "
            << layer_normalized_t.get(0, 1) << std::endl;
  std::cout << layer_normalized_t.get(1, 0) << " "
            << layer_normalized_t.get(1, 1) << std::endl;

  Tensor t1({1, 3});
  t1.set(0, 0, -1.0f);
  t1.set(0, 1, 0.0f);
  t1.set(0, 2, 3.0f);
  Tensor gelu_result = gelu(t1);
  std::cout << "gelu of t1 = " << std::endl;
  std::cout << gelu_result.get(0, 0) << " " << gelu_result.get(0, 1) << " "
            << gelu_result.get(0, 2) << std::endl;
  return 0;
}