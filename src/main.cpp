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

  // ---- Test transpose ----
  Tensor t2({2, 3});
  int val = 0;
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++) {
      t2.set(i, j, (float)val++);
    }
  }
  std::cout << "t2 (before transpose), shape " << t2.shape[0] << "x"
            << t2.shape[1] << " = " << std::endl;
  for (int i = 0; i < t2.shape[0]; i++) {
    for (int j = 0; j < t2.shape[1]; j++) {
      std::cout << t2.get(i, j) << " ";
    }
    std::cout << std::endl;
  }

  Tensor t2_transposed = transpose(t2);
  std::cout << "t2 transposed, shape " << t2_transposed.shape[0] << "x"
            << t2_transposed.shape[1] << " = " << std::endl;
  for (int i = 0; i < t2_transposed.shape[0]; i++) {
    for (int j = 0; j < t2_transposed.shape[1]; j++) {
      std::cout << t2_transposed.get(i, j) << " ";
    }
    std::cout << std::endl;
  }

  Tensor t3({2, 6});
  val = 0;
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 6; j++) {
      t3.set(i, j, (float)val++);
    }
  }

  reshape(t3, {3, 4});
  std::cout << "t3 reshaped to " << t3.shape[0] << "x" << t3.shape[1] << " = "
            << std::endl;
  for (int i = 0; i < t3.shape[0]; i++) {
    for (int j = 0; j < t3.shape[1]; j++) {
      std::cout << t3.get(i, j) << " ";
    }
    std::cout << std::endl;
  }

  try {
    reshape(t3, {3, 5});
    std::cout << "ERROR: reshape should have thrown but didn't" << std::endl;
  } catch (const std::invalid_argument &e) {
    std::cout << "Correctly threw: " << e.what() << std::endl;
  }

  return 0;
}