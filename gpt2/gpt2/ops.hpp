#include "tensor.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

Tensor matmul(const Tensor &a, const Tensor &b) {
  if (a.shape.size() < 2 || b.shape.size() < 2) {
    throw std::invalid_argument("matmul expects at least 2D tensors");
  }
  if (a.shape[1] != b.shape[0]) {
    throw std::invalid_argument("Matrices shape mismatch");
  }
  Tensor result({a.shape[0], b.shape[1]});
  for (int i = 0; i < a.shape[0]; i++) {
    for (int j = 0; j < b.shape[1]; j++) {
      float sum = 0.0f;
      for (int k = 0; k < a.shape[1]; k++) {
        sum += a.get(i, k) * b.get(k, j);
      }
      result.set(i, j, sum);
    }
  }
  return result;
}

Tensor softmax(const Tensor &a) {
  Tensor res(a.shape);
  for (int i = 0; i < a.shape[0]; i++) {
    float max_ele = a.get(i, 0);
    for (int j = 1; j < a.shape[1]; j++) {
      if (a.get(i, j) > max_ele)
        max_ele = a.get(i, j);
    }
    float sum = 0.0f;
    for (int j = 0; j < a.shape[1]; j++) {
      sum += std::exp(a.get(i, j) - max_ele);
    }
    for (int j = 0; j < a.shape[1]; j++) {
      res.set(i, j, std::exp(a.get(i, j) - max_ele) / sum);
    }
  }
  return res;
}

Tensor layernorm(const Tensor &t, const Tensor &gamma, const Tensor &beta) {
  Tensor result(t.shape);
  for (int i = 0; i < t.shape[0]; i++) {
    float mu = 0.0f;
    for (int j = 0; j < t.shape[1]; j++) {
      mu += t.get(i, j);
    }
    mu /= t.shape[1];

    float var = 0.0f;
    for (int j = 0; j < t.shape[1]; j++) {
      var += (t.get(i, j) - mu) * (t.get(i, j) - mu);
    }
    var /= t.shape[1];

    const float epsilon = 1e-5;
    for (int j = 0; j < t.shape[1]; j++) {
      float normalized = (t.get(i, j) - mu) / std::sqrt(var + epsilon);
      float output = normalized * gamma.get(0, j) + beta.get(0, j);
      result.set(i, j, output);
    }
  }
  return result;
}