#include "tensor.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

Tensor matmul(Tensor &a, Tensor &b) {
  if (a.shape.size() < 2 || b.shape.size() < 2) {
    throw std::invalid_argument("matmul expects at least 2D tensors");
  }
  if (a.shape[1] != b.shape[0]) {
    throw std::invalid_argument("Shapes Are Mismatched");
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

Tensor softmax(Tensor &a) {
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

Tensor layernorm(Tensor &t, Tensor &gamma, Tensor &beta) {
  Tensor result = Tensor(t.shape);
  for (int i = 0; i < t.shape[0]; i++) {
    float mu = 0;
    float var = 0;
    float normalized = 0;
    float output = 0;
    float epsilon = 1e-5;
    for (int j = 0; j < t.shape[1]; j++) {
      mu += t.get(i, j);
    }
    mu /= t.shape[1];

    for (int j = 0; j < t.shape[1]; j++) {
      var += (t.get(i, j) - mu) * (t.get(i, j) - mu);
    }
    var /= t.shape[1];

    for (int j = 0; j < t.shape[1]; j++) {
      normalized = (t.get(i, j) - mu) / sqrt(var + epsilon);
      output = normalized * gamma.get(0, j) + beta.get(0, j);
      result.set(i, j, output);
    }
  }
  return result;
}

Tensor gelu(Tensor &a) {
  Tensor result = Tensor(a.shape);
  for (int i = 0; i < a.shape[0]; i++) {
    for (int j = 0; j < a.shape[1]; j++) {
      float x = a.get(i, j);
      result.set(i, j,
                 0.5 * x *
                     (1 + std::tanh(std::sqrt(2.0 / M_PI) *
                                    (x + 0.044715 * std::pow(x, 3)))));
    }
  }
  return result;
}

Tensor transpose(Tensor &a) {
  Tensor result({a.shape[1], a.shape[0]});
  for (int i = 0; i < a.shape[0]; i++) {
    for (int j = 0; j < a.shape[1]; j++) {
      result.set(j, i, a.get(i, j));
    }
  }
  return result;
}

void reshape(Tensor &a, std::vector<int> new_shape) {
  int new_total = 1;
  for (int i = 0; i < new_shape.size(); i++) {
    new_total *= new_shape[i];
  }
  if (new_total != a.data.size()) {
    throw std::invalid_argument("Reshape Failed");
  }
  a.shape = new_shape;
}