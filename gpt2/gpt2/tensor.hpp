#pragma once
#include <stdexcept>
#include <vector>
class Tensor {
public:
  std::vector<float> data;
  std::vector<int> shape;
  Tensor(std::vector<int> input_shape) {
    shape = input_shape;
    int total_size = 1;
    for (int dim : shape) {
      total_size *= dim;
    }
    data = std::vector<float>(total_size, 0.0f);
  }
  float get(int row, int col) {
    if (row < 0 || row >= shape[0] || col < 0 || col >= shape[1]) {
      throw std::out_of_range("Exceeded");
    }
    return data[row * shape[1] + col];
  }

  void set(int row, int col, float value) {
    if (row < 0 || row >= shape[0] || col < 0 || col >= shape[1]) {
      throw std::out_of_range("Exceeded");
    }
    data[row * shape[1] + col] = value;
  }
};