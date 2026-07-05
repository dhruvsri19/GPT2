#include "gpt2/tensor.hpp"
#include <iostream>
using namespace std;
int main() {
  Tensor t({2, 2});
  t.set(0, 0, 1.0f);
  t.set(0, 1, 2.0f);
  t.set(1, 0, 3.0f);
  t.set(1, 1, 4.0f);
  std::cout << t.get(0, 0) << std::endl;
  std::cout << t.get(0, 1) << std::endl;
  std::cout << t.get(1, 0) << std::endl;
  std::cout << t.get(1, 1) << std::endl;
  return 0;
}