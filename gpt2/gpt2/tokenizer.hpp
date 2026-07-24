#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#pragma once

class tokenizer {
private:
  std::unordered_map<std::string, int> token_to_int;
  std::unordered_map<int, std::string> int_to_token;

public:
  int get_id(std::string word) {
    auto it = token_to_int.find(word);
    if (it == token_to_int.end()) {
      throw std::runtime_error("Word Not Found");
    }
    return it->second;
  }

  int add_token(std::string word) {
    auto it = token_to_int.find(word);
    if (it != token_to_int.end()) {
      return it->second;
    }
    int new_id = token_to_int.size();
    token_to_int[word] = new_id;
    int_to_token[new_id] = word;
    return new_id;
  }
};