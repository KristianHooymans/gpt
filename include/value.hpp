#pragma once

#include <vector>

enum class Op {
  None, 
  Add,
  Multiply
};

struct Value {
  double data;
  double grad;

  std::vector<Value*> parents;//pointer so that we can modify the actual values 
  Op op;

  Value(double data);
  
  Value operator+(Value& other);
  Value operator*(Value& other);
};
