#include "../include/value.hpp"


#include <vector>


//below is from: https://en.wikipedia.org/wiki/Backpropagation
//input x, output y, C -> loss/cost function, L = N.O. layers, W -> weights between layers, f^l -> activation function at layer l, a_{j}^{l} = actiavtion of j-th node in layer l.
//overall g(x):= f^L(W^L f^L-1(....f^1(W^1 x)...))




Value::Value(double data)
  : data(data),
    grad(0.0),
    op(Op::None)
{}


Value Value::operator+(Value& other) {
  Value output(this->data + other.data);

  output.parents.push_back(this);
  output.parents.push_back(&other);

  output.op = Op::Add;

  return output;
}

Value Value::operator*(Value& other){
  Value output(this->data * other.data);

  output.parents.push_back(this);
  output.parents.push_back(&other);

  output.op = Op::Multiply;

  return output;
}


double gradFinder(const Value& a) {

  return 0;
}
