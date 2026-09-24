#include "../include/value.hpp"


#include <vector>
#include <unordered_set>


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

//calculate the gradient at this particular node
void backward(Value& a) {
  if (a.op == Op::Add) {
    for (Value* parent : a.parents) {
      parent->grad += a.grad;
    }
  }
  else if (a.op == Op::Multiply) {
    Value* p0 = a.parents[0];
    Value* p1 = a.parents[1];
    p0->grad += a.grad * p1->data;
    p1->grad += a.grad * p0->data;
  }
}


//topological ordering to avoid double counting like: a -> b -> c,d -> b -> c,d
void buildTopologicalSort(Value& v, std::unordered_set<Value*>& visited, std::vector<Value*>&topo){
  if (visited.contains(&v)) {
    return;
  }
  visited.insert(&v);

  for(Value* parent : v.parents) {
    buildTopologicalSort(*parent, visited, topo);
  }
  topo.push_back(&v);
}


//calculate the gradient from the end to the start for all nodes.
void backProp(Value& a) {
  std::vector<Value*> topo;
  std::unordered_set<Value*> visited;

  buildTopologicalSort(a, visited, topo);

  a.grad = 1.0;

  for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
    backward(**it);
  }
}

