#pragma once



#include <cstddef>
#include <vector>
#include <functional>


struct Matrix {
  size_t rows;
  size_t cols;
  std::vector<double> data;

  Matrix(size_t rows, size_t cols);

  double& operator()(size_t r, size_t c);
  const double& operator()(size_t r, size_t c) const;
};


Matrix add(const Matrix& a, const Matrix& b);

Matrix transpose(const Matrix& a);

Matrix matmul(const Matrix& a, const Matrix& b);

Matrix scalarMult(double scalar, const Matrix& a);

Matrix subtract(const Matrix& a, const Matrix& b);

Matrix hadamard(const Matrix& a, const Matrix& b);

double sum(const Matrix& a);

Matrix apply(const Matrix& a, std::function<double(double)>);

Matrix sumRows(const Matrix& a);

Matrix softmaxRows(const Matrix& a);
