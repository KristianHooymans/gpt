#pragma once



#include <cstddef>
#include <vector>


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
