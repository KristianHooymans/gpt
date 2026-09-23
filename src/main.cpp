#include <iostream>
#include <vector>
#include <stdexcept>




struct Matrix {
  size_t rows;
  size_t cols;
  std::vector<double> data;

  Matrix(size_t rows, size_t cols);

  double& operator()(size_t r, size_t c);
  const double& operator()(size_t r, size_t c) const;
};


//matrix constructor

Matrix::Matrix(size_t rows, size_t cols) 
    : rows(rows),
      cols(cols),
      data(rows * cols, 0.0)
{}


double& Matrix::operator()(size_t r, size_t c) {
  return data[r * cols + c]; 
}

const double& Matrix::operator()(size_t r, size_t c) const {
  return data[r * cols + c];
}

Matrix add (const Matrix& a, const Matrix& b) {
  if ((a.rows != b.rows) || (a.cols != b.cols)) {
      throw std::runtime_error(std::format("Cannot add matrices with different dimensions: Matrix a has dimension: {}x{}, and Matrix b has dimension: {}x{}", a.rows, a.cols, b.rows, b.cols));
  }
  Matrix c (a.rows, a.cols);
  for (size_t i = 0; i < a.rows; ++i) {
    for (size_t j = 0; j < a.cols; ++j) {
      c(i,j) = a(i,j) + b(i,j);
    }
  }
  return c;
}




int main() {
    Matrix a(2, 2);
    Matrix b(2, 2);

    a(0, 0) = 1;
    a(0, 1) = 2;
    a(1, 0) = 3;
    a(1, 1) = 4;

    b(0, 0) = 5;
    b(0, 1) = 6;
    b(1, 0) = 7;
    b(1, 1) = 8;
    std::cout << "testing" << '\n';

    Matrix c = add(a, b);

    for (int i = 0; i < c.rows; ++i) {
      for (int j = 0; j < c.cols; ++j) {
        std::cout << c(i,j) << " ";
      }
    }
}
