#include "../include/matrix.hpp"

#include <vector>
#include <format>
#include <stdexcept>



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


Matrix transpose(const Matrix& a) {
  //
  //
  //



  Matrix b(a.cols, a.rows);

  for (size_t i = 0; i < a.rows; ++i) {
    for (size_t j = 0; j < a.cols; ++j) {
      b(j,i) = a(i,j);
    }
  }
  return b;
}

Matrix matmul(const Matrix& a, const Matrix& b){
  if (a.cols != b.rows) {
    throw std::runtime_error(
        std::format("Cannot multiply matrices where first cols != second rows, a dimension: {}x{}, b dimension: {}x{}",
        a.rows, a.cols, b.rows, b.cols
        )
      );
    }
  Matrix c(a.rows, b.cols);
  for (size_t i = 0; i < a.rows; ++i) {
    for (size_t j = 0; j < b.cols; ++j) {
      double sum = 0;
      for (size_t k = 0; k < a.cols; ++k) {
        sum += a(i,k) * b(k,j);
      }

      c(i,j) = sum;
    }
  }
  return c;
}


Matrix scalarMult(double scal, const Matrix& a) {
  Matrix c(a.rows, a.cols);
  for (size_t i = 0; i < a.rows; ++i) {
    for (size_t j = 0; j < a.cols; ++j) {
      c(i,j) = scal * a(i,j);
    }
  }
  return c;
}
