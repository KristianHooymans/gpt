#include "../include/matrix.hpp"

#include <iostream>




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
