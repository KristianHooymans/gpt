
#include "../include/matrix.hpp"

#include <cassert>
#include <cmath>
#include <iostream>

bool approxEqual(double a, double b, double eps = 1e-9) {
    return std::abs(a - b) < eps;
}

void testAdd() {
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

    Matrix c = add(a, b);

    assert(approxEqual(c(0, 0), 6));
    assert(approxEqual(c(0, 1), 8));
    assert(approxEqual(c(1, 0), 10));
    assert(approxEqual(c(1, 1), 12));

    std::cout << "testAdd passed\n";
}

void testTranspose() {
    Matrix a(2, 3);

    a(0, 0) = 1;
    a(0, 1) = 2;
    a(0, 2) = 3;
    a(1, 0) = 4;
    a(1, 1) = 5;
    a(1, 2) = 6;

    Matrix b = transpose(a);

    assert(b.rows == 3);
    assert(b.cols == 2);

    assert(approxEqual(b(0, 0), 1));
    assert(approxEqual(b(0, 1), 4));

    assert(approxEqual(b(1, 0), 2));
    assert(approxEqual(b(1, 1), 5));

    assert(approxEqual(b(2, 0), 3));
    assert(approxEqual(b(2, 1), 6));

    std::cout << "testTranspose passed\n";
}

void testMatmul() {
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

    Matrix c = matmul(a, b);

    // [1 2] [5 6] = [19 22]
    // [3 4] [7 8]   [43 50]

    assert(approxEqual(c(0, 0), 19));
    assert(approxEqual(c(0, 1), 22));
    assert(approxEqual(c(1, 0), 43));
    assert(approxEqual(c(1, 1), 50));

    std::cout << "testMatmul passed\n";
}

void testScalarMult() {
    Matrix a(2, 2);

    a(0, 0) = 1;
    a(0, 1) = -2;
    a(1, 0) = 3;
    a(1, 1) = 4;

    Matrix b = scalarMult(2.0, a);

    assert(approxEqual(b(0, 0), 2));
    assert(approxEqual(b(0, 1), -4));
    assert(approxEqual(b(1, 0), 6));
    assert(approxEqual(b(1, 1), 8));

    std::cout << "testScalarMult passed\n";
}

void testApplySquare() {
    Matrix a(2, 2);

    a(0, 0) = 1;
    a(0, 1) = -2;
    a(1, 0) = 3;
    a(1, 1) = -4;

    Matrix b = apply(a, [](double x) {
        return x * x;
    });

    assert(approxEqual(b(0, 0), 1));
    assert(approxEqual(b(0, 1), 4));
    assert(approxEqual(b(1, 0), 9));
    assert(approxEqual(b(1, 1), 16));

    std::cout << "testApplySquare passed\n";
}

void testApplyExp() {
    Matrix a(2, 2);

    a(0, 0) = 0;
    a(0, 1) = 1;
    a(1, 0) = 2;
    a(1, 1) = -1;

    Matrix b = apply(a, [](double x) {
        return std::exp(x);
    });

    assert(approxEqual(b(0, 0), std::exp(0.0)));
    assert(approxEqual(b(0, 1), std::exp(1.0)));
    assert(approxEqual(b(1, 0), std::exp(2.0)));
    assert(approxEqual(b(1, 1), std::exp(-1.0)));

    std::cout << "testApplyExp passed\n";
}

void testApplyReLU() {
    Matrix a(2, 3);

    a(0, 0) = -5;
    a(0, 1) = 0;
    a(0, 2) = 3;

    a(1, 0) = -1;
    a(1, 1) = 7;
    a(1, 2) = -10;

    Matrix b = apply(a, [](double x) {
        return std::max(0.0, x);
    });

    assert(approxEqual(b(0, 0), 0));
    assert(approxEqual(b(0, 1), 0));
    assert(approxEqual(b(0, 2), 3));

    assert(approxEqual(b(1, 0), 0));
    assert(approxEqual(b(1, 1), 7));
    assert(approxEqual(b(1, 2), 0));

    std::cout << "testApplyReLU passed\n";
}

int main() {
    testAdd();
    testTranspose();
    testMatmul();
    testScalarMult();

    testApplySquare();
    testApplyExp();
    testApplyReLU();

    std::cout << "\nAll tests passed!\n";
}
