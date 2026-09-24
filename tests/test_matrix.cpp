#include "../include/matrix.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

bool approxEqual(double a, double b, double eps = 1e-9) {
    return std::abs(a - b) < eps;
}

void testConstructor() {
    Matrix a(2, 3);

    assert(a.rows == 2);
    assert(a.cols == 3);

    for (size_t i = 0; i < a.rows; ++i) {
        for (size_t j = 0; j < a.cols; ++j) {
            assert(approxEqual(a(i, j), 0.0));
        }
    }

    std::cout << "testConstructor passed\n";
}

void testIndexing() {
    Matrix a(2, 2);

    a(0, 0) = 1.5;
    a(0, 1) = -2.0;
    a(1, 0) = 3.25;
    a(1, 1) = 4.0;

    assert(approxEqual(a(0, 0), 1.5));
    assert(approxEqual(a(0, 1), -2.0));
    assert(approxEqual(a(1, 0), 3.25));
    assert(approxEqual(a(1, 1), 4.0));

    std::cout << "testIndexing passed\n";
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

void testAddDimensionMismatch() {
    Matrix a(2, 2);
    Matrix b(2, 3);

    bool threw = false;

    try {
        add(a, b);
    } catch (const std::runtime_error&) {
        threw = true;
    }

    assert(threw);

    std::cout << "testAddDimensionMismatch passed\n";
}

void testSubtract() {
    Matrix a(2, 2);
    Matrix b(2, 2);

    a(0, 0) = 5;
    a(0, 1) = 7;
    a(1, 0) = 9;
    a(1, 1) = 11;

    b(0, 0) = 1;
    b(0, 1) = 2;
    b(1, 0) = 3;
    b(1, 1) = 4;

    Matrix c = subtract(a, b);

    assert(approxEqual(c(0, 0), 4));
    assert(approxEqual(c(0, 1), 5));
    assert(approxEqual(c(1, 0), 6));
    assert(approxEqual(c(1, 1), 7));

    std::cout << "testSubtract passed\n";
}

void testSubtractDimensionMismatch() {
    Matrix a(2, 2);
    Matrix b(3, 2);

    bool threw = false;

    try {
        subtract(a, b);
    } catch (const std::runtime_error&) {
        threw = true;
    }

    assert(threw);

    std::cout << "testSubtractDimensionMismatch passed\n";
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
    Matrix a(2, 3);
    Matrix b(3, 2);

    a(0, 0) = 1;
    a(0, 1) = 2;
    a(0, 2) = 3;

    a(1, 0) = 4;
    a(1, 1) = 5;
    a(1, 2) = 6;

    b(0, 0) = 7;
    b(0, 1) = 8;

    b(1, 0) = 9;
    b(1, 1) = 10;

    b(2, 0) = 11;
    b(2, 1) = 12;

    Matrix c = matmul(a, b);

    assert(c.rows == 2);
    assert(c.cols == 2);

    /*
        [1 2 3]   [7  8 ]   [58  64 ]
        [4 5 6] * [9  10] = [139 154]
                  [11 12]
    */

    assert(approxEqual(c(0, 0), 58));
    assert(approxEqual(c(0, 1), 64));
    assert(approxEqual(c(1, 0), 139));
    assert(approxEqual(c(1, 1), 154));

    std::cout << "testMatmul passed\n";
}

void testMatmulDimensionMismatch() {
    Matrix a(2, 3);
    Matrix b(2, 2);

    bool threw = false;

    try {
        matmul(a, b);
    } catch (const std::runtime_error&) {
        threw = true;
    }

    assert(threw);

    std::cout << "testMatmulDimensionMismatch passed\n";
}

void testScalarMult() {
    Matrix a(2, 2);

    a(0, 0) = 1;
    a(0, 1) = -2;
    a(1, 0) = 3;
    a(1, 1) = -4;

    Matrix b = scalarMult(2.5, a);

    assert(approxEqual(b(0, 0), 2.5));
    assert(approxEqual(b(0, 1), -5.0));
    assert(approxEqual(b(1, 0), 7.5));
    assert(approxEqual(b(1, 1), -10.0));

    std::cout << "testScalarMult passed\n";
}

void testHadamard() {
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

    Matrix c = hadamard(a, b);

    assert(approxEqual(c(0, 0), 5));
    assert(approxEqual(c(0, 1), 12));
    assert(approxEqual(c(1, 0), 21));
    assert(approxEqual(c(1, 1), 32));

    std::cout << "testHadamard passed\n";
}

void testHadamardRowMismatch() {
    Matrix a(2, 2);
    Matrix b(3, 2);

    bool threw = false;

    try {
        hadamard(a, b);
    } catch (const std::runtime_error&) {
        threw = true;
    }

    assert(threw);

    std::cout << "testHadamardRowMismatch passed\n";
}

void testHadamardColumnMismatch() {
    Matrix a(2, 2);
    Matrix b(2, 3);

    bool threw = false;

    try {
        hadamard(a, b);
    } catch (const std::runtime_error&) {
        threw = true;
    }

    assert(threw);

    std::cout << "testHadamardColumnMismatch passed\n";
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
    a(1, 0) = -1;
    a(1, 1) = 2;

    Matrix b = apply(a, [](double x) {
        return std::exp(x);
    });

    assert(approxEqual(b(0, 0), std::exp(0.0)));
    assert(approxEqual(b(0, 1), std::exp(1.0)));
    assert(approxEqual(b(1, 0), std::exp(-1.0)));
    assert(approxEqual(b(1, 1), std::exp(2.0)));

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

void testSum() {
    Matrix a(2, 3);

    a(0, 0) = 1;
    a(0, 1) = 2;
    a(0, 2) = 3;

    a(1, 0) = 4;
    a(1, 1) = 5;
    a(1, 2) = 6;

    double result = sum(a);

    assert(approxEqual(result, 21));

    std::cout << "testSum passed\n";
}

void testSumRows() {
    Matrix a(3, 3);

    a(0, 0) = 1;
    a(0, 1) = 2;
    a(0, 2) = 3;

    a(1, 0) = 4;
    a(1, 1) = 5;
    a(1, 2) = 6;

    a(2, 0) = -1;
    a(2, 1) = -2;
    a(2, 2) = -3;

    Matrix result = sumRows(a);

    assert(result.rows == 3);
    assert(result.cols == 1);

    assert(approxEqual(result(0, 0), 6));
    assert(approxEqual(result(1, 0), 15));
    assert(approxEqual(result(2, 0), -6));

    std::cout << "testSumRows passed\n";
}

void testTransposeTwice() {
    Matrix a(2, 3);

    a(0, 0) = 1;
    a(0, 1) = 2;
    a(0, 2) = 3;

    a(1, 0) = 4;
    a(1, 1) = 5;
    a(1, 2) = 6;

    Matrix b = transpose(transpose(a));

    assert(b.rows == a.rows);
    assert(b.cols == a.cols);

    for (size_t i = 0; i < a.rows; ++i) {
        for (size_t j = 0; j < a.cols; ++j) {
            assert(approxEqual(a(i, j), b(i, j)));
        }
    }

    std::cout << "testTransposeTwice passed\n";
}

void testMatmulIdentity() {
    Matrix a(2, 2);

    a(0, 0) = 3;
    a(0, 1) = 5;
    a(1, 0) = 7;
    a(1, 1) = 9;

    Matrix identity(2, 2);

    identity(0, 0) = 1;
    identity(0, 1) = 0;
    identity(1, 0) = 0;
    identity(1, 1) = 1;

    Matrix result = matmul(a, identity);

    assert(approxEqual(result(0, 0), 3));
    assert(approxEqual(result(0, 1), 5));
    assert(approxEqual(result(1, 0), 7));
    assert(approxEqual(result(1, 1), 9));

    std::cout << "testMatmulIdentity passed\n";
}

int main() {
    std::cout << "Running Matrix tests...\n\n";

    testConstructor();
    testIndexing();

    testAdd();
    testAddDimensionMismatch();

    testSubtract();
    testSubtractDimensionMismatch();

    testTranspose();
    testTransposeTwice();

    testMatmul();
    testMatmulIdentity();
    testMatmulDimensionMismatch();

    testScalarMult();

    testHadamard();
    testHadamardRowMismatch();
    testHadamardColumnMismatch();

    testApplySquare();
    testApplyExp();
    testApplyReLU();

    testSum();
    testSumRows();

    std::cout << "\nAll Matrix tests passed!\n";
}
