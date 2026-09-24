#include "../include/value.hpp"

#include <cassert>
#include <cmath>
#include <iostream>

bool approxEqual(double a, double b, double eps = 1e-9) {
    return std::abs(a - b) < eps;
}

void testConstructor() {
    Value a(3.5);

    assert(approxEqual(a.data, 3.5));
    assert(approxEqual(a.grad, 0.0));
    assert(a.parents.empty());
    assert(a.op == Op::None);

    std::cout << "testConstructor passed\n";
}

void testAdditionForward() {
    Value a(2.0);
    Value b(3.0);

    Value c = a + b;

    assert(approxEqual(c.data, 5.0));

    assert(c.op == Op::Add);

    assert(c.parents.size() == 2);
    assert(c.parents[0] == &a);
    assert(c.parents[1] == &b);

    std::cout << "testAdditionForward passed\n";
}

void testMultiplicationForward() {
    Value a(2.0);
    Value b(3.0);

    Value c = a * b;

    assert(approxEqual(c.data, 6.0));

    assert(c.op == Op::Multiply);

    assert(c.parents.size() == 2);
    assert(c.parents[0] == &a);
    assert(c.parents[1] == &b);

    std::cout << "testMultiplicationForward passed\n";
}

void testAdditionBackward() {
    Value a(2.0);
    Value b(3.0);

    Value c = a + b;

    // pretend dL/dc = 1
    c.grad = 1.0;

    backward(c);

    // dc/da = 1
    // dc/db = 1
    assert(approxEqual(a.grad, 1.0));
    assert(approxEqual(b.grad, 1.0));

    std::cout << "testAdditionBackward passed\n";
}

void testAdditionBackwardWithIncomingGradient() {
    Value a(2.0);
    Value b(3.0);

    Value c = a + b;

    // test the chain-rule part
    c.grad = 4.0;

    backward(c);

    assert(approxEqual(a.grad, 4.0));
    assert(approxEqual(b.grad, 4.0));

    std::cout << "testAdditionBackwardWithIncomingGradient passed\n";
}

void testMultiplicationBackward() {
    Value a(2.0);
    Value b(3.0);

    Value c = a * b;

    c.grad = 1.0;

    backward(c);

    // c = ab
    //
    // dc/da = b = 3
    // dc/db = a = 2
    assert(approxEqual(a.grad, 3.0));
    assert(approxEqual(b.grad, 2.0));

    std::cout << "testMultiplicationBackward passed\n";
}

void testMultiplicationBackwardWithIncomingGradient() {
    Value a(2.0);
    Value b(3.0);

    Value c = a * b;

    // suppose dL/dc = 4
    c.grad = 4.0;

    backward(c);

    // dL/da = dL/dc * dc/da = 4 * 3
    // dL/db = dL/dc * dc/db = 4 * 2
    assert(approxEqual(a.grad, 12.0));
    assert(approxEqual(b.grad, 8.0));

    std::cout << "testMultiplicationBackwardWithIncomingGradient passed\n";
}

void testBackPropSimpleChain() {
    Value a(2.0);
    Value b(3.0);

    Value c = a * b;
    Value d = c + a;

    /*
        d = ab + a

        when a = 2, b = 3:

        d = 2*3 + 2 = 8

        dd/da = b + 1 = 4
        dd/db = a     = 2
    */

    assert(approxEqual(d.data, 8.0));

    // We currently have to seed the final gradient ourselves.
    d.grad = 1.0;

    backProp(d);

    assert(approxEqual(a.grad, 4.0));
    assert(approxEqual(b.grad, 2.0));

    std::cout << "testBackPropSimpleChain passed\n";
}

void testSameValueUsedTwice() {
    Value a(5.0);

    Value b = a + a;

    /*
        b = a + a = 2a

        db/da = 2
    */

    b.grad = 1.0;

    backProp(b);

    assert(approxEqual(a.grad, 2.0));

    std::cout << "testSameValueUsedTwice passed\n";
}

void testMultiplyValueByItself() {
    Value a(3.0);

    Value b = a * a;

    /*
        b = a^2

        db/da = 2a = 6
    */

    b.grad = 1.0;

    backProp(b);

    assert(approxEqual(a.grad, 6.0));

    std::cout << "testMultiplyValueByItself passed\n";
}

void testSharedIntermediate() {
    Value a(2.0);
    Value b(3.0);

    Value c = a * b;
    Value d = c + c;

    /*
        d = 2(ab)

        dd/da = 2b = 6
        dd/db = 2a = 4
    */

    d.grad = 1.0;

    backProp(d);

    assert(approxEqual(a.grad, 6.0));
    assert(approxEqual(b.grad, 4.0));

    std::cout << "testSharedIntermediate passed\n";
}

int main() {
    std::cout << "Running Value tests...\n\n";

    testConstructor();

    testAdditionForward();
    testMultiplicationForward();

    testAdditionBackward();
    testAdditionBackwardWithIncomingGradient();

    testMultiplicationBackward();
    testMultiplicationBackwardWithIncomingGradient();

    testBackPropSimpleChain();

    testSameValueUsedTwice();
    testMultiplyValueByItself();

    testSharedIntermediate();

    std::cout << "\nAll Value tests passed!\n";
}
