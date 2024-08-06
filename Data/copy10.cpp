#include <iostream>

int main() {
    int x = 10;
    int* p = &x;
    std::cout << "The value of x is " << x << std::endl;
    std::cout << "The address of x is " << p << std::endl;
    return 0;
}