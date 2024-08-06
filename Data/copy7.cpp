#include <iostream>

int maxOfThree(int a, int b, int c) {
    if (a == b && b == c) {
        return a;
    } else if (a > b && a > c) {
        return a;
    } else if (b > a && b > c) {
        return b;
    } else {
        return c;
    }
}

int main() {
    int a, b, c;
    std::cout << "Enter three numbers: ";
    std::cin >> a >> b >> c;
    std::cout << "Maximum of three numbers is: " << maxOfThree(a, b, c) << std::endl;
    return 0;
}