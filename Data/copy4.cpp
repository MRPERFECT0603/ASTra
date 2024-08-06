#include <iostream>

int maxOfThree(int a, int b, int c) {
    return (a + b + c) - std::min({a, b, c}) - std::min({b, c, a}) - std::min({c, a, b});
}

int main() {
    int a, b, c;
    std::cout << "Enter three numbers: ";
    std::cin >> a >> b >> c;
    std::cout << "Maximum of three numbers is: " << maxOfThree(a, b, c) << std::endl;
    return 0;
}