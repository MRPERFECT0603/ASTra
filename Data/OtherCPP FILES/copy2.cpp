#include <iostream>

int maxOfThree(int a, int b, int c) {
    return (a > b) ? (a > c) ? a : c : (b > c) ? b : c;
}

int main() {
    int a, b, c;
    std::cout << "Enter three numbers: ";
    std::cin >> a >> b >> c;
    std::cout << "Maximum of three numbers is: " << maxOfThree(a, b, c) << std::endl;
    return 0;
}