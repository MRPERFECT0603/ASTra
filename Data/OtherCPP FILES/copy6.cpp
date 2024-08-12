#include <iostream>

int maxOfThree(int a, int b, int c) {
    int max = a;
    if (b > max) max = b;
    if (c > max) max = c;
    return max;
}

int main() {
    int a, b, c;
    std::cout << "Enter three numbers: ";
    std::cin >> a >> b >> c;
    std::cout << "Maximum of three numbers is: " << maxOfThree(a, b, c) << std::endl;
    return 0;
}