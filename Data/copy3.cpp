#include <iostream>
#include <algorithm>

int main() {
    int a, b, c;
    std::cout << "Enter three numbers: ";
    std::cin >> a >> b >> c;
    std::cout << "Maximum of three numbers is: " << std::max({a, b, c}) << std::endl;
    return 0;
}