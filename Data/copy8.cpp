#include <iostream>

int main() {
    bool isAdmin = true;
    if (isAdmin) {
        std::cout << "You are an administrator." << std::endl;
    } else {
        std::cout << "You are not an administrator." << std::endl;
    }
    return 0;
}