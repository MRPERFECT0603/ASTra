#include <iostream>

int maxOfThree(int a, int b, int c) {
    int arr[3] = {a, b, c};
    int max = arr[0];
    for (int i = 1; i < 3; i++) {
        if (arr[i] > max) max = arr[i];
    }
    return max;
}

int main() {
    int a, b, c;
    std::cout << "Enter three numbers: ";
    std::cin >> a >> b >> c;
    std::cout << "Maximum of three numbers is: " << maxOfThree(a, b, c) << std::endl;
    return 0;
}