#include <iostream>

int main() {
    int arr[5] = {1, 2, 3, 4, 5};
    for (int i = 0; i < 5; i++) {
        std::cout << "Element at index " << i << " is " << arr[i] << std::endl;
    }
    return 0;
}