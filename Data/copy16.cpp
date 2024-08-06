#include <vector>
#include <algorithm>
#include <iostream>
int main() {
    std::vector<int> numbers = {4, 2, 7, 1, 3};
    std::sort(numbers.begin(), numbers.end());
    for (int num : numbers) {
        std::cout << num << " ";
    }
    std::cout << std::endl;
    return 0;
}