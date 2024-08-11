#include <iostream>
#include <vector>
#include <algorithm>

int maxSubarraySum(std::vector<int>& arr) {
    int max_sum = arr[0];
    int current_sum = arr[0];
    for (int i = 1; i < arr.size(); i++) {
        current_sum = std::max(arr[i], current_sum + arr[i]);
        max_sum = std::max(max_sum, current_sum);
    }
    return max_sum;
}

int main() {
    std::vector<int> arr = {-2, 1, -3, 4, -1, 2, 1, -5, 4};
    int n = arr.size();
    std::cout << "Maximum sum of subarray: " << maxSubarraySum(arr) << std::endl;
    return 0;
}