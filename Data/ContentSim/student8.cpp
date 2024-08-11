#include <iostream>
#include <vector>
#include <algorithm>

int maxSubarraySum(std::vector<int>& arr) {
    int max_sum = 0;
    int current_sum = 0;
    for (int i = 0; i < arr.size(); i++) {
        current_sum += arr[i];
        if (current_sum > max_sum) {
            max_sum = current_sum;
        }
        if (current_sum < 0) {
            current_sum = 0;
        }
    }
    return max_sum;
}

int main() {
    std::vector<int> arr = {-2, 1, -3, 4, -1, 2, 1, -5, 4};
    int n = arr.size();
    std::cout << "Maximum sum of subarray: " << maxSubarraySum(arr) << std::endl;
    return 0;
}