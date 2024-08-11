#include <iostream>
using namespace std;

int main() {
    int a, b, c, maxValue;
    cout << "Enter three numbers: ";
    cin >> a >> b >> c;
    maxValue = (a > b) ? (a > c ? a : c) : (b > c ? b : c);
    cout << "Maximum: " << maxValue << endl;
    return 0;
}
