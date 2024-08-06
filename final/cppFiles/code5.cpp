#include <iostream>
using namespace std;

int main() {
    int base, limit;
    cout << "Enter a number: ";
    cin >> base;
    cout << "Multiplication table for " << base << ":" << endl;
    for (int i = 1; i <= 10; ++i) {
        cout << base << " * " << i << " = " << (base * i) << endl;
    }
    return 0;
}
