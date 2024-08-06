#include <iostream>
#include <string>

using namespace std;

bool isPalindrome(string str) {
    int l = 0;
    int h = str.length() - 1;

    while (h > l) {
        if (str[l++] != str[h--]) {
            return false;
        }
    }
    return true;
}

int main() {
    string str;
    cout << "Enter a string: ";
    getline(cin, str);
    if (isPalindrome(str))
        cout << str << " is a palindrome.";
    else
        cout << str << " is not a palindrome.";
    return 0;
}
