#include <iostream>
#include <string>

using namespace std;

void countVowelsAndConsonants(string str, int &vowels, int &consonants) {
    vowels = 0;
    consonants = 0;
    for (char c : str) {
        if (isalpha(c)) {
            c = tolower(c);
            if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u')
                vowels++;
            else
                consonants++;
        }
    }
}

int main() {
    string str;
    cout << "Enter a string: ";
    getline(cin, str);
    int vowels, consonants;
    countVowelsAndConsonants(str, vowels, consonants);
    cout << "Vowels: " << vowels << ", Consonants: " << consonants;
    return 0;
}
