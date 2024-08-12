#include <fstream>

int main() {
    std::ofstream file("example.txt");
    file << "This is an example file." << std::endl;
    file.close();
    return 0;
}