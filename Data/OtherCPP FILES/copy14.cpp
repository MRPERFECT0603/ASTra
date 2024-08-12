#include <iostream>

class Rectangle {
public:
    int width;
    int height;
    int area() { return width * height; }
};

int main() {
    Rectangle rect;
    rect.width = 5;
    rect.height = 10;
    std::cout << "The area of the rectangle is " << rect.area() << std::endl;
    return 0;
}