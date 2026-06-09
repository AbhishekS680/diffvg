#pragma once
#include <vector>

struct Point {
    float x, y;
    float r, g, b;
};

struct Image {
    int width, height;
    std::vector<unsigned char> pixels;
};

Image render (const std::vector<Point>& points, int width, int height, float q);