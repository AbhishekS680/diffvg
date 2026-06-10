#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "shepard.h"
#include <vector>

int main() {
    int width = 512;
    int height = 512;
    float q = 15.0; // How fast a point's influence drops with distance

    // Points (x, y, r, g, b)
    std::vector<Point> points = {
        {100, 100, 1.0f, 0.0f, 0.0f},  // red
        {400, 100, 0.0f, 1.0f, 0.0f},  // green
        {250, 250, 0.0f, 0.0f, 1.0f},  // blue
        {400, 400, 1.0f, 1.0f, 0.0f},  // yellow
        {100, 400, 0.0f, 1.0f, 1.0f},  // cyan
        {250, 100, 1.0f, 0.0f, 1.0f},  // magenta
        {100, 250, 1.0f, 0.5f, 0.0f},  // orange
        {400, 250, 0.5f, 0.0f, 1.0f},  // purple
        {175, 350, 0.0f, 0.5f, 0.5f},  // teal
        {350, 175, 1.0f, 1.0f, 1.0f},  // white
    };

    Image img = render(points, width, height, q);

    stbi_write_png("output.png", width, height, 3, img.pixels.data(), width * 3);

    return 0;
}