#include "shepard.h"
#include <cmath>

Image render (const std::vector<Point>& points, int width, int height, float q) {
    Image img;
    img.width = width;
    img.height = height;
    img.pixels.resize(width * height * 3); // Multiply by 3 to reserve space for a pixel's RGB values

    for (int py = 0; py < height; py++) {
        for (int px = 0; px < width; px++) {
            float total_weight = 0.0;
            float r = 0.0, g = 0.0, b = 0.0;

            for (const Point& point : points) {
                float dist_x = px - point.x;
                float dist_y = py - point.y;
                float dist = std::sqrt(dist_x*dist_x + dist_y*dist_y); // Using the pythagorean theorem to find the distance b/w the pixel and the point

                if (dist < 1e-6) { // Cannot be zero
                    r = point.r; g = point.g; b = point.b;
                    total_weight = -1.0;
                    break;
                }

                float weight = 1.0 / std::pow(dist, q);
                r += weight * point.r;
                g += weight * point.g;
                b += weight * point.b;
                total_weight += weight;
            }

                // Normalize the values
                if (total_weight > 0.0) {
                    r /= total_weight;
                    g /= total_weight;
                    b /= total_weight;
                }

            int index = (py * width + px) * 3; // Find the index of the pixel in the array using the formula

            img.pixels[index + 0] = (unsigned char)(r * 255.0); // For PNG conversion
            img.pixels[index + 1] = (unsigned char)(g * 255.0);
            img.pixels[index + 2] = (unsigned char)(b * 255.0);
        }
    }
    return img;
}