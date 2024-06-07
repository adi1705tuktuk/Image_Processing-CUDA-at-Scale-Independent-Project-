#include <iostream>
#include "edge_detection.h"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_image> <output_image>" << std::endl;
        return 1;
    }

    const char* inputImagePath = argv[1];
    const char* outputImagePath = argv[2];

    if (!edge_detection(inputImagePath, outputImagePath)) {
        std::cerr << "Edge detection failed!" << std::endl;
        return 1;
    }

    std::cout << "Edge detection completed successfully!" << std::endl;
    return 0;
}
