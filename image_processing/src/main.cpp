#include <iostream>
#include "image_processing.h"

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <operation> <input_image> <output_image> [additional_args...]" << std::endl;
        return 1;
    }

    std::string operation = argv[1];
    const char* inputImagePath = argv[2];
    const char* outputImagePath = argv[3];

    if (operation == "canny") {
        if (!canny_edge_detection(inputImagePath, outputImagePath)) {
            std::cerr << "Canny edge detection failed." << std::endl;
            return 1;
        }
    } else if (operation == "erosion" || operation == "dilation") {
        if (argc < 5) {
            std::cerr << "Morphological operations require an additional argument: <kernel_size>" << std::endl;
            return 1;
        }
        int kernelSize = std::stoi(argv[4]);
        int morphOperation = (operation == "erosion") ? 0 : 1;
        if (!morphological_operations(inputImagePath, outputImagePath, morphOperation, kernelSize)) {
            std::cerr << "Morphological operation failed." << std::endl;
            return 1;
        }
    } else if (operation == "histogram") {
        if (!histogram_equalization(inputImagePath, outputImagePath)) {
            std::cerr << "Histogram equalization failed." << std::endl;
            return 1;
        }
    } else if (operation == "sobel") {
        if (!sobel_filter(inputImagePath, outputImagePath)) {
            std::cerr << "Sobel filter failed." << std::endl;
            return 1;
        }
    } else {
        std::cerr << "Unknown operation: " << operation << std::endl;
        return 1;
    }

    std::cout << "Operation completed successfully." << std::endl;
    return 0;
}


// ./image_processing canny input.jpg output.jpg
// ./image_processing erosion input.jpg output.jpg 3
// ./image_processing histogram input.jpg output.jpg
// ./image_processing sobel input.jpg output.jpg
