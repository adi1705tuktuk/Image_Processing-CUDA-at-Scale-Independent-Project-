# CUDA Image Processing

This project provides CUDA implementations of various image processing operations, including Canny edge detection, morphological operations (erosion and dilation), histogram equalization, and the Sobel filter. The goal is to perform these operations efficiently using CUDA on large amounts of image data.

## Features

- **Canny Edge Detection**: Detects edges in images.
- **Morphological Operations**: Erosion and dilation.
- **Histogram Equalization**: Enhances the contrast of images.
- **Sobel Filter**: Detects edges using the Sobel operator.

## Prerequisites

- CUDA toolkit installed
- OpenCV library installed
- C++ compiler (e.g., g++)


## Building the Project

1. **Clone the repository:**

    ```sh
    git clone <repository_url>
    cd image_processing
    ```

2. **Create necessary directories:**

    ```sh
    mkdir -p build bin data/input data/output
    ```

3. **Build the project using Makefile:**

    ```sh
    make
    ```

## Running the Project

### Using the `run.sh` Script

The `run.sh` script processes all images in the `data/input` directory and saves the processed images to the `data/output` directory.

```sh
Usage: ./run.sh [-o operation] [-k kernel_size] [-i input_dir] [-u output_dir]
  -o operation: The operation to perform (canny, erosion, dilation, histogram, sobel)
  -k kernel_size: Kernel size for morphological operations (default: 3)
  -i input_dir: Directory containing input images (default: data/input)
  -u output_dir: Directory to save output images (default: data/output)


