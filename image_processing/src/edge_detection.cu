#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "edge_detection.h"

__constant__ float d_gaussianKernel[25]; // 5x5 Gaussian kernel

__global__ void gradientKernel(const uchar* input, float* gradient, float* direction, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Sobel operators
    const float Gx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    const float Gy[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

    float sumX = 0;
    float sumY = 0;

    for (int ky = -1; ky <= 1; ++ky) {
        for (int kx = -1; kx <= 1; ++kx) {
            int px = min(max(x + kx, 0), width - 1);
            int py = min(max(y + ky, 0), height - 1);
            float pixel = static_cast<float>(input[py * width + px]);
            sumX += pixel * Gx[ky + 1][kx + 1];
            sumY += pixel * Gy[ky + 1][kx + 1];
        }
    }

    float grad = sqrtf(sumX * sumX + sumY * sumY);
    float angle = atan2f(sumY, sumX) * 180.0f / M_PI;

    gradient[y * width + x] = grad;
    direction[y * width + x] = angle;
}

__global__ void nonMaximumSuppressionKernel(const float* gradient, const float* direction, uchar* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float angle = direction[y * width + x];
    float grad = gradient[y * width + x];

    float neighbor1 = 0;
    float neighbor2 = 0;

    if ((angle > -22.5 && angle <= 22.5) || (angle > 157.5 || angle <= -157.5)) {
        neighbor1 = (x > 0) ? gradient[y * width + (x - 1)] : 0;
        neighbor2 = (x < width - 1) ? gradient[y * width + (x + 1)] : 0;
    } else if ((angle > 22.5 && angle <= 67.5) || (angle > -157.5 && angle <= -112.5)) {
        neighbor1 = (x > 0 && y > 0) ? gradient[(y - 1) * width + (x - 1)] : 0;
        neighbor2 = (x < width - 1 && y < height - 1) ? gradient[(y + 1) * width + (x + 1)] : 0;
    } else if ((angle > 67.5 && angle <= 112.5) || (angle > -112.5 && angle <= -67.5)) {
        neighbor1 = (y > 0) ? gradient[(y - 1) * width + x] : 0;
        neighbor2 = (y < height - 1) ? gradient[(y + 1) * width + x] : 0;
    } else if ((angle > 112.5 && angle <= 157.5) || (angle > -67.5 && angle <= -22.5)) {
        neighbor1 = (x < width - 1 && y > 0) ? gradient[(y - 1) * width + (x + 1)] : 0;
        neighbor2 = (x > 0 && y < height - 1) ? gradient[(y + 1) * width + (x - 1)] : 0;
    }

    output[y * width + x] = (grad >= neighbor1 && grad >= neighbor2) ? static_cast<uchar>(grad) : 0;
}

bool edge_detection(const char* inputImagePath, const char* outputImagePath) {
    cv::Mat inputImage = cv::imread(inputImagePath, cv::IMREAD_GRAYSCALE);
    if (inputImage.empty()) {
        return false;
    }

    int width = inputImage.cols;
    int height = inputImage.rows;

    uchar* d_input;
    float* d_gradient;
    float* d_direction;
    uchar* d_output;

    cudaMalloc(&d_input, width * height * sizeof(uchar));
    cudaMalloc(&d_gradient, width * height * sizeof(float));
    cudaMalloc(&d_direction, width * height * sizeof(float));
    cudaMalloc(&d_output, width * height * sizeof(uchar));

    cudaMemcpy(d_input, inputImage.data, width * height * sizeof(uchar), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    gradientKernel<<<gridSize, blockSize>>>(d_input, d_gradient, d_direction, width, height);
    cudaDeviceSynchronize();

    nonMaximumSuppressionKernel<<<gridSize, blockSize>>>(d_gradient, d_direction, d_output, width, height);
    cudaDeviceSynchronize();

    uchar* outputImage = new uchar[width * height];
    cudaMemcpy(outputImage, d_output, width * height * sizeof(uchar), cudaMemcpyDeviceToHost);

    cv::Mat resultImage(height, width, CV_8UC1, outputImage);
    cv::imwrite(outputImagePath, resultImage);

    cudaFree(d_input);
    cudaFree(d_gradient);
    cudaFree(d_direction);
    cudaFree(d_output);
    delete[] outputImage;

    return true;
}
