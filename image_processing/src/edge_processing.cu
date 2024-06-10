#include "image_processing.h"
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

// Canny edge detection kernel

__global__ void gaussianBlurKernel(const uchar* input, uchar* output, int width, int height, const float* kernel, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float sum = 0;
    int halfSize = kernelSize / 2;

    for (int ky = -halfSize; ky <= halfSize; ++ky) {
        for (int kx = -halfSize; kx <= halfSize; ++kx) {
            int px = min(max(x + kx, 0), width - 1);
            int py = min(max(y + ky, 0), height - 1);
            sum += kernel[(ky + halfSize) * kernelSize + (kx + halfSize)] * input[py * width + px];
        }
    }

    output[y * width + x] = static_cast<uchar>(sum);
}

__global__ void sobelKernel(const uchar* input, float* gradient, float* direction, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float Gx = 0;
    float Gy = 0;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        Gx = -input[(y - 1) * width + (x - 1)] - 2 * input[y * width + (x - 1)] - input[(y + 1) * width + (x - 1)] +
             input[(y - 1) * width + (x + 1)] + 2 * input[y * width + (x + 1)] + input[(y + 1) * width + (x + 1)];

        Gy = -input[(y - 1) * width + (x - 1)] - 2 * input[(y - 1) * width + x] - input[(y - 1) * width + (x + 1)] +
             input[(y + 1) * width + (x - 1)] + 2 * input[(y + 1) * width + x] + input[(y + 1) * width + (x + 1)];
    }

    gradient[y * width + x] = sqrt(Gx * Gx + Gy * Gy);
    direction[y * width + x] = atan2(Gy, Gx);
}

__global__ void nonMaxSuppressionKernel(const float* gradient, const float* direction, uchar* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float angle = direction[y * width + x] * 180.0 / M_PI;
    angle = angle < 0 ? angle + 180 : angle;

    float q = 255, r = 255;


    // Determine the neighboring pixels to interpolate
    if ((angle > 0 && angle <= 22.5) || (angle > 157.5 && angle <= 180)) {
        q = gradient[y * width + (x + 1)];
        r = gradient[y * width + (x - 1)];
    } else if (angle > 22.5 && angle <= 67.5) {//Initialize from 0
        q = gradient[(y + 1) * width + (x - 1)];
        r = gradient[(y - 1) * width + (x + 1)];
    } else if (angle > 67.5 && angle <= 112.5) {//45
        q = gradient[(y + 1) * width + x];
        r = gradient[(y - 1) * width + x];
    } else if (angle > 112.5 && angle <= 157.5) {//90
        q = gradient[(y - 1) * width + (x - 1)];
        r = gradient[(y + 1) * width + (x + 1)];
    }

    if (gradient[y * width + x] >= q && gradient[y * width + x] >= r) {
        output[y * width + x] = gradient[y * width + x];
    } else {
        output[y * width + x] = 0;
    }
}

__global__ void hysteresisKernel(const uchar* input, uchar* output, int width, int height, int lowThreshold, int highThreshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    if (input[y * width + x] >= highThreshold) {
        output[y * width + x] = 255;
    } else if (input[y * width + x] <= lowThreshold) {
        output[y * width + x] = 0;
    } else {
        bool connected = false;
        for (int ky = -1; ky <= 1; ++ky) {
            for (int kx = -1; kx <= 1; ++kx) {
                int nx = x + kx;
                int ny = y + ky;
                if (nx >= 0 && nx < width && ny >= 0 && ny < height && input[ny * width + nx] >= highThreshold) {
                    connected = true;
                    break;
                }
            }
        }
        output[y * width + x] = connected ? 255 : 0;
    }
}


bool canny_edge_detection(const char* inputImagePath, const char* outputImagePath) {
    cv::Mat inputImage = cv::imread(inputImagePath, cv::IMREAD_GRAYSCALE);
    if (inputImage.empty()) {
        return false;
    }

    int width = inputImage.cols;
    int height = inputImage.rows;
    cv::Mat blurredImage(height, width, CV_8UC1);
    cv::Mat gradientImage(height, width, CV_32FC1);
    cv::Mat directionImage(height, width, CV_32FC1);
    cv::Mat nonMaxSuppressedImage(height, width, CV_8UC1);
    cv::Mat outputImage(height, width, CV_8UC1);

    uchar* d_input;
    uchar* d_blurred;
    float* d_gradient;
    float* d_direction;
    uchar* d_nonMaxSuppressed;
    uchar* d_output;
    float* d_gaussianKernel;

    const int kernelSize = 5;
    float h_gaussianKernel[kernelSize * kernelSize] = {
        1, 4, 7, 4, 1,
        4, 16, 26, 16, 4,
        7, 26, 41, 26, 7,
        4, 16, 26, 16, 4,
        1, 4, 7, 4, 1
    };

    for (int i = 0; i < kernelSize * kernelSize; ++i) {
        h_gaussianKernel[i] /= 273.0f;
    }

    cudaMalloc(&d_input, width * height * sizeof(uchar));
    cudaMalloc(&d_blurred, width * height * sizeof(uchar));
    cudaMalloc(&d_gradient, width * height * sizeof(float));
    cudaMalloc(&d_direction, width * height * sizeof(float));
    cudaMalloc(&d_nonMaxSuppressed, width * height * sizeof(uchar));
    cudaMalloc(&d_output, width * height * sizeof(uchar));
    cudaMalloc(&d_gaussianKernel, kernelSize * kernelSize * sizeof(float));

    cudaMemcpy(d_input, inputImage.data, width * height * sizeof(uchar), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gaussianKernel, h_gaussianKernel, kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    gaussianBlurKernel<<<gridSize, blockSize>>>(d_input, d_blurred, width, height, d_gaussianKernel, kernelSize);
    cudaDeviceSynchronize();

    sobelKernel<<<gridSize, blockSize>>>(d_blurred, d_gradient, d_direction, width, height);
    cudaDeviceSynchronize();

    nonMaxSuppressionKernel<<<gridSize, blockSize>>>(d_gradient, d_direction, d_nonMaxSuppressed, width, height);
    cudaDeviceSynchronize();

    int lowThreshold = 50;
    int highThreshold = 150;
    hysteresisKernel<<<gridSize, blockSize>>>(d_nonMaxSuppressed, d_output, width, height, lowThreshold, highThreshold);
    cudaDeviceSynchronize();

    cudaMemcpy(outputImage.data, d_output, width * height * sizeof(uchar), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_blurred);
    cudaFree(d_gradient);
    cudaFree(d_direction);
    cudaFree(d_nonMaxSuppressed);
    cudaFree(d_output);
    cudaFree(d_gaussianKernel);

    cv::imwrite(outputImagePath, outputImage);
    return true;
}


// Implement Gaussian Blur
__global__ void gaussianBlurKernel(const uchar* input, uchar* output, int width, int height, const float* kernel, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float sum = 0;
    int halfSize = kernelSize / 2;

    for (int ky = -halfSize; ky <= halfSize; ++ky) {
        for (int kx = -halfSize; kx <= halfSize; ++kx) {
            int px = min(max(x + kx, 0), width - 1);
            int py = min(max(y + ky, 0), height - 1);
            sum += kernel[(ky + halfSize) * kernelSize + (kx + halfSize)] * input[py * width + px];
        }
    }

    output[y * width + x] = static_cast<uchar>(sum);
}

bool gaussian_blur(const char* inputImagePath, const char* outputImagePath) {
    cv::Mat inputImage = cv::imread(inputImagePath, cv::IMREAD_GRAYSCALE);
    if (inputImage.empty()) {
        return false;
    }

    int width = inputImage.cols;
    int height = inputImage.rows;
    cv::Mat outputImage(height, width, CV_8UC1);

    const int kernelSize = 5;
    float h_kernel[kernelSize * kernelSize] = {
        1, 4, 7, 4, 1,
        4, 16, 26, 16, 4,
        7, 26, 41, 26, 7,
        4, 16, 26, 16, 4,
        1, 4, 7, 4, 1
    };

    for (int i = 0; i < kernelSize * kernelSize; ++i) {
        h_kernel[i] /= 273.0f;
    }

    uchar* d_input;
    uchar* d_output;
    float* d_kernel;

    cudaMalloc(&d_input, width * height * sizeof(uchar));
    cudaMalloc(&d_output, width * height * sizeof(uchar));
    cudaMalloc(&d_kernel, kernelSize * kernelSize * sizeof(float));

    cudaMemcpy(d_input, inputImage.data, width * height * sizeof(uchar), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    gaussianBlurKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height, d_kernel, kernelSize);
    cudaDeviceSynchronize();

    cudaMemcpy(outputImage.data, d_output, width * height * sizeof(uchar), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);

    cv::imwrite(outputImagePath, outputImage);
    return true;
}

// Implement Thresholding
__global__ void thresholdingKernel(const uchar* input, uchar* output, int width, int height, int threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    output[y * width + x] = (input[y * width + x] > threshold) ? 255 : 0;
}

bool thresholding(const char* inputImagePath, const char* outputImagePath, int threshold) {
    cv::Mat inputImage = cv::imread(inputImagePath, cv::IMREAD_GRAYSCALE);
    if (inputImage.empty()) {
        return false;
    }

    int width = inputImage.cols;
    int height = inputImage.rows;
    cv::Mat outputImage(height, width, CV_8UC1);

    uchar* d_input;
    uchar* d_output;

    cudaMalloc(&d_input, width * height * sizeof(uchar));
    cudaMalloc(&d_output, width * height * sizeof(uchar));

    cudaMemcpy(d_input, inputImage.data, width * height * sizeof(uchar), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    thresholdingKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height, threshold);
    cudaDeviceSynchronize();

    cudaMemcpy(outputImage.data, d_output, width * height * sizeof(uchar), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    cv::imwrite(outputImagePath, outputImage);
    return true;
}

// Implement other image processing functions similarly...

// Placeholder for morphological operations, histogram equalization, sobel filter, etc.
__global__ void erosionKernel(const uchar* input, uchar* output, int width, int height, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int halfSize = kernelSize / 2;

    if (x >= width || y >= height) return;

    uchar minVal = 255;
    for (int ky = -halfSize; ky <= halfSize; ++ky) {
        for (int kx = -halfSize; kx <= halfSize; ++kx) {
            int px = min(max(x + kx, 0), width - 1);
            int py = min(max(y + ky, 0), height - 1);
            minVal = min(minVal, input[py * width + px]);
        }
    }

    output[y * width + x] = minVal;
}

__global__ void dilationKernel(const uchar* input, uchar* output, int width, int height, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int halfSize = kernelSize / 2;

    if (x >= width || y >= height) return;

    uchar maxVal = 0;
    for (int ky = -halfSize; ky <= halfSize; ++ky) {
        for (int kx = -halfSize; kx <= halfSize; ++kx) {
            int px = min(max(x + kx, 0), width - 1);
            int py = min(max(y + ky, 0), height - 1);
            maxVal = max(maxVal, input[py * width + px]);
        }
    }

    output[y * width + x] = maxVal;
}

bool morphological_operations(const char* inputImagePath, const char* outputImagePath, int operation, int kernelSize) {
    cv::Mat inputImage = cv::imread(inputImagePath, cv::IMREAD_GRAYSCALE);
    if (inputImage.empty()) {
        return false;
    }

    int width = inputImage.cols;
    int height = inputImage.rows;
    cv::Mat outputImage(height, width, CV_8UC1);

    uchar* d_input;
    uchar* d_output;

    cudaMalloc(&d_input, width * height * sizeof(uchar));
    cudaMalloc(&d_output, width * height * sizeof(uchar));

    cudaMemcpy(d_input, inputImage.data, width * height * sizeof(uchar), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    if (operation == 0) {
        erosionKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height, kernelSize);
    } else if (operation == 1) {
        dilationKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height, kernelSize);
    } else {
        std::cerr << "Unknown morphological operation: " << operation << std::endl;
        return false;
    }
    cudaDeviceSynchronize();

    cudaMemcpy(outputImage.data, d_output, width * height * sizeof(uchar), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    cv::imwrite(outputImagePath, outputImage);
    return true;
}

// histogram_equalization

__global__ void histogramEqualizationKernel(uchar* input, uchar* output, int width, int height) {
    __shared__ int hist[256];
    __shared__ int cdf[256];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    if (tid < 256) {
        hist[tid] = 0;
        cdf[tid] = 0;
    }
    __syncthreads();

    if (x < width && y < height) {
        atomicAdd(&hist[input[y * width + x]], 1);
    }
    __syncthreads();

    if (tid < 256) {
        for (int i = 0; i <= tid; ++i) {
            cdf[tid] += hist[i];
        }
    }
    __syncthreads();

    if (x < width && y < height) {
        int pixel = input[y * width + x];
        output[y * width + x] = (uchar)((cdf[pixel] - cdf[0]) * 255 / (width * height - cdf[0]));
    }
}

bool histogram_equalization(const char* inputImagePath, const char* outputImagePath) {
    cv::Mat inputImage = cv::imread(inputImagePath, cv::IMREAD_GRAYSCALE);
    if (inputImage.empty()) {
        return false;
    }

    int width = inputImage.cols;
    int height = inputImage.rows;
    cv::Mat outputImage(height, width, CV_8UC1);

    uchar* d_input;
    uchar* d_output;

    cudaMalloc(&d_input, width * height * sizeof(uchar));
    cudaMalloc(&d_output, width * height * sizeof(uchar));

    cudaMemcpy(d_input, inputImage.data, width * height * sizeof(uchar), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    histogramEqualizationKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize();

    cudaMemcpy(outputImage.data, d_output, width * height * sizeof(uchar), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    cv::imwrite(outputImagePath, outputImage);
    return true;
}

// sobel filter

__global__ void sobelFilterKernel(const uchar* input, uchar* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float Gx = 0;
    float Gy = 0;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        Gx = -input[(y - 1) * width + (x - 1)] - 2 * input[y * width + (x - 1)] - input[(y + 1) * width + (x - 1)] +
             input[(y - 1) * width + (x + 1)] + 2 * input[y * width + (x + 1)] + input[(y + 1) * width + (x + 1)];

        Gy = -input[(y - 1) * width + (x - 1)] - 2 * input[(y - 1) * width + x] - input[(y - 1) * width + (x + 1)] +
             input[(y + 1) * width + (x - 1)] + 2 * input[(y + 1) * width + x] + input[(y + 1) * width + (x + 1)];
    }

    output[y * width + x] = sqrt(Gx * Gx + Gy * Gy);
}

bool sobel_filter(const char* inputImagePath, const char* outputImagePath) {
    cv::Mat inputImage = cv::imread(inputImagePath, cv::IMREAD_GRAYSCALE);
    if (inputImage.empty()) {
        return false;
    }

    int width = inputImage.cols;
    int height = inputImage.rows;
    cv::Mat outputImage(height, width, CV_8UC1);

    uchar* d_input;
    uchar* d_output;

    cudaMalloc(&d_input, width * height * sizeof(uchar));
    cudaMalloc(&d_output, width * height * sizeof(uchar));

    cudaMemcpy(d_input, inputImage.data, width * height * sizeof(uchar), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    sobelFilterKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize();

    cudaMemcpy(outputImage.data, d_output, width * height * sizeof(uchar), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    cv::imwrite(outputImagePath, outputImage);
    return true;
}

