#ifndef IMAGE_PROCESSING_H
#define IMAGE_PROCESSING_H

bool canny_edge_detection(const char* inputImagePath, const char* outputImagePath);
bool gaussian_blur(const char* inputImagePath, const char* outputImagePath);
bool thresholding(const char* inputImagePath, const char* outputImagePath, int threshold);
bool morphological_operations(const char* inputImagePath, const char* outputImagePath, int operation, int kernelSize);
bool histogram_equalization(const char* inputImagePath, const char* outputImagePath);
bool sobel_filter(const char* inputImagePath, const char* outputImagePath);

#endif // IMAGE_PROCESSING_H
