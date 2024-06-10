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
```


## Purpose of the Work
The project focuses on performing clustering analysis on a dataset using two different clustering algorithms: K-Means and Agglomerative Clustering. The main objective is to compare the performance of these algorithms in terms of cluster quality and to gain insights into the underlying structure of the dataset.

### Algorithms and Kernels

1. **K-Means Clustering:**
   - **Algorithm:** K-Means is an iterative algorithm that partitions the dataset into K clusters. Each observation belongs to the cluster with the nearest mean.
   - **Steps:**
     - Initialize K centroids randomly.
     - Assign each data point to the nearest centroid.
     - Recompute the centroids as the mean of the assigned points.
     - Repeat the assignment and update steps until convergence.
   - **Kernel:** The algorithm uses the Euclidean distance to measure similarity between data points and centroids.

2. **Agglomerative Clustering:**
   - **Algorithm:** Agglomerative Clustering is a hierarchical clustering method that builds nested clusters by merging or splitting them successively.
   - **Steps:**
     - Each data point starts as its own cluster.
     - Iteratively merge the closest pair of clusters based on a distance metric.
     - Continue until all points are merged into a single cluster or a specified number of clusters is reached.
   - **Kernel:** The algorithm typically uses a linkage criterion (e.g., single, complete, average) to determine the distance between sets of observations.

## Lessons Learned

1. **Cluster Initialization:**
   - The choice of initial centroids in K-Means significantly affects the clustering outcome and convergence speed. Various initialization methods (like K-Means++) can help achieve better results.

2. **Distance Metrics:**
   - The performance and quality of clusters are heavily influenced by the distance metric used. Euclidean distance is common, but other metrics may be more appropriate depending on the dataset's characteristics.

3. **Algorithm Selection:**
   - K-Means is efficient for large datasets but may struggle with clusters of varying shapes and densities. Agglomerative Clustering is more flexible but computationally intensive, making it less suitable for large datasets.

4. **Evaluating Clusters:**
   - Evaluation metrics such as silhouette score and inertia provide insights into the quality of the clusters formed. It's essential to use multiple metrics to get a comprehensive understanding of clustering performance.

5. **Parameter Tuning:**
   - Choosing the right number of clusters (K) is crucial. Methods like the elbow method and silhouette analysis are helpful in determining the optimal number of clusters.

## Significance of the Work

The project demonstrates a comprehensive application of clustering techniques on a real dataset, showcasing the practical considerations in clustering analysis. The comparative study of K-Means and Agglomerative Clustering algorithms highlights their strengths and weaknesses, providing valuable insights into their applicability for different types of data. By exploring various evaluation metrics and parameter tuning methods, the project contributes to a deeper understanding of effective clustering practices, which is significant for tasks such as market segmentation, image segmentation, and anomaly detection in various fields.
