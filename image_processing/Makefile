NVCC = /usr/local/cuda/bin/nvcc
CXX = g++
CXXFLAGS = -I/usr/local/include/opencv4 -I/Users/adityaprasad/Desktop/image_processing/include
LDFLAGS = -L/usr/local/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc

all: edge_detection

edge_detection: main.o edge_detection.o
	$(NVCC) -o edge_detection main.o edge_detection.o $(LDFLAGS)

main.o: src/main.cpp include/edge_detection.h
	$(CXX) $(CXXFLAGS) -c src/main.cpp

edge_detection.o: src/edge_detection.cu include/edge_detection.h
	$(NVCC) -c src/edge_detection.cu

clean:
	rm -f *.o edge_detection


