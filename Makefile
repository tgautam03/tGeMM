CC = nvcc

DEVICE_USAGE = --ptxas-options=-v
HOST_COMPILE_FLAG = -c
DEVICE_COMPILE_FLAG = -dc
LINK_CUBLAS = -lcublas
ADD_EIGEN = -I lib/eigen-3.4.0/
CPU_OPTIMIZE = -O3 -Xcompiler "-Ofast -march=native -funroll-loops -ffast-math -msse2 -msse3 -msse4 -mavx -mavx2 -flto"

# MatrixFP32
build/MatrixFP32.o: src/MatrixFP32.cu
	$(CC) $(DEVICE_COMPILE_FLAG) src/MatrixFP32.cu -o build/MatrixFP32.o

# MatrixFP16
build/MatrixFP16.o: src/MatrixFP16.cu
	$(CC) $(DEVICE_COMPILE_FLAG) src/MatrixFP16.cu -o build/MatrixFP16.o

# Utils
build/utils.o: src/utils.cu
	$(CC) $(DEVICE_COMPILE_FLAG) $(ADD_EIGEN) src/utils.cu -o build/utils.o

# cuBLAS
00_benchmark_cuBLAS.out: test/00_benchmark_cuBLAS.cu build/MatrixFP32.o build/MatrixFP16.o build/utils.o
	$(CC) $(LINK_CUBLAS) build/MatrixFP32.o build/MatrixFP16.o build/utils.o test/00_benchmark_cuBLAS.cu -o 00_benchmark_cuBLAS.out

# Naive Tensor vs cuBLAS
01_benchmark_naive.out: src/naive_tensor_xgemm.cu test/01_benchmark_naive.cu build/MatrixFP32.o build/MatrixFP16.o build/utils.o
	$(CC) -arch=sm_86 $(LINK_CUBLAS) build/MatrixFP32.o build/MatrixFP16.o build/utils.o src/naive_tensor_xgemm.cu test/01_benchmark_naive.cu -o 01_benchmark_naive.out

# Clean executable files
clean: 
	@echo "Removing object files..."
	rm *.out build/*.o