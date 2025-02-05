#include <iostream>
#include <iomanip>
#include <cublas_v2.h>
#include <device_launch_parameters.h>

#include "../include/MatrixFP32.cuh"
#include "../include/MatrixFP16.cuh"
#include "../include/utils.cuh"

#include "../include/naive_tensor_tgemm.cuh"

int main(int argc, char const *argv[])
{
    // Options: Anything!
    int mat_sizes[] = {128, 256, 512, 1024, 2048, 4096};
    int n_sizes = sizeof(mat_sizes) / sizeof(mat_sizes[0]);

    // For recording time
    float elapsed_time;
    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);

    // Store time and GFLOPS
    double tgemm_time[n_sizes];
    double tgemm_gflops[n_sizes];
    double cublas_time[n_sizes];
    double cublas_gflops[n_sizes];

    for (int mat_size = 0; mat_size < n_sizes; mat_size++)
    {
        // Matrix Size
        int n = mat_sizes[mat_size];

        // Define MatrixFP16 & MatrixFP32
        MatrixFP16 A_FP16 = MatrixFP16(n, n, false);
        MatrixFP16 B_FP16 = MatrixFP16(n, n, false);
        MatrixFP32 C_FP32_cublas = MatrixFP32(n, n, false);
        MatrixFP32 C_FP32_tgemm = MatrixFP32(n, n, false);

        // Initialize Matrices
        random_init_mat(A_FP16, -10, 10);          // Random Initialization between -10 and 10
        random_init_mat(B_FP16, -10, 10);          // Random Initialization between -10 and 10
        init_mat(C_FP32_cublas, 1.0f);     // Initialize to 1
        init_mat(C_FP32_tgemm, -1.0f);     // Initialize to 1

        // Move matrices to device
        MatrixFP16 d_A_FP16 = MatrixFP16(n, n, true); 
        A_FP16.copy_to_device(d_A_FP16);
        MatrixFP16 d_B_FP16 = MatrixFP16(n, n, true); 
        B_FP16.copy_to_device(d_B_FP16);
        MatrixFP32 d_C_FP32_cublas = MatrixFP32(n, n, true); 
        C_FP32_cublas.copy_to_device(d_C_FP32_cublas);
        MatrixFP32 d_C_FP32_tgemm = MatrixFP32(n, n, true); 
        C_FP32_tgemm.copy_to_device(d_C_FP32_tgemm);
        cudaDeviceSynchronize();

        //----------------------------------------------------//
        //-------------------- Warmup Run --------------------//
        //----------------------------------------------------//
        // Create and initialize cuBLAS handle
        cublasHandle_t handle;
        cublas_check(cublasCreate(&handle));
        
        // Perform matrix multiplication: C = A * B 
        float alpha = 1;
        float beta = 0;
        cublas_check(cublasGemmEx(handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                d_C_FP32_cublas.n_cols, d_C_FP32_cublas.n_rows, d_A_FP16.n_cols, // Num Cols of C, Num rows of C, Shared dim of A and B
                                &alpha,
                                d_B_FP16.ptr, CUDA_R_16F, d_B_FP16.n_cols, // Num cols of B
                                d_A_FP16.ptr, CUDA_R_16F, d_A_FP16.n_cols, // Num cols of A
                                &beta,
                                d_C_FP32_cublas.ptr, CUDA_R_32F, d_C_FP32_cublas.n_cols,
                                CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) // Num cols of C
                    );
        cudaDeviceSynchronize();

        // Naive Kernel execution
        naive_tensor_tgemm(d_A_FP16.ptr, d_B_FP16.ptr, d_C_FP32_tgemm.ptr, d_C_FP32_tgemm.n_rows, d_C_FP32_tgemm.n_cols, d_A_FP16.n_cols);
        cudaDeviceSynchronize();

        // Assert that naive implementation is correct
        d_C_FP32_cublas.copy_to_host(C_FP32_cublas);
        d_C_FP32_tgemm.copy_to_host(C_FP32_tgemm);
        std::cout << "Asserting Results for N: " << n << "\n";
        assert_mat(C_FP32_tgemm, C_FP32_cublas, 1e-8);
        std::cout << "Assertion Passed! \n \n";

        // Printing the smallest matrix result
        if (n <= 8)
        {
            std::cout << "Matrix C (cuBLAS): \n";
            print_mat(C_FP32_cublas, true);
            std::cout << "\n";

            std::cout << "Matrix C (tGeMM): \n";
            print_mat(C_FP32_tgemm, true);
            std::cout << "\n";
        }

        //----------------------------------------------------//
        //---------------------- tGeMM -----------------------//
        //----------------------------------------------------//
        cudaEventRecord(beg);
        for (int n_runs = 0; n_runs < 10; n_runs++)
        {
            naive_tensor_tgemm(d_A_FP16.ptr, d_B_FP16.ptr, d_C_FP32_tgemm.ptr, d_C_FP32_tgemm.n_rows, d_C_FP32_tgemm.n_cols, d_A_FP16.n_cols);
            cudaDeviceSynchronize();
        }
        cudaEventRecord(end);
        cudaEventSynchronize(beg);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&elapsed_time, beg, end);
        elapsed_time /= 1000.;

        tgemm_time[mat_size] = (elapsed_time) / 10;
        tgemm_gflops[mat_size] = 2. * 1e-9 * 10 * n * n * n / (elapsed_time);

        // Free Memory
        A_FP16.free_mat();
        B_FP16.free_mat();
        C_FP32_cublas.free_mat();
        C_FP32_tgemm.free_mat();

        d_A_FP16.free_mat();
        d_B_FP16.free_mat();
        d_C_FP32_cublas.free_mat();
        d_C_FP32_tgemm.free_mat();
    }

    // Reading cuBLAS times and GFLOPS
    std::ifstream inputFile("txt_benchmarks/cublas.txt");
    if (!inputFile.is_open()) 
    {
        std::cerr << "Error opening the file!" << std::endl;
        return 1;
    }
    std::string line;
    // Skip the first two lines
    for (int i = 0; i < 2; ++i) 
    {
        if (!std::getline(inputFile, line)) {
            std::cerr << "File has fewer than 3 lines!" << std::endl;
            return 1;
        }
    }
    // Read the third line into cublas_time
    if (std::getline(inputFile, line)) 
    {
        std::istringstream iss(line);
        std::string word;
        // Skip "Time (Seconds):"
        iss >> word >> word;
        
        // Read the double values
        int count = 0;
        while (iss >> cublas_time[count] && count < n_sizes)
            count++;
    }
    // Skip the fourth line
    if (!std::getline(inputFile, line)) 
    {
        std::cerr << "File has fewer than 5 lines!" << std::endl;
        return 1;
    }
    // Read the fifth line into cublas_gflops
    if (std::getline(inputFile, line)) 
    {
        std::istringstream iss(line);
        std::string word;
        // Skip "GFLOPS:"
        iss >> word;
        
        // Read the double values
        int count = 0;
        while (iss >> cublas_gflops[count] && count < n_sizes)
            count++;
    }

    // Printing Results
    std::cout << "Matrix Size: ";
    for (int mat_size = 0; mat_size < n_sizes; mat_size++)
        std::cout << mat_sizes[mat_size] << " ";
    std::cout << "\n \n";

    std::cout << "cuBLAS Time (seconds): ";
    for (int mat_size = 0; mat_size < n_sizes; mat_size++)
        std::cout << cublas_time[mat_size] << " ";
    std::cout << "\n";
    std::cout << "tGeMM Time (seconds): ";
    for (int mat_size = 0; mat_size < n_sizes; mat_size++)
        std::cout << tgemm_time[mat_size] << " ";
    std::cout << "\n \n";

    std::cout << "cuBLAS GFLOPS: ";
    for (int mat_size = 0; mat_size < n_sizes; mat_size++)
        std::cout << cublas_gflops[mat_size] << " ";
    std::cout << "\n";
    std::cout << "tGeMM GFLOPS: ";
    for (int mat_size = 0; mat_size < n_sizes; mat_size++)
        std::cout << tgemm_gflops[mat_size] << " ";
    std::cout << "\n \n";

    std::cout << "cuBLAS vs Naive tGeMM (CuBLAS/tGeMM): ";
    for (int mat_size = 0; mat_size < n_sizes; mat_size++)
        std::cout << std::fixed << std::setprecision(2) << cublas_time[mat_size]/tgemm_time[mat_size]*100 << "% ";
    std::cout << "\n";

    // Saving to benchmark file
    update_benckmark_txt("txt_benchmarks/naive_tgemm.txt", tgemm_time, tgemm_gflops, mat_sizes, n_sizes);

    return 0;
}
