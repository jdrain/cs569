/*
 * Rectangular matrix multiplication
 * A[M][K] * B[k][N] = C[M][N]
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/timeb.h>
#include <string.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

/* read timer in second */
double read_timer() {
    struct timeb tm;
    ftime(&tm);
    return (double) tm.time + (double) tm.millitm / 1000.0;
}

/* read timer in ms */
double read_timer_ms() {
    struct timeb tm;
    ftime(&tm);
    return (double) tm.time * 1000.0 + (double) tm.millitm;
}

#define REAL float
#define BLOCK_SIZE 16

void init(int M, int N, REAL * A) {
    int i, j;

    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            A[i*N+j] = (REAL) drand48();
        }
    }
}

double maxerror(int M, int N, REAL * A, REAL *B) {
    int i, j;
    double error = 0.0;

    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            double diff = (A[i*N+j] - B[i*N+j]) / A[i*N+j];
            if (diff < 0)
                diff = -diff;
            if (diff > error)
                error = diff;
        }
    }
    return error;
}

void matmul_base(int N, REAL *A, REAL * B, REAL *C);
void matmul_openmp(int N, REAL *A, REAL *B, REAL *C, int num_tasks);
void matmul_cuda_v1_vanilla(int N, REAL *A, REAL *B, REAL *C);
void matmul_cuda_v1_shmem(int N, REAL *A, REAL *B, REAL *C);
void matmul_cuda_v1_cublas(int N, REAL *A, REAL *B, REAL *C);

int main(int argc, char *argv[]) {
    int N;
    int num_tasks = 5; /* 5 is default number of tasks */
    double elapsed_base, elapsed_openmp, elapsed_cuda_v1, elapsed_cuda_v2, elapsed_cuda_v3;

    //double elapsed_cuda_v1, elapsed_cuda_v2, elapsed_cuda_v3; /* for timing */
    if (argc < 2) {
        fprintf(stderr, "Usage: matmul <n> [<#tasks(%d)>]\n", num_tasks);
        exit(1);
    }
    N = atoi(argv[1]);
    if (argc > 2) num_tasks = atoi(argv[2]);
    REAL * heap_buffer = (REAL*)malloc(sizeof(REAL)*N*N*7); /* we use 7 matrices in this example */

    /* below is a cast from memory buffer to a 2-d row-major array */
    REAL *A = heap_buffer;
    REAL *B = &heap_buffer[N*N];
    REAL *C_base = &heap_buffer[2*N*N];
    REAL *C_openmp = &heap_buffer[3*N*N];
    REAL *C_cuda_vanilla = &heap_buffer[4*N*N];
    REAL *C_cuda_shmem = &heap_buffer[5*N*N];
    REAL *C_cuda_cublas  = &heap_buffer[6*N*N];

    /* Init A and B with values */
    srand48((1 << 12));
    init(N, N, A);
    init(N, N, B);

    /* example run */
    elapsed_base = read_timer();
    matmul_base(N, A, B, C_base);
    elapsed_base = (read_timer() - elapsed_base);

    elapsed_openmp = read_timer();
    matmul_openmp(N, A, B, C_openmp, num_tasks);
    elapsed_openmp = (read_timer() - elapsed_openmp);

    /* call and timing for the three CUDA versions */
    //TODO: call and time for matmul_cuda_v1_vanilla(int N, REAL *A, REAL *B, REAL *C);
    elapsed_cuda_v1 = read_timer();
    matmul_cuda_v1_vanilla(N, A, B, C_cuda_vanilla);
    elapsed_cuda_v1 = read_timer() - elapsed_cuda_v1;

    //TODO: call and time for matmul_cuda_v1_shmem(int N, REAL *A, REAL *B, REAL *C);
    elapsed_cuda_v2 = read_timer();
    matmul_cuda_v1_shmem(N, A, B, C_cuda_shmem);
    elapsed_cuda_v2 = read_timer() - elapsed_cuda_v2;

    //TODO: call and time for matmul_cuda_v1_cublas(int N, REAL *A, REAL *B, REAL *C);
    elapsed_cuda_v3 = read_timer();
    matmul_cuda_v1_cublas(N, A, B, C_cuda_cublas);
    elapsed_cuda_v3 = read_timer() - elapsed_cuda_v3;

    printf("======================================================================================================\n");
    printf("Matrix Multiplication: A[M][K] * B[k][N] = C[M][N], M=K=N=%d, %d threads/tasks\n", N, num_tasks);
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("Performance:\t\tRuntime (ms)\t MFLOPS \t\tError (compared to base)\n");
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("matmul_base:\t\t%4f\t%4f \t\t%g\n", elapsed_base * 1.0e3, ((((2.0 * N) * N) * N) / (1.0e6 * elapsed_base)), maxerror(N, N, C_base, C_base));
    printf("matmul_openmp:\t\t%4f\t%4f \t\t%g\n", elapsed_openmp * 1.0e3, ((((2.0 * N) * N) * N) / (1.0e6 * elapsed_openmp)), maxerror(N, N, C_base, C_openmp));
    /* TODO: put other printf statements for outputing results for GPU execution */
    free(heap_buffer);
    return 0;
}

void matmul_base(int N, REAL *A, REAL * B, REAL *C) {
    int i, j, k;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            REAL temp = 0.0;
            for (k = 0; k < N; k++) {
                temp += A[i*N+k] * B[k*N+j];
            }
            C[i*N+j] = temp;
        }
    }
}

void matmul_openmp(int N, REAL *A, REAL *B, REAL *C, int num_tasks) {
    int i, j, k;
#pragma omp parallel for shared(N,A,B,C,num_tasks) private(i,j,k) num_threads(num_tasks)
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            REAL temp = 0.0;
            for (k = 0; k < N; k++) {
                temp += A[i*N+k] * B[k*N+j];
            }
            C[i*N+j] = temp;
        }
    }
}

/** 
  * TODO: kernel implementation 
  */
__global__ void matmul_cuda_v1_vanilla_kernel(int N, REAL *A, REAL *B, REAL *C ) {
    float C_val = 0.0;
    int row = BlockIdx.y * BlockDim.y + ThreadIdx.y;
    int col = BlockIdx.x * BlockDim.x + ThreadIdx.x;
    for ( int i = 0; i < N; i++ ) {
        C_val += A[row * N + i] * B[i * N + col];
    }
    C[ row * N + col ] = C_val;
}

/*
 * call to kernel that uses GPU global memory
 */
void matmul_cuda_v1_vanilla(int N, REAL *A, REAL *B, REAL *C) {

    // Copy A to device memory
    const REAL d_A;
    int size = N * N * sizeof(float);
    cudaMalloc(&d_A, size);
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);

    // Copy B to device memory
    const REAL d_B;
    cudaMalloc(&d_B, size);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Malloc for C
    const REAL d_C;
    cudaMalloc(&d_C, size);

    // Invoke kernel function
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(N / dimBlock.x, N / dimBlock.y);
    matmul_cuda_v1_vanilla_kernel<<<dimGrid, dimBlock>>>(N, d_A, d_B, d_C);
    
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// helper for shmem_kernel
__device__ REAL GetSubMatrix( REAL *A, int row, int col, int width ) {
    return &A[ row * width + col * BLOCK_SIZE ];
}

// helper for shmem_kernel
__device__ void SetElement( REAL *A, int row, int col, int width, float value ) {
    A[ row * width + col ] = value;
}

// helper for shmem_kernel
__device__ REAL GetElement( REAL *A, int row, int col, int width ) {
    return A[ row * width + col ]
}

/** 
  * TODO: kernel implementation 
  */
__global__ void matmul_cuda_v1_shmem_kernel( int N, REAL *A, REAL *B, REAL *C ) {
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Each thread block computes one sub-matrix Csub of C
    REAL * Csub = GetSubMatrix(C, blockRow, blockCol, N );

    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    float Cvalue = 0;

    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int m = 0; m < (N / BLOCK_SIZE); ++m) {

        // Get sub-matrix Asub of A
        REAL * Asub = GetSubMatrix(A, blockRow, m, N );

        // Get sub-matrix Bsub of B
        REAL * Bsub = GetSubMatrix(B, m, blockCol, N );

        // Shared memory used to store Asub and Bsub respectively
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        As[row][col] = GetElement(Asub, row, col, N);
        Bs[row][col] = GetElement(Bsub, row, col, N);

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();
        // Multiply Asub and Bsub together
        for (int e = 0; e < BLOCK_SIZE; ++e)
            Cvalue += As[row][e] * Bs[e][col];

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write Csub to device memory
    // Each thread writes one element
    SetElement(Csub, row, col, N, Cvalue);
}

/*
 * call to kernel that use GPU shared memory
 */
void matmul_cuda_v1_shmem(int N, REAL *A, REAL *B, REAL *C) {

    // Copy A to device memory
    const REAL d_A;
    int size = N * N * sizeof( REAL );
    cudaMalloc(&d_A, size);
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);

    // Copy B to device memory
    const REAL d_B;
    cudaMalloc(&d_B, size);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Malloc for C
    const REAL d_C;
    cudaMalloc(&d_C, size);

    // Invoke kernel function
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(N/ dimBlock.x, N / dimBlock.y);
    matmul_cuda_v1_shmem_kernel<<<dimGrid, dimBlock>>>(N, d_A, d_B, d_C);

    // Copy data back to host
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Free memory on GPU
    cudaFree(d_C);
    cudaFree(d_B);
    cudaFree(d_A);
}

/*
 * call to sgemm of cublas library 
 */
 // cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
 //             matrix_size.uiWB, matrix_size.uiHA, matrix_size.uiWA, &alpha, d_B, matrix_size.uiWB, d_A,
 //             matrix_size.uiWA, &beta, d_C, matrix_size.uiWA)
void matmul_cuda_v1_cublas(int N, REAL *A, REAL *B, REAL *C) {
    REAL alpha, beta = 1.0;

    // Copy A to device memory
    const REAL d_A;
    int size = N * N * sizeof( REAL );
    cudaMalloc(&d_A, size);
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);

    // Copy B to device memory
    const REAL d_B;
    cudaMalloc(&d_B, size);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Malloc for C
    const REAL d_C;
    cudaMalloc(&d_C, size);

    cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                N, N, N,
                &alpha, d_B, N, d_A,
                N, &beta, d_C, N);

    // Copy data back to host
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Free memory on GPU
    cudaFree(d_C);
    cudaFree(d_B);
    cudaFree(d_A);   
}
