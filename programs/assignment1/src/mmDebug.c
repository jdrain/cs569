/**
 *
 * Author: Jason Drain
 * Class: CSCE 569
 * Date: 2/6/18
 *
 * Subject: A program that multiplies matrices and records the execution times.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/timeb.h>
#include <omp.h>

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
#define VECTOR_LENGTH 512

/* initialize a vector with random floating point numbers */
void init(REAL A[], int N) {
    int i;
    for (i = 0; i < N; i++) {
        A[i] = (double) drand48();
    }
}

int num_ths = 4;

/**
 *
 * Multiply two matrices
 *
 * A[N][K] * B[K][M] = C[N][M]
 *
 */
void mm(int N, int K, int M, REAL * A, REAL * B, REAL * C, int A_rowMajor, int B_rowMajor) {
    if (A_rowMajor && B_rowMajor) {
	      mm_rm(N, K, M, A, B, C);
        printf("\n===================================================\n");
	      printMat(A, N, K, 1);
	      printMat(B, K, M, 1);
	      printMat(C, N, M, 1);
        printf("\n===================================================\n");
    }
    if (A_rowMajor && !B_rowMajor) {
        mm_A_rm(N, K, M, A, B, C);
        printf("\n===================================================\n");
	      printMat(A, N, K, 1);
	      printMat(B, K, M, 0);
	      printMat(C, N, M, 1);
        printf("\n===================================================\n");
    }
    if (!A_rowMajor && B_rowMajor) {
        mm_B_rm(N, K, M, A, B, C);
	      printf("\n===================================================\n");
	      printMat(A, N, K, 0);
	      printMat(B, K, M, 1);
	      printMat(C, N, M, 1);
        printf("\n===================================================\n");
    }
    if (!A_rowMajor && !B_rowMajor) {
	      mm_cm(N, K, M, A, B, C);
	      printf("\n===================================================\n");
	      printMat(A, N, K, 0);
	      printMat(B, K, M, 0);
	      printMat(C, N, M, 1);
	      printf("\n===================================================\n");
    }
}

/**
 *
 * Multiply when both are row major
 *
 */
void mm_rm(int N, int K, int M, REAL * A, REAL * B, REAL * C) {
    for (int i = 0; i < N; i++) { // for A rows
	      for (int j=0; j < M; j++) { // for B columns
            // do the ops
	          REAL tmp = 0.0;
	          for (int k=0; k < K; k++) { // for elems in A rows and B columns
		            tmp += A[i*K + k] * B[k*M + j];
	          }
	          C[i * M + j] = tmp;
        }
    }
}

/**
 *
 * Multiply when A is row major and B is column major
 *
 */
void mm_A_rm(int N, int K, int  M, REAL * A, REAL * B, REAL * C) {
    for (int i = 0; i < N; i++) { // for A rows
        for (int j = 0; j < M; j++) { // for B cols
            REAL tmp = 0.0;
	          for (int k = 0; k < K; k++) { // elems in row and column
                tmp += A[i * K + k] * B[j * K + k];
	          }
	          C[i * M + j] = tmp;
        }
    }
}

/**
 *
 * Multiply when A is column major and B is row major
 *
 */
void mm_B_rm(int N, int K, int M, REAL * A, REAL * B, REAL * C) {
    for (int i = 0; i < N; i++) { // for A rows
        for (int j = 0; j < M; j++) { // for B cols
            REAL tmp = 0.0;
	          for (int k = 0; k < K; k++) { // elems in row and column
                tmp += A[i + k * N] * B[j + k * M];
	          }
	          C[i * M + j] = tmp;
        }
    }
}

/**
 *
 * Multiply when both are column major
 *
 */
void mm_cm(int N, int K, int M, REAL * A, REAL * B, REAL * C) {
    for (int i = 0; i < N; i++) { // for A rows
	      for (int j = 0; j < M; j++) { // for B columns
            REAL tmp = 0.0;
	          for (int k = 0; k < K; k++) { // for elems in A rows and B columns
		            tmp += A[i + k*N] * B[k + j*K];
	          }
	          C[i * M + j] = tmp;
	      }
    }
}

/**
 *
 * To compile: gcc mm.c -fopenmp -o mm
 *
 */
int main(int argc, char *argv[]) {

    int N = VECTOR_LENGTH;
    int M = N;
    int K = N;
    double elapsed; /* for timing */

    if (argc < 5) {
        fprintf(stderr, "Usage: mm [<N(%d)>] <K(%d) [<M(%d)>] [<num_threads(%d)]\n", N,K,M, num_ths);
        fprintf(stderr, "\t Example: ./mm %d %d %d %d (default)\n", N,K,M,num_ths);
    } else {
    	  N = atoi(argv[1]);
    	  K = atoi(argv[2]);
    	  M = atoi(argv[3]);
    	  num_ths = atoi(argv[4]);
    }

    // printf("\tC[%d][%d] = A[%d][%d] * B[%d][%d] with %d threads\n", N, M, N, K, K, M, num_ths);
    REAL * A = malloc(sizeof(REAL)*N*K);
    REAL * B = malloc(sizeof(REAL)*K*M);
    REAL * C = malloc(sizeof(REAL)*N*M);

    srand48((1 << 12));
    init(A, N*K);
    init(B, K*M);

    // both matrices row major
    double elapsed_mm_rm = read_timer();
    mm(N, K, M, A, B, C, 1, 1);
    elapsed_mm_rm  = (read_timer() - elapsed_mm_rm);

    // A row major
    double elapsed_mm_A_rm = read_timer();
    mm(N, K, M, A, B, C, 1, 0);
    elapsed_mm_A_rm = (read_timer() - elapsed_mm_A_rm);

    // B row major
    double elapsed_mm_B_rm = read_timer();
    mm(N, K, M, A, B, C, 0, 1);
    elapsed_mm_B_rm = (read_timer() - elapsed_mm_B_rm);

    // both matrices column major
    double elapsed_mm_cm = read_timer();
    mm(N, K, M, A, B, C, 0, 0);
    elapsed_mm_cm = (read_timer() - elapsed_mm_cm);

    /* you should add the call to each function and time the execution */
    printf("======================================================================================================\n");
    printf("\tC[%d][%d] = A[%d][%d] * B[%d][%d] with %d threads for OpenMP\n", N, M, N, K, K, M, num_ths);
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("Performance:   \t\t\t\tRuntime (ms)\t MFLOPS \n");
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("mm row major:  \t\t\t\t%4f\t%4f\n",  elapsed_mm_rm * 1.0e3, M*N*K / (1.0e6 *  elapsed_mm_rm));
    printf("mm col major:  \t\t\t\t%4f\t%4f\n", elapsed_mm_cm * 1.0e3, M*N*K / (1.0e6 * elapsed_mm_cm));
    printf("mm A row major:\t\t\t\t%4f\t%4f\n", elapsed_mm_A_rm * 1.0e3, M*N*K / (1.0e6 * elapsed_mm_A_rm));
    printf("mm B row major:\t\t\t\t%4f\t%4f\n", elapsed_mm_B_rm * 1.0e3, M*N*K / (1.0e6 * elapsed_mm_B_rm));

    free(A);
    free(B);
    free(C);

    return 0;
}
