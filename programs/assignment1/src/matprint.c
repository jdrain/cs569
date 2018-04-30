#include "matprint.h"

/**
 *
 * Print functions for result verification/debugging
 *
 */
void printMat(REAL * A, int N, int M, int rowMajor) {
    if (rowMajor) {
        printf("\nRow Major:\t%d by %d\n", N, M);
	      print_rmMat(A, N, M);
    } else {
	      printf("\nColumn Major:\t%d by %d\n", N, M);
	      print_cmMat(A, N, M);
    }
}

void print_rmMat(REAL * A, int N, int M) {
    for (int i = 0; i < N; i++) {
	      for (int j = 0; j < M; j++) {
            printf("A[%i]: %f\t", i*M + j, A[i*M + j]);
	      }
	      printf("\n");
    }
}

void print_cmMat(REAL * A, int N, int M) {
    for (int i = 0; i < N; i++) {
	      for (int j = 0; j < M; j++) {
            printf("A[%i]: %f\t", j*N + i, A[j*N + i]);
	      }
	      printf("\n");
    }
}
