/*
 * Author: Jason Drain
 * Class: CSCE 569
 * Date: 2/8/18
 *
 * Subject: Implement matrix vector multiplication
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/timeb.h>
#include <string.h>
#include "matprint.h"

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

void zero(REAL A[], int n) {
    int i, j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            A[i * n + j] = 0.0;
        }
    }
}

void init(int N, REAL A[]) {
    int i;
    for (i = 0; i < N; i++) {
        A[i] = (REAL) drand48();
    }
}

double check(REAL A[], REAL B[], int N) {
    int i;
    double sum = 0.0;
    for (i = 0; i < N; i++) {
        sum += A[i] - B[i];
    }
    return sum;
}

/**
 *
 * A[N][M] * B[M] = C[N]
 *
 */
void mv(int N, int M, REAL * A, REAL * B, REAL * C, int A_rowMajor) {
    if (A_rowMajor) {
        mv_rm(N, M, A, B, C);
        printf("\nRow Major\n");
        printf("\n======================================================\n");
        printMat(A, N, M, 1);
        printMat(B, M, 1, 0);
        printMat(C, N, 1, 0);
        printf("\n======================================================\n");
    } else {
        mv_cm(N, M, A, B, C);
        printf("\nColumn Major\n");
        printf("\n======================================================\n");
        printMat(A, N, M, 0);
        printMat(B, M, 1, 0);
        printMat(C, N, 1, 0);
        printf("\n======================================================\n");
    }
}

// row major matrix A
void mv_rm(int N, int M, REAL  * A, REAL * B, REAL * C) {
    for (int i = 0; i < N; i++) {
        REAL tmp = 0.0;
        for (int j = 0; j < M; j++) {
	          tmp += A[i*M + j] * B[j];
        }
        C[i] = tmp;
    }
}

// col major matrix A
void mv_cm(int N, int M, REAL * A, REAL * B, REAL * C) {
    for (int i = 0; i < N; i++) {
        REAL tmp = 0.0;
	      for (int j = 0; j < M; j++) {
	          tmp += A[i + j*N] * B[j];
	      }
        C[i] = tmp;
    }
}

int main(int argc, char *argv[]) {

    int N;
    int M;
    double elapsed_rm;
    double elapsed_cm;

    if (argc < 3) {
        printf("Usage: matvec <n> <m>\n");
	      printf("Defaults: n=512, m=512\n");
        N = 512;
	      M = 512;
    } else {
        N = atoi(argv[1]);
        M = atoi(argv[2]);
    }

    REAL * A = malloc(sizeof(REAL) * N * M);
    REAL * B = malloc(sizeof(REAL) * M);
    REAL * C = malloc(sizeof(REAL) * N);

    srand48((1 << 12));
    init(N * M, (REAL *) A);
    init(M, B);

    /* row major */
    elapsed_rm = read_timer();
    mv(N, M, A, B, C, 1);
    elapsed_rm = (read_timer() - elapsed_rm);

    /* colum major */
    elapsed_cm = read_timer();
    mv(N, M, A, B, C, 0);
    elapsed_cm = (read_timer() - elapsed_cm);

    /* you should add the call to each function and time the execution */
    printf("======================================================================================================\n");
    printf("\tMatrix Vector Multiplication: C[N] = A[N][M] * B[M], N=%d, M=%d\n", N, M);
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("Performance:\t\tRuntime (ms)\t MFLOPS\n");
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("row major:\t\t%4f\t%4f\n", elapsed_rm * 1.0e3, (2.0 * N * N) / (1.0e6 * elapsed_rm));
    printf("col major:\t\t%4f\t%4f\n", elapsed_cm * 1.0e3, (2.0 * N * N) / (1.0e6 * elapsed_cm));

    free(A);
    free(B);
    free(C);

    return 0;
}
