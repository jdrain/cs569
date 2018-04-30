#ifndef MATPRINT_H_
#define MATPRINT_H_
#define REAL float

void printMat(REAL * A, int N,  int M, int rowMajor);
void print_rmMat(REAL * A, int N, int M);
void print_cmMat(REAL * A, int N, int M);

#endif
