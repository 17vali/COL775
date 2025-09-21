#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifndef BS
#define BS 64
#endif

static double gflops_calc(int n, double secs) {
    return (2.0 * n * (double)n * (double)n) / (secs * 1e9);
}

int min_int(int a,int b){ return a<b?a:b; }

int main(int argc, char** argv) {
    if (argc != 2) { fprintf(stderr, "usage: %s N\n", argv[0]); return 1; }
    int n = atoi(argv[1]);
    double *A = (double*)malloc((size_t)n*n*sizeof(double));
    double *B = (double*)malloc((size_t)n*n*sizeof(double));
    double *C = (double*)calloc((size_t)n*n, sizeof(double));

    for (int i=0;i<n*n;i++){ A[i]=rand()/(double)RAND_MAX; B[i]=rand()/(double)RAND_MAX; }

    clock_t t0 = clock();
    for (int ii=0; ii<n; ii+=BS) {
        for (int kk=0; kk<n; kk+=BS) {
            for (int i=ii; i<min_int(ii+BS,n); ++i) {
                for (int k=kk; k<min_int(kk+BS,n); ++k) {
                    double aik = A[i*(long)n + k];
                    double *cij = &C[i*(long)n];
                    double *bkj = &B[k*(long)n];
                    for (int j=0; j<n; ++j) {
                        cij[j] += aik * bkj[j];
                    }
                }
            }
        }
    }
    clock_t t1 = clock();
    double secs = (t1 - t0) / (double)CLOCKS_PER_SEC;
    printf("%f\n", gflops_calc(n, secs));

    free(A); free(B); free(C);
    return 0;
}
