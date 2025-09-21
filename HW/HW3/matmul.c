#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double gflops_calc(int n, double secs) {
    return (2.0 * n * (double)n * (double)n) / (secs * 1e9);
}

int main(int argc, char** argv) {
    if (argc != 2) { fprintf(stderr, "usage: %s N\n", argv[0]); return 1; }
    int n = atoi(argv[1]);
    double *A = (double*)malloc((size_t)n*n*sizeof(double));
    double *B = (double*)malloc((size_t)n*n*sizeof(double));
    double *C = (double*)calloc((size_t)n*n, sizeof(double));

    for (int i=0;i<n*n;i++){ A[i]=rand()/(double)RAND_MAX; B[i]=rand()/(double)RAND_MAX; }

    clock_t t0 = clock();
    for (int i=0;i<n;i++){
        for (int j=0;j<n;j++){
            double s = 0.0;
            for (int k=0;k<n;k++) s += A[i*(long)n+k]*B[k*(long)n+j];
            C[i*(long)n+j]=s;
        }
    }
    clock_t t1 = clock();
    double secs = (t1 - t0) / (double)CLOCKS_PER_SEC;
    printf("%f\n", gflops_calc(n, secs));

    free(A); free(B); free(C);
    return 0;
}
