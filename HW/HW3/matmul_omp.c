#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#ifndef BS
#define BS 128
#endif

static double gflops_calc(int n, double secs) {
    return (2.0 * n * (double)n * (double)n) / (secs * 1e9);
}

static inline int min_int(int a,int b){ return a<b?a:b; }

int main(int argc, char** argv) {
    if (argc != 2) { fprintf(stderr, "usage: %s N\n", argv[0]); return 1; }
    int n = atoi(argv[1]);
    float *A = (float*)malloc((size_t)n*n*sizeof(float));
    float *B = (float*)malloc((size_t)n*n*sizeof(float));
    float *C = (float*)malloc((size_t)n*n*sizeof(float));

    #pragma omp parallel for schedule(static)
    for (int i=0;i<n*n;i++){ A[i]= (float)rand()/RAND_MAX; B[i]=(float)rand()/RAND_MAX; C[i]=0.0f; }

    double t0 = omp_get_wtime();
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int ii=0; ii<n; ii+=BS){
        for (int kk=0; kk<n; kk+=BS){
            for (int i=ii; i<min_int(ii+BS,n); ++i) {
                for (int k=kk; k<min_int(kk+BS,n); ++k) {
                    float aik = A[i*(long)n + k];
                    float *cij = &C[i*(long)n];
                    float *bkj = &B[k*(long)n];
                    #pragma omp simd
                    for (int j=0; j<n; ++j) {
                        cij[j] += aik * bkj[j];
                    }
                }
            }
        }
    }
    double t1 = omp_get_wtime();
    printf("%f\n", gflops_calc(n, t1 - t0));

    free(A); free(B); free(C);
    return 0;
}
