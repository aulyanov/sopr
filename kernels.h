#include <cuda_runtime_api.h>
#include <cstdio>
#include "GlobalEntitys.h"

void __global__ copyKernel(rett *src, rett *dst, const int n);
void __global__ multVectorOnSkalarKernel(rett *A, rett *k, const int N);

//scalar multiply
void __global__ skalarMultInSingleBlockKernel(rett *a_Mass, rett *b_Mass, rett *res_Single, const int N_Mass);

//multiply matrix on vector (large)
void __global__ matrixOnVectorMultiplyKernel(rett *A, rett *b, rett *result, const countt N);