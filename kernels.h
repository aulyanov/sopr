#include <cuda_runtime_api.h>
#include <cstdio>
#include "GlobalEntitys.h"

#pragma once

void __global__ copyKernel(rett *src, rett *dst, const int n);
void __global__ myltiplyVectorOnScalarKernel(rett *v, rett *scalar,rett *result, const countt N);
void __global__ multVectorOnSkalarKernel(rett *A, rett *k, const int N);

//scalar multiply
void __global__ skalarMultInSingleBlockKernel(rett *a_Mass, rett *b_Mass, rett *res_Single, const int N_Mass);

//myltiply matrix on vector (only less then total GRAM)
void __global__ matrixOnVectorMultiplyKernel(rett *A, rett *b, rett *result, const countt N);
//myltiply part of  matrix on vector (for part of a large matrix)
void __global__ matrixPartOnVectorMultiplyKernel(rett *partA, rett *b, rett *result, const countt N, const countt partNumber, const countt partSize);
//multiply strip matrix on vector
void __global__ stripMatrixOnVectorMultiplyKernel(rett *stripA, rett *b, rett *result,const countt N, const countt B);

//что касается скалярного произведения
void __global__ multiplyVectorsAndPartialSumKernel(rett* a, rett* b, rett* resultAfterReductionGridSize, const countt N);
void __global__ reductionSumAtSingleBlockKernel(rett *input, rett *rScalar, const countt n);
