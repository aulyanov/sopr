#include <cuda_runtime_api.h>
#include <cstdio>
#include "GlobalEntitys.h"

#pragma once

//Векторная арифметика
	void __global__ copyKernel(rett *src, rett *dst, const countt N);
	//result = X + aY
	void __global__ xPlusAlfaYKernel(rett *X, rett *Y, rett* aScalar,rett *result, const countt N);
	//result = X - aY
	void __global__ xMinusAlfaYKernel(rett *X, rett *Y, rett* aScalar,rett *result, const countt N);
	//result = X - Y
	void __global__ xSubYKernel(rett *X, rett *Y,rett *result, const countt N);
	//result = scalar * v
	void __global__ myltiplyVectorOnScalarKernel(rett *v, rett *scalar,rett *result, const countt N);

//Умножение матрицы на вектор
	//myltiply matrix on vector (only less then total GRAM)
	void __global__ matrixOnVectorMultiplyKernel(rett *A, rett *b, rett *result, const countt N);
	//myltiply part of  matrix on vector (for part of a large matrix)
	void __global__ matrixPartOnVectorMultiplyKernel(rett *partA, rett *b, rett *result, const countt N, const countt partNumber, const countt partSize);
	//multiply strip matrix on vector
	void __global__ bandMatrixOnVectorMultiplyKernel(rett *bandA, rett *b, rett *result,const countt N, const countt B);

//2 функции в связке реализующие скаларное произведение
	/*
	Сначала 2 вектора почленно умножаются и частично суммируются
	Потом вектор частичных сумм суммируется в скаляр
	*/
void __global__ multiplyVectorsAndPartialSumKernel(rett* a, rett* b, rett* resultAfterReductionGridSize, const countt N);
void __global__ reductionSumAtSingleBlockKernel(rett *input, rett *rScalar, const countt n);

//Специальное ядро для скаларного произведения (второй шаг)
//Делит результат суммирования вектора частичных сумм на divScalar
void __global__ reductionSumAtSingleBlockSpecialKernelWithDivide(rett *input, rett *rScalar, const countt n, rett *divScalar);

void __global__ matrixOnMatrixMultiplyKernel(rett* a, rett* b, rett* c, const countt N);
__global__ void matrixOnMatrixMultiplyKernel1( float * a, float * b, int n, float * c );