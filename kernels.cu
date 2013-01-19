#include <cuda_runtime_api.h>
#include "kernels.h"

/*
 *  опирует один массив в другой вне зависимости от размерности грида
 */
void __global__ copyKernel(rett *src, rett *dst, const countt N){
    countt globalIndex = threadIdx.x + blockIdx.x*blockDim.x;
	countt globalLimit = gridDim.x*blockDim.x;
	countt count = (countt)(N/globalLimit);
	countt index = globalIndex;
	for (countt pass = 0; pass < count; pass++ ){
		dst[index] = src[index];
		index += globalLimit;
	}
    if(index < N)
        dst[index] = src[index];
}

void __global__ myltiplyVectorOnScalarKernel(rett *v, rett *scalar,rett *result, const countt N){
    countt globalIndex = threadIdx.x + blockIdx.x*blockDim.x;
	countt globalLimit = gridDim.x*blockDim.x;
	countt count = (countt)(N/globalLimit);
	countt index = globalIndex;
	for (countt pass = 0; pass < count; pass++ ){
		result[index] = (*scalar)*v[index];
		index += globalLimit;
	}
    if(index < N)
        result[index] = (*scalar)*v[index];
}

/*
 * —кал€рное произведение с частичным суммированием
 * ¬ выходной массив размерности равной размерности грида пишетс€ частично просуммированный р€д
 * !!! об€зательно просуммировать оставшуюс€ часть (редукци€ или процессор?)
 * при формировании грида желательно уменьшить колличество блоков (выходной массив будет короче)
 */
void __global__ multiplyVectorsAndPartialSumKernel(rett* a, rett* b, rett* resultAfterReductionGridSize, const countt N){

	countt globalIndex = threadIdx.x + blockIdx.x*blockDim.x;
	countt globalLimit = gridDim.x*blockDim.x;
	countt count = (countt)(N/globalLimit);
	rett __shared__ result[THREADS_PER_BLOCK];

	result[threadIdx.x] = 0;

	countt index = globalIndex;
	for (countt pass = 0; pass < count; pass++ ){
		result[threadIdx.x] += a[index]*b[index];
		index += globalLimit;
	}

    if(index < N)
        result[threadIdx.x] += a[index]*b[index];

	__syncthreads();
	//reduction of block part
	int n = blockDim.x/2;
	while(n != 0){
		if (threadIdx.x < n){
			result[threadIdx.x] += result[threadIdx.x + n];
		}
		__syncthreads();
		n/=2;
	}
	//writing to output part
	if (threadIdx.x == 0)
		resultAfterReductionGridSize[blockIdx.x] = result[0];

}

/*
 * «апускать только на 1м блоке
 * ¬ рамках одного блока суммирует небольшие р€ды
 */
void __global__ reductionSumAtSingleBlockKernel(rett *input, rett *rScalar, const countt n){
	rett __shared__ result[THREADS_PER_BLOCK];
	result[threadIdx.x] = 0;

	for(countt indexAdd = 0; indexAdd < n; indexAdd += blockDim.x){
		countt index = threadIdx.x + indexAdd;
		if ( index < n ){
			result[threadIdx.x] += input[index];
		}
	}
	__syncthreads();

	int m = blockDim.x/2;
	while(m != 0){
		if (threadIdx.x < m){
			result[threadIdx.x] += result[threadIdx.x + m];
		}
		__syncthreads();
		m/=2;
	}

	if (threadIdx.x == 0)
		rScalar[0] = result[0];
}

void __global__ stripMatrixOnVectorMultiplyKernel(rett *stripA, rett *b, rett *result,const countt N, const countt B){
	rett __shared__ blockResult[THREADS_PER_BLOCK];
	countt BB = B + 1;
	countt globalLimit = gridDim.x*blockDim.x;
	for (countt I = blockIdx.x*(THREADS_PER_BLOCK) + threadIdx.x; I < N; I += globalLimit) {
		blockResult[threadIdx.x] = stripA[I*BB + B]*b[I];
		for (countt k = 1; k <= B; k++){
			blockResult[threadIdx.x] += stripA[I*BB + B - k]*b[I - k];
			blockResult[threadIdx.x] += stripA[(I + k)*BB + B - k]*b[I + k];
		}

		result[I] = blockResult[threadIdx.x];
	}
}

void __global__ matrixOnVectorMultiplyKernel(rett *A, rett *b, rett *result, const countt N){
	__shared__ rett res[THREADS_PER_BLOCK];

	for(countt blStep = 0; blStep < N; blStep += gridDim.x){//!
		res[threadIdx.x] = 0;
		countt block_i = blockIdx.x + blStep;
		if ( (block_i) < N){
			for (countt thStep = 0; thStep < N; thStep += blockDim.x){
				countt thread_j = threadIdx.x + thStep;
				if ( (thread_j) < N){
					res[threadIdx.x] += \
						A[ (block_i)*N + (thread_j) ] *\
						b[ thread_j ];
				}
			}
		}
		__syncthreads();

		int n = blockDim.x/2;
		while(n != 0){
			if (threadIdx.x < n){
				res[threadIdx.x] += res[threadIdx.x + n];
			}
			__syncthreads();
			n/=2;
		}
		if (threadIdx.x == 0){
			result[blockIdx.x + blStep] = res[threadIdx.x];
		}
		__syncthreads();
	}
}

/*
 * A - часть (слой) матрицы тольщины partSize (строки)
 * partNumber - номер такого сло€
 * b - передаетс€ полностью
 * result - передаетс€ полностью но заполн€етс€ в соответствии со слоем матрицы
 *
 * в целом метод надо вызывать несколько раз, перезаполн€€ слой матрицы
 */

void __global__ matrixPartOnVectorMultiplyKernel(rett *partA, rett *b, rett *result, const countt N, const countt partNumber, const countt partSize){
	__shared__ rett res[THREADS_PER_BLOCK];

	for(countt blStep = 0; blStep < partSize; blStep += gridDim.x){//!
		res[threadIdx.x] = 0;
		countt block_i = blockIdx.x + blStep;
		if (block_i < partSize){
			for (countt thStep = 0; thStep < N; thStep += blockDim.x){
				countt thread_j = threadIdx.x + thStep;
				if (thread_j < N){
					res[threadIdx.x] += \
						partA[ (block_i)*N + (thread_j) ] *\
						b[ thread_j ];
				}
			}
		}
		__syncthreads();

		int n = blockDim.x/2;
		while(n != 0){
			if (threadIdx.x < n){
				res[threadIdx.x] += res[threadIdx.x + n];
			}
			__syncthreads();
			n/=2;
		}
		if (threadIdx.x == 0){
			result[ \
				partSize*partNumber + blockIdx.x + blStep\
			] = res[threadIdx.x];
		}
		__syncthreads();
	}
}