#include <cuda_runtime_api.h>
#include "kernels.h"

void __global__ copyKernel(rett *src, rett *dst, const int n){
    int globalIndex = threadIdx.x + blockIdx.x*blockDim.x;
    if(globalIndex < n){
        dst[globalIndex] = src[globalIndex];
    }
}

void __global__ multVectorOnSkalarKernel(rett *A, rett *k, const int N){
    int globalIndex = threadIdx.x + blockIdx.x*blockDim.x;
    if(globalIndex < N){
        rett temp = A[globalIndex];
        temp *= k[0];
        A[globalIndex] = temp;
    }
}

void __global__ skalarMultInSingleBlockKernel(rett *a_Global, rett *b_Global, rett *res_Single, const int N_Global){
    /*const int cache_size = SHARED_MEMORY_SIZE_BYTE/sizeof(rett);
    __shared__ rett cache_res[cache_size];
    
    int countOfGlobalLoad = (int)((N_Mass + cache_size - 1)/cache_size);
    int count = 0;
    while (count < countOfGlobalLoad){
		for(int i = 0; i < cache_size; i++){
			cache_a[i] = a_Global[count];
		}
		count ++;
    }
    (*res_Single) = 0;*/
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
 * partNumber - номер такого слоя
 * b - передается полностью
 * result - передается полностью но заполняется в соответствии со слоем матрицы
 *
 * в целом метод надо вызывать несколько раз, перезаполняя слой матрицы

 *TO-DO: пока ничего не сделано (надо ли вообще)

 */

void __global__ matrixPartOnVectorMultiplyKernel(rett *partA, rett *b, rett *result, const countt N, const countt partNumber, const countt partSize){
	__shared__ rett res[THREADS_PER_BLOCK];

	for(countt blStep = 0; blStep < N; blStep += gridDim.x){//!
		res[threadIdx.x] = 0;
		countt block_i = blockIdx.x + blStep;
		if ( (block_i) < N){
			for (countt thStep = 0; thStep < N; thStep += blockDim.x){
				countt thread_j = threadIdx.x + thStep;
				if ( (thread_j) < N){
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
			result[blockIdx.x + blStep] = res[threadIdx.x];
		}
		__syncthreads();
	}
}