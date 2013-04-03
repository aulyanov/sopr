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
	__syncthreads();
}

//error X = F(X) ????
void __global__ xPlusAlfaYKernel(rett *X, rett *Y, rett* aScalar,rett *result, const countt N){
    countt globalIndex = threadIdx.x + blockIdx.x*blockDim.x;
	countt globalLimit = gridDim.x*blockDim.x;

	countt count = (countt)(N/globalLimit);
	countt index = globalIndex;
	rett a = (*aScalar);
	for (countt pass = 0; pass < count; pass++ ){
		rett temp =  a*Y[index] + X[index];
		result[index] = temp;
		index += globalLimit;
	}
	
	if(index < N){
        rett temp =  a*Y[index] + X[index];
		result[index] = temp;
	}
	__syncthreads();
}

void __global__ xMinusAlfaYKernel(rett *X, rett *Y, rett* aScalar,rett *result, const countt N){
    countt globalIndex = threadIdx.x + blockIdx.x*blockDim.x;
	countt globalLimit = gridDim.x*blockDim.x;
	countt count = (countt)(N/globalLimit);
	rett a = (*aScalar);
	countt index = globalIndex;
	for (countt pass = 0; pass < count; pass++ ){
		result[index] = X[index] - a*Y[index];
		index += globalLimit;
	}
    if(index < N)
        result[index] = X[index] - a*Y[index];
	__syncthreads();
}

void __global__ xSubYKernel(rett *X, rett *Y,rett *result, const countt N){
    countt globalIndex = threadIdx.x + blockIdx.x*blockDim.x;
	countt globalLimit = gridDim.x*blockDim.x;
	countt count = (countt)(N/globalLimit);
	countt index = globalIndex;
	for (countt pass = 0; pass < count; pass++ ){
		result[index] = X[index] - Y[index];
		index += globalLimit;
	}
    if(index < N)
        result[index] = X[index] - Y[index];
	__syncthreads();
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
	__syncthreads();
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
	__syncthreads();
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
	__syncthreads();
}

/*
 * «апускать только на 1м блоке
 * ¬ рамках одного блока суммирует небольшие р€ды
 * ƒелит результат на divScalar
 */
void __global__ reductionSumAtSingleBlockSpecialKernelWithDivide(rett *input, rett *rScalar, const countt n, rett *divScalar){
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
	while(m > 0){
		if (threadIdx.x < m){
			result[threadIdx.x] += result[threadIdx.x + m];
		}
		__syncthreads();
		m/=2;
	}

	if (threadIdx.x == 0){
		(*rScalar) = (*result)/(*divScalar);//*pow((*divScalar),-1);
	}
	__syncthreads();
}


// TODO: fix NAN
void __global__ bandMatrixOnVectorMultiplyKernel(rett *bandA, rett *b, rett *result,const countt N, const countt B){
	rett __shared__ blockResult[THREADS_PER_BLOCK];
	countt BB = B + 1;
	countt globalLimit = gridDim.x*blockDim.x;
	for (countt I = blockIdx.x*(THREADS_PER_BLOCK) + threadIdx.x; I < N; I += globalLimit) {
		blockResult[threadIdx.x] = bandA[I*BB + B]*b[I];
		for (countt k = 1; k <= B; k++){
			if ( ( (I - k) < N ) && ( (I*BB + B - k) < N*(B+1) ) )
				blockResult[threadIdx.x] += bandA[I*BB + B - k]*b[I - k];//!
			if ((I + k) < N)
				blockResult[threadIdx.x] += bandA[(I + k)*BB + B - k]*b[I + k];
		}

		result[I] = blockResult[threadIdx.x];
	}
	__syncthreads();
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
	__syncthreads();
}

extern "C" __declspec(dllexport) float __stdcall matrixOnMatrixMultiply(rett* A, rett* B, const countt N){
	return 0;
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
	__syncthreads();
}

// peremnogenie obichnih matric

void __global__ matrixOnMatrixMultiplyKernel(rett* a, rett* b, rett* c, const countt N){
	__shared__ rett strA[THREADS_PER_BLOCK];
	__shared__ rett colB[THREADS_PER_BLOCK];
	__shared__ rett resToReduction[THREADS_PER_BLOCK];
	
	countt globalLimit = blockDim.x*gridDim.x;

	for (countt colIndex = blockIdx.x; colIndex < N; colIndex += blockDim.x){
		for (countt rowIndex = 0; rowIndex < N; rowIndex ++){
			//rowIndex, colIndex
			//mult row on col
			resToReduction[threadIdx.x] = 0;

			for (countt runThIndex = threadIdx.x; runThIndex < N; runThIndex += blockDim.x ){
				strA[runThIndex] = a[rowIndex*N + runThIndex];// copy matrix string
				colB[runThIndex] = b[runThIndex*N + colIndex];// copy matrix column with run index - j
				resToReduction[threadIdx.x] += strA[runThIndex]*colB[runThIndex];
			}
			__syncthreads();
			//reduction -->

			int n = blockDim.x/2;
			while(n != 0){
				if (threadIdx.x < n){
					resToReduction[threadIdx.x] += resToReduction[threadIdx.x + n];
				}
				__syncthreads();
				n/=2;
			}

			if (threadIdx.x == 0)
				c[rowIndex*N + colIndex] = resToReduction[0];
		}
	}
}

__global__ void matrixOnMatrixMultiplyKernel1 ( float * a, float * b, int n, float * c ){
    int bx = blockIdx.x;        // block index
    int by = blockIdx.y;

    int tx = threadIdx.x;       // thread index
    int ty = threadIdx.y;

	int dp = THREADS_PER_SQUARE_BLOCK - (THREADS_PER_SQUARE_BLOCK*gridDim.x - n);

    int aBegin = n*THREADS_PER_SQUARE_BLOCK*by;
    int aStep = THREADS_PER_SQUARE_BLOCK;

	int bBegin = THREADS_PER_SQUARE_BLOCK*bx;
    int bStep = THREADS_PER_SQUARE_BLOCK * n;

	int aEnd = aBegin + n - 1;

    float sum = 0.0f;           // computed subelement
    for ( int ia = aBegin, ib = bBegin, p = 0; ia <= aEnd; ia += aStep, ib += bStep, p ++){
        __shared__ float as [THREADS_PER_SQUARE_BLOCK][THREADS_PER_SQUARE_BLOCK];
        __shared__ float bs [THREADS_PER_SQUARE_BLOCK][THREADS_PER_SQUARE_BLOCK];

		if (p == gridDim.x - 1){
			if (tx < dp){
				as[ty][tx] = a[ia + n*ty + tx];
			}else{
				as[ty][tx] = 0;
			}
			if (ty < dp){
				bs[ty][tx] = b[ib + n*ty + tx];
			}else{
				bs[ty][tx] = 0;
			}
		}else{
			as[ty][tx] = a[ia + n*ty + tx];
			bs[ty][tx] = b[ib + n*ty + tx];
		}

        __syncthreads();
        for ( int k = 0; k < THREADS_PER_SQUARE_BLOCK; k++ )
            sum += as[ty][k]*bs[k][tx];
        __syncthreads();
    }
    int ic = n * THREADS_PER_SQUARE_BLOCK * by + THREADS_PER_SQUARE_BLOCK * bx;

	if ( bx == gridDim.x - 1 && by != gridDim.y - 1 ){
		if (tx < dp){
			c [ic + n*ty + tx] = sum;
		}
	}
	if ( bx != gridDim.x - 1 && by == gridDim.y - 1 ){
		if (ty < dp){
			c [ic + n*ty + tx] = sum;
		}
	}
	if ( bx == gridDim.x - 1 && by == gridDim.y - 1 ){
		if (tx < dp && ty < dp){
			c [ic + n*ty + tx] = sum;
		}
	}

	if ( bx != gridDim.x - 1 && by != gridDim.y - 1 ){
		c [ic + n*ty + tx] = sum;
	}
}