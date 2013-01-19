#include <cuda_runtime_api.h>
#include "cuda_math.h"
#include "kernels.h"

extern "C" __declspec(dllexport) float __stdcall getDeviceInfo(DeviceInfo &di){
    printf("Execution getDeviceInfo->\n");
	int deviceCount;
	cudaDeviceProp deviceProp;
	cudaGetDeviceCount(&deviceCount);
	
	di.deviceCount = deviceCount;
	
	//printf("Device count: %d\n\n", deviceCount);
	for (int i = 0; i < deviceCount; i++){
		cudaGetDeviceProperties(&deviceProp, i);

		di.totalGlobalMem[i] = (int)deviceProp.totalGlobalMem;
		di.sharedMemPerBlock[i] = (int)deviceProp.sharedMemPerBlock;
		di.maxThreadsPerBlock[i] = (int)deviceProp.maxThreadsPerBlock;
/*		
		printf("Device name: %s\n", deviceProp.name);
		printf("Total global memory: %d\n", deviceProp.totalGlobalMem);
		printf("Shared memory per block: %d\n", deviceProp.sharedMemPerBlock);
		printf("Registers per block: %d\n", deviceProp.regsPerBlock);
		printf("Warp size: %d\n", deviceProp.warpSize);
		printf("Memory pitch: %d\n", deviceProp.memPitch);
		printf("Max threads per block: %d\n", deviceProp.maxThreadsPerBlock);

		printf("Max threads dimensions: x = %d, y = %d, z = %d\n",
		deviceProp.maxThreadsDim[0],
		deviceProp.maxThreadsDim[1],
		deviceProp.maxThreadsDim[2]);

		printf("Max grid size: x = %d, y = %d, z = %d\n", 
		deviceProp.maxGridSize[0], 
		deviceProp.maxGridSize[1], 
		deviceProp.maxGridSize[2]); 

		printf("Clock rate: %d\n", deviceProp.clockRate);
		printf("Total constant memory: %d\n", deviceProp.totalConstMem); 
		printf("Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
		printf("Texture alignment: %d\n", deviceProp.textureAlignment);
		printf("Device overlap: %d\n", deviceProp.deviceOverlap);
		printf("Multiprocessor count: %d\n", deviceProp.multiProcessorCount);

		printf("Kernel execution timeout enabled: %s\n",
		deviceProp.kernelExecTimeoutEnabled ? "true" : "false");
*/
	}
    return 0;
}


extern "C" __declspec(dllexport) float __stdcall stripMatrixOnVectorMultiply(rett *stripA, rett *b, rett *result,const countt N, const countt B){
	//allocation memory on device
	rett *devStripMatrixA;
		CUDA_CHECK_ERROR(cudaMalloc((void**)&devStripMatrixA, N*(2*B + 1)*sizeof(rett)));
	rett *devVector_b;
		CUDA_CHECK_ERROR(cudaMalloc((void**)&devVector_b, N*sizeof(rett)))
	rett *devResult;
		CUDA_CHECK_ERROR(cudaMalloc((void**)&devResult, N*sizeof(rett)))
	//initialization data on device
	CUDA_CHECK_ERROR(cudaMemcpy(devStripMatrixA, stripA, N*(2*B + 1)*sizeof(rett), cudaMemcpyHostToDevice));
	CUDA_CHECK_ERROR(cudaMemcpy(devVector_b, b, N*sizeof(rett), cudaMemcpyHostToDevice));
	//registration of events
	cudaEvent_t start;
    cudaEvent_t stop;
	//create events to sync and get timeout
	CUDA_CHECK_ERROR(cudaEventCreate(&start));
	CUDA_CHECK_ERROR(cudaEventCreate(&stop));
	//point of start GPU calc
	cudaEventRecord(start, 0);
	//init grid parametres
	dim3 gridSizeForStripMatrixMult = dim3(MAX_BLOCKS/THREADS_PER_BLOCK, 1, 1);
	dim3 blockSizeForStripMatrixMult = dim3(THREADS_PER_BLOCK, 1, 1);
	//call of kernel
	stripMatrixOnVectorMultiplyKernel<<< gridSizeForStripMatrixMult, blockSizeForStripMatrixMult >>>(devStripMatrixA, devVector_b, devResult, N, B);
	//point of end calculation
	cudaEventRecord(stop, 0);

	float time = 0;
	//sunc all threads
    cudaEventSynchronize(stop);
	//get execution time
    cudaEventElapsedTime(&time, start, stop);
	//copy from device
	CUDA_CHECK_ERROR(cudaMemcpy(result, devResult, sizeof(rett)*N, cudaMemcpyDeviceToHost));
	//free allocated GPU memory
	CUDA_CHECK_ERROR(cudaFree(devStripMatrixA));
	CUDA_CHECK_ERROR(cudaFree(devVector_b));
	CUDA_CHECK_ERROR(cudaFree(devResult));
	//destroy of events
	CUDA_CHECK_ERROR(cudaEventDestroy(start));
	CUDA_CHECK_ERROR(cudaEventDestroy(stop));
	return time;
}

extern "C" __declspec(dllexport) float __stdcall matrixOnVectorMultiply(rett *A, rett *b, rett *result,const countt N){
	//allocation memory on device
	rett *devMatrixA;
		CUDA_CHECK_ERROR(cudaMalloc((void**)&devMatrixA, N*N*sizeof(rett)));
	rett *devVector_b;
		CUDA_CHECK_ERROR(cudaMalloc((void**)&devVector_b, N*sizeof(rett)))
	rett *devResult;
		CUDA_CHECK_ERROR(cudaMalloc((void**)&devResult, N*sizeof(rett)))
	//initialization data on device
	CUDA_CHECK_ERROR(cudaMemcpy(devMatrixA, A, N*N*sizeof(rett), cudaMemcpyHostToDevice));
	CUDA_CHECK_ERROR(cudaMemcpy(devVector_b, b, N*sizeof(rett), cudaMemcpyHostToDevice));
	//registration of events
	cudaEvent_t start;
    cudaEvent_t stop;
	//create events to sync and get timeout
	CUDA_CHECK_ERROR(cudaEventCreate(&start));
	CUDA_CHECK_ERROR(cudaEventCreate(&stop));
	//point of start GPU calc
	cudaEventRecord(start, 0);
	//init grid parametres
	dim3 gridSize = dim3(MAX_BLOCKS/THREADS_PER_BLOCK, 1, 1);
	dim3 blockSize = dim3(THREADS_PER_BLOCK, 1, 1);
	//call of kernel
	matrixOnVectorMultiplyKernel<<< gridSize, blockSize >>>(devMatrixA, devVector_b, devResult, N);
	//point of end calculation
	cudaEventRecord(stop, 0);

	float time = 0;
	//sunc all threads
    cudaEventSynchronize(stop);
	//get execution time
    cudaEventElapsedTime(&time, start, stop);
	//copy from device
	CUDA_CHECK_ERROR(cudaMemcpy(result, devResult, sizeof(rett)*N, cudaMemcpyDeviceToHost));
	//free allocated GPU memory
	CUDA_CHECK_ERROR(cudaFree(devMatrixA));
	CUDA_CHECK_ERROR(cudaFree(devVector_b));
	CUDA_CHECK_ERROR(cudaFree(devResult));
	//destroy of events
	CUDA_CHECK_ERROR(cudaEventDestroy(start));
	CUDA_CHECK_ERROR(cudaEventDestroy(stop));
	return time;
}

extern "C" __declspec(dllexport) float __stdcall matrixPartOnVectorMultiply(rett *A, rett *b, rett *result,const countt N, const countt partNumber, const countt partSize){
	//allocation memory on device
	rett *devMatrixA;
		CUDA_CHECK_ERROR(cudaMalloc((void**)&devMatrixA, N*partSize*sizeof(rett)));
	rett *devVector_b;
		CUDA_CHECK_ERROR(cudaMalloc((void**)&devVector_b, N*sizeof(rett)))
	rett *devResult;
		CUDA_CHECK_ERROR(cudaMalloc((void**)&devResult, N*sizeof(rett)))
	//initialization data on device
	CUDA_CHECK_ERROR(cudaMemcpy(devMatrixA, A, N*partSize*sizeof(rett), cudaMemcpyHostToDevice));
	CUDA_CHECK_ERROR(cudaMemcpy(devVector_b, b, N*sizeof(rett), cudaMemcpyHostToDevice));
	CUDA_CHECK_ERROR(cudaMemcpy(devResult, result, N*sizeof(rett), cudaMemcpyHostToDevice));
	//registration of events
	cudaEvent_t start;
    cudaEvent_t stop;
	//create events to sync and get timeout
	CUDA_CHECK_ERROR(cudaEventCreate(&start));
	CUDA_CHECK_ERROR(cudaEventCreate(&stop));
	//point of start GPU calc
	cudaEventRecord(start, 0);
	//init grid parametres
	dim3 gridSize = dim3(MAX_BLOCKS/THREADS_PER_BLOCK, 1, 1);
	dim3 blockSize = dim3(THREADS_PER_BLOCK, 1, 1);
	//call of kernel
	matrixPartOnVectorMultiplyKernel<<< gridSize, blockSize >>>(devMatrixA, devVector_b, devResult, N , partNumber, partSize);
	//point of end calculation
	cudaEventRecord(stop, 0);

	float time = 0;
	//sunc all threads
    cudaEventSynchronize(stop);
	//get execution time
    cudaEventElapsedTime(&time, start, stop);
	//copy from device
	CUDA_CHECK_ERROR(cudaMemcpy(result, devResult, sizeof(rett)*N, cudaMemcpyDeviceToHost));
	//free allocated GPU memory
	CUDA_CHECK_ERROR(cudaFree(devMatrixA));
	CUDA_CHECK_ERROR(cudaFree(devVector_b));
	CUDA_CHECK_ERROR(cudaFree(devResult));
	//destroy of events
	CUDA_CHECK_ERROR(cudaEventDestroy(start));
	CUDA_CHECK_ERROR(cudaEventDestroy(stop));
	return time;
}

extern "C" __declspec(dllexport) float __stdcall skalarMultiply(rett *a, rett *b, rett *result,const countt N){
	//allocation memory on device
	rett *devVectorA;
		CUDA_CHECK_ERROR(cudaMalloc((void**)&devVectorA, N*sizeof(rett)));
	rett *devVectorB;
		CUDA_CHECK_ERROR(cudaMalloc((void**)&devVectorB, N*sizeof(rett)));
	rett *devMetaResult;
		const countt DEV_META_RESULT_SIZE = MAX_BLOCKS/THREADS_PER_BLOCK;//колличество блоков на 1м шаге
		CUDA_CHECK_ERROR(cudaMalloc((void**)&devMetaResult, DEV_META_RESULT_SIZE*sizeof(rett)));
	rett *devResultScalar;
		CUDA_CHECK_ERROR(cudaMalloc((void**)&devResultScalar, sizeof(rett)));
	//initialization data on device
	CUDA_CHECK_ERROR(cudaMemcpy(devVectorA, a, N*sizeof(rett), cudaMemcpyHostToDevice));
	CUDA_CHECK_ERROR(cudaMemcpy(devVectorB, b, N*sizeof(rett), cudaMemcpyHostToDevice));
	//registration of events
	cudaEvent_t start;
    cudaEvent_t stop;
	//create events to sync and get timeout
	CUDA_CHECK_ERROR(cudaEventCreate(&start));
	CUDA_CHECK_ERROR(cudaEventCreate(&stop));
	//point of start GPU calc
	cudaEventRecord(start, 0);
	//init grid parametres on Step 1
	dim3 gridSizeStep1 = dim3(DEV_META_RESULT_SIZE, 1, 1);
	dim3 blockSizeStep1 = dim3(THREADS_PER_BLOCK, 1, 1);
	//call of kernel
	multiplyVectorsAndPartialSumKernel<<< gridSizeStep1, blockSizeStep1 >>>(devVectorA, devVectorB, devMetaResult, N);
	//init grid parametres on Step 2
	dim3 gridSizeStep2 = dim3(1, 1, 1);
	dim3 blockSizeStep2 = dim3(THREADS_PER_BLOCK, 1, 1);
	//call of kernel (tested)
	reductionSumAtSingleBlockKernel<<< gridSizeStep2, blockSizeStep2 >>>(devMetaResult,devResultScalar,DEV_META_RESULT_SIZE);
	//point of end calculation
	cudaEventRecord(stop, 0);

	float time = 0;
	//sunc all threads
    cudaEventSynchronize(stop);
	//get execution time
    cudaEventElapsedTime(&time, start, stop);
	//copy from device

	CUDA_CHECK_ERROR(cudaMemcpy(result, devResultScalar, sizeof(rett), cudaMemcpyDeviceToHost));
//	CUDA_CHECK_ERROR(cudaMemcpy(result, devMetaResult, sizeof(rett), cudaMemcpyDeviceToHost));
	//free allocated GPU memory
	CUDA_CHECK_ERROR(cudaFree(devVectorA));
	CUDA_CHECK_ERROR(cudaFree(devVectorB));
	CUDA_CHECK_ERROR(cudaFree(devMetaResult));
	CUDA_CHECK_ERROR(cudaFree(devResultScalar));
	//destroy of events
	CUDA_CHECK_ERROR(cudaEventDestroy(start));
	CUDA_CHECK_ERROR(cudaEventDestroy(stop));
	return time;
}

extern "C" __declspec(dllexport) float __stdcall myltiplyVectorOnScalar(rett *v, rett *scalar,rett *result, const countt N){
	//allocation memory on device
	rett *devVector;
		CUDA_CHECK_ERROR(cudaMalloc((void**)&devVector, N*sizeof(rett)));
	rett *devResultVector;
		CUDA_CHECK_ERROR(cudaMalloc((void**)&devResultVector, N*sizeof(rett)));
	rett *devScalar;
		CUDA_CHECK_ERROR(cudaMalloc((void**)&devScalar, sizeof(rett)));
	//initialization data on device
	CUDA_CHECK_ERROR(cudaMemcpy(devVector, v, N*sizeof(rett), cudaMemcpyHostToDevice));
	CUDA_CHECK_ERROR(cudaMemcpy(devScalar, scalar, sizeof(rett), cudaMemcpyHostToDevice));
	//registration of events
	cudaEvent_t start;
    cudaEvent_t stop;
	//create events to sync and get timeout
	CUDA_CHECK_ERROR(cudaEventCreate(&start));
	CUDA_CHECK_ERROR(cudaEventCreate(&stop));
	//point of start GPU calc
	cudaEventRecord(start, 0);
	//init grid parametres on Step 1
	dim3 gridSizeStep = dim3(MAX_BLOCKS/THREADS_PER_BLOCK, 1, 1);
	dim3 blockSizeStep = dim3(THREADS_PER_BLOCK, 1, 1);
	//call of kernel
	myltiplyVectorOnScalarKernel<<< gridSizeStep, blockSizeStep >>>(devVector, devScalar, devResultVector, N);
	//init grid parametres on Step 2
	cudaEventRecord(stop, 0);

	float time = 0;
	//sunc all threads
    cudaEventSynchronize(stop);
	//get execution time
    cudaEventElapsedTime(&time, start, stop);
	//copy from device

	CUDA_CHECK_ERROR(cudaMemcpy(result, devResultVector, N*sizeof(rett), cudaMemcpyDeviceToHost));
	//free allocated GPU memory
	CUDA_CHECK_ERROR(cudaFree(devVector));
	CUDA_CHECK_ERROR(cudaFree(devScalar));
	CUDA_CHECK_ERROR(cudaFree(devResultVector));
	//destroy of events
	CUDA_CHECK_ERROR(cudaEventDestroy(start));
	CUDA_CHECK_ERROR(cudaEventDestroy(stop));
	return time;
}

extern "C" __declspec(dllexport) float __stdcall methodConjugateGradient(rett *matrixA, rett *vectorB, rett *vectorX, const countt N, const countt B){
	float time = 0;
	return time;
}