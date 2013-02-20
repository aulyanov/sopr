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


extern "C" __declspec(dllexport) float __stdcall bandMatrixOnVectorMultiply(rett *stripA, rett *b, rett *result,const countt N, const countt B){
	//allocation memory on device
	rett *devStripMatrixA;
		CUDA_CHECK_ERROR(cudaMalloc((void**)&devStripMatrixA, N*(B + 1)*sizeof(rett)));
	rett *devVector_b;
		CUDA_CHECK_ERROR(cudaMalloc((void**)&devVector_b, N*sizeof(rett)))
	rett *devResult;
		CUDA_CHECK_ERROR(cudaMalloc((void**)&devResult, N*sizeof(rett)))
	//initialization data on device
	CUDA_CHECK_ERROR(cudaMemcpy(devStripMatrixA, stripA, N*(B + 1)*sizeof(rett), cudaMemcpyHostToDevice));
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
	bandMatrixOnVectorMultiplyKernel<<< gridSizeForStripMatrixMult, blockSizeForStripMatrixMult >>>(devStripMatrixA, devVector_b, devResult, N, B);
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

extern "C" __declspec(dllexport) float __stdcall methodConjugateGradientForBandMatrix(rett *matrixA, rett *vectorB, rett *vectorX, const countt N, const countt B, rett eps){
//allocation memory on device
	//Alloc for input data
	rett *devVectorB;
		CUDA_CHECK_ERROR(cudaMalloc((void**)&devVectorB, N*sizeof(rett)));
	rett *devVectorX;
		CUDA_CHECK_ERROR(cudaMalloc((void**)&devVectorX, N*sizeof(rett)));
	rett *devMatrixA;
		CUDA_CHECK_ERROR(cudaMalloc((void**)&devMatrixA, N*(B + 1)*sizeof(rett)));
	//Alloc for meta data
		//Scalars
		rett *devAlfaScalar;
			CUDA_CHECK_ERROR(cudaMalloc((void**)&devAlfaScalar, sizeof(rett)));
		rett *devBetaScalar;
			CUDA_CHECK_ERROR(cudaMalloc((void**)&devBetaScalar, sizeof(rett)));
		rett *devEps;
			CUDA_CHECK_ERROR(cudaMalloc((void**)&devEps, sizeof(rett)));
		//Temptory scalar
		rett *devDividerScalar; // for (P,Q)
			CUDA_CHECK_ERROR(cudaMalloc((void**)&devDividerScalar, sizeof(rett)));
		//Vectors
		rett *devR;
			CUDA_CHECK_ERROR(cudaMalloc((void**)&devR, N*sizeof(rett)));
		rett *devP;
			CUDA_CHECK_ERROR(cudaMalloc((void**)&devP, N*sizeof(rett)));
		rett *devQ;
			CUDA_CHECK_ERROR(cudaMalloc((void**)&devQ, N*sizeof(rett)));
		//Temptory vectors
		rett *devVectorTemp1;
			CUDA_CHECK_ERROR(cudaMalloc((void**)&devVectorTemp1, N*sizeof(rett)));
		rett *devMetaScalarResult;
		const countt DEV_META_RESULT_SIZE = MAX_BLOCKS/THREADS_PER_BLOCK;//колличество блоков на 1м шаге скалярного произведения
			CUDA_CHECK_ERROR(cudaMalloc((void**)&devMetaScalarResult, DEV_META_RESULT_SIZE*sizeof(rett)));
	//initialization data on device
	CUDA_CHECK_ERROR(cudaMemcpy(devVectorB, vectorB, N*sizeof(rett), cudaMemcpyHostToDevice));
	//Задаем начальное приближение
	CUDA_CHECK_ERROR(cudaMemcpy(devVectorX, vectorX, N*sizeof(rett), cudaMemcpyHostToDevice));
	CUDA_CHECK_ERROR(cudaMemcpy(devMatrixA, matrixA, N*(B + 1)*sizeof(rett), cudaMemcpyHostToDevice));
	CUDA_CHECK_ERROR(cudaMemcpy(devEps, &eps, sizeof(rett), cudaMemcpyHostToDevice));
	//registration of events
	cudaEvent_t start,stop,s1,s2,s3;
	//create events to sync and get timeout
	CUDA_CHECK_ERROR(cudaEventCreate(&start));
	CUDA_CHECK_ERROR(cudaEventCreate(&stop));
	CUDA_CHECK_ERROR(cudaEventCreate(&s1));
	CUDA_CHECK_ERROR(cudaEventCreate(&s2));
	CUDA_CHECK_ERROR(cudaEventCreate(&s3));
	//point of start GPU calc
	cudaEventRecord(start, 0);
//Method step 1
//Вычисляем невязку на 0м шаге
	//init grid parametres
	dim3 gridSizeStandart = dim3(MAX_BLOCKS/THREADS_PER_BLOCK, 1, 1);
	dim3 blockSizeStandart = dim3(THREADS_PER_BLOCK, 1, 1);
	//A*X
	bandMatrixOnVectorMultiplyKernel<<< gridSizeStandart, blockSizeStandart >>>(devMatrixA, devVectorX, devVectorTemp1, N, B);
	cudaThreadSynchronize();
	//R0 = B - AX
	xSubYKernel<<< gridSizeStandart, blockSizeStandart >>>(devVectorB,devVectorTemp1,devR ,N);
	cudaThreadSynchronize();
	//P0 = R0
	copyKernel<<< gridSizeStandart, blockSizeStandart >>>(devR, devP, N);
	cudaThreadSynchronize();

for (countt k = 0; k < N; k++){
//Method step 2

	bandMatrixOnVectorMultiplyKernel<<< gridSizeStandart, blockSizeStandart >>>(devMatrixA, devP, devQ, N, B);
	cudaThreadSynchronize();
	//devAlfaScalar = (Rk,Pk) / (Qk,Pk)
		// 1. devDividerScalar = (Qk,Pk) //вычисляем знаменатель - далее он еще пригодится для devBetaScalar
		dim3 gridSizeStep1 = dim3(DEV_META_RESULT_SIZE, 1, 1);
		dim3 blockSizeStep1 = dim3(THREADS_PER_BLOCK, 1, 1);
			//call of kernel for step 1 of scalar mult
			multiplyVectorsAndPartialSumKernel<<< gridSizeStep1, blockSizeStep1 >>>(devQ, devP, devMetaScalarResult, N);
			cudaThreadSynchronize();
			//init grid parametres on Step 2
		dim3 gridSizeStep2 = dim3(1, 1, 1);
		dim3 blockSizeStep2 = dim3(THREADS_PER_BLOCK, 1, 1);
			//call of kernel for step 2 of scalar mult
			reductionSumAtSingleBlockKernel<<< gridSizeStep2, blockSizeStep2 >>>(devMetaScalarResult,devDividerScalar,DEV_META_RESULT_SIZE);
			cudaThreadSynchronize();
		// 2. devAlfaScalar = (R,P) / devDividerScalar
			multiplyVectorsAndPartialSumKernel<<< gridSizeStep1, blockSizeStep1 >>>(devR, devP, devMetaScalarResult, N);
			cudaThreadSynchronize();
			reductionSumAtSingleBlockSpecialKernelWithDivide<<< gridSizeStep2, blockSizeStep2 >>>(devMetaScalarResult,devAlfaScalar,DEV_META_RESULT_SIZE,devDividerScalar);
			cudaThreadSynchronize();
	//X = X + devAlfaScalar * P
		xPlusAlfaYKernel<<< gridSizeStandart, blockSizeStandart >>>(devVectorX,devP,devAlfaScalar,devVectorX,N);
		cudaThreadSynchronize();
	//R = R - devAlfaScalar * Q
		xMinusAlfaYKernel<<< gridSizeStandart, blockSizeStandart >>>(devR,devQ,devAlfaScalar,devR,N);
		cudaThreadSynchronize();
//Method step 3
	// |X| < eps ?
	// devBetaScalar = (R,Q) / (Q,P)
		// 1. devDividerScalar уже вычислен ранее (devDividerScalar == (Q,P))
		// 2. devBetaScalar = (R,Q) / devDividerScalar 
			multiplyVectorsAndPartialSumKernel<<< gridSizeStep1, blockSizeStep1 >>>(devR, devQ, devMetaScalarResult, N);
			cudaThreadSynchronize();
			reductionSumAtSingleBlockSpecialKernelWithDivide<<< gridSizeStep2, blockSizeStep2 >>>(devMetaScalarResult,devBetaScalar,DEV_META_RESULT_SIZE,devDividerScalar);
			cudaThreadSynchronize();
	// P = R - devBetaScalar * P (новое направление минимизации)
		xMinusAlfaYKernel<<< gridSizeStandart, blockSizeStandart >>>(devR,devP,devBetaScalar,devP,N);
		cudaThreadSynchronize();
}

	//point of end calculation
	cudaEventRecord(stop, 0);
	float time = 0;
	//sunc all threads
    cudaEventSynchronize(stop);
	//get execution time
    cudaEventElapsedTime(&time, start, stop);
	//copy from device
	CUDA_CHECK_ERROR(cudaMemcpy(vectorX, devVectorX, sizeof(rett)*N, cudaMemcpyDeviceToHost));
	//free allocated GPU memory
	CUDA_CHECK_ERROR(cudaFree(devMatrixA));
	CUDA_CHECK_ERROR(cudaFree(devVectorB));
	CUDA_CHECK_ERROR(cudaFree(devVectorX));
	CUDA_CHECK_ERROR(cudaFree(devAlfaScalar));
	CUDA_CHECK_ERROR(cudaFree(devBetaScalar));
	CUDA_CHECK_ERROR(cudaFree(devEps));
	CUDA_CHECK_ERROR(cudaFree(devR));
	CUDA_CHECK_ERROR(cudaFree(devP));
	CUDA_CHECK_ERROR(cudaFree(devQ));
	CUDA_CHECK_ERROR(cudaFree(devVectorTemp1));
	//destroy of events
	CUDA_CHECK_ERROR(cudaEventDestroy(start));
	CUDA_CHECK_ERROR(cudaEventDestroy(stop));
	return time;
}

extern "C" __declspec(dllexport) float __stdcall methodConjugateGradient(rett *matrixA, rett *vectorB, rett *vectorX, const countt N, rett eps){
//allocation memory on device
	//Alloc for input data
	rett *devVectorB;
		CUDA_CHECK_ERROR(cudaMalloc((void**)&devVectorB, N*sizeof(rett)));
	rett *devVectorX;
		CUDA_CHECK_ERROR(cudaMalloc((void**)&devVectorX, N*sizeof(rett)));
	rett *devMatrixA;
		CUDA_CHECK_ERROR(cudaMalloc((void**)&devMatrixA, N*N*sizeof(rett)));
	//Alloc for meta data
		//Scalars
		rett *devAlfaScalar;
			CUDA_CHECK_ERROR(cudaMalloc((void**)&devAlfaScalar, sizeof(rett)));
		rett *devBetaScalar;
			CUDA_CHECK_ERROR(cudaMalloc((void**)&devBetaScalar, sizeof(rett)));
		rett *devEps;
			CUDA_CHECK_ERROR(cudaMalloc((void**)&devEps, sizeof(rett)));
		//Temptory scalar
		rett *devDividerScalar; // for (P,Q)
			CUDA_CHECK_ERROR(cudaMalloc((void**)&devDividerScalar, sizeof(rett)));
		//Vectors
		rett *devR;
			CUDA_CHECK_ERROR(cudaMalloc((void**)&devR, N*sizeof(rett)));
		rett *devP;
			CUDA_CHECK_ERROR(cudaMalloc((void**)&devP, N*sizeof(rett)));
		rett *devQ;
			CUDA_CHECK_ERROR(cudaMalloc((void**)&devQ, N*sizeof(rett)));
		//Temptory vectors
		rett *devVectorTemp1;
			CUDA_CHECK_ERROR(cudaMalloc((void**)&devVectorTemp1, N*sizeof(rett)));
		rett *devMetaScalarResult;
		const countt DEV_META_RESULT_SIZE = MAX_BLOCKS/THREADS_PER_BLOCK;//колличество блоков на 1м шаге скалярного произведения
			CUDA_CHECK_ERROR(cudaMalloc((void**)&devMetaScalarResult, DEV_META_RESULT_SIZE*sizeof(rett)));
	//initialization data on device
	CUDA_CHECK_ERROR(cudaMemcpy(devVectorB, vectorB, N*sizeof(rett), cudaMemcpyHostToDevice));
	//Задаем начальное приближение
	CUDA_CHECK_ERROR(cudaMemcpy(devVectorX, vectorX, N*sizeof(rett), cudaMemcpyHostToDevice));
	CUDA_CHECK_ERROR(cudaMemcpy(devMatrixA, matrixA, N*N*sizeof(rett), cudaMemcpyHostToDevice));
	CUDA_CHECK_ERROR(cudaMemcpy(devEps, &eps, sizeof(rett), cudaMemcpyHostToDevice));
	//registration of events
	cudaEvent_t start;
    cudaEvent_t stop;
	//create events to sync and get timeout
	CUDA_CHECK_ERROR(cudaEventCreate(&start));
	CUDA_CHECK_ERROR(cudaEventCreate(&stop));
	//point of start GPU calc
	cudaEventRecord(start, 0);
//Method step 1
//Вычисляем невязку на 0м шаге
	//init grid parametres
	dim3 gridSizeStandart = dim3(MAX_BLOCKS/THREADS_PER_BLOCK, 1, 1);
	dim3 blockSizeStandart = dim3(THREADS_PER_BLOCK, 1, 1);
	//A*X
	matrixOnVectorMultiplyKernel<<< gridSizeStandart, blockSizeStandart >>>(devMatrixA, devVectorX, devVectorTemp1, N);
	cudaThreadSynchronize();
	//R0 = B - AX
	xSubYKernel<<< gridSizeStandart, blockSizeStandart >>>(devVectorB,devVectorTemp1,devR ,N);
	cudaThreadSynchronize();
	//P0 = R0
	copyKernel<<< gridSizeStandart, blockSizeStandart >>>(devR, devP, N);
	cudaThreadSynchronize();

for (countt k = 0; k < N; k++){
//Method step 2
	//Qk = A*Pk
	matrixOnVectorMultiplyKernel<<< gridSizeStandart, blockSizeStandart >>>(devMatrixA, devP, devQ, N);
	cudaThreadSynchronize();
	//devAlfaScalar = (Rk,Pk) / (Qk,Pk)
		// 1. devDividerScalar = (Qk,Pk) //вычисляем знаменатель - далее он еще пригодится для devBetaScalar
		dim3 gridSizeStep1 = dim3(DEV_META_RESULT_SIZE, 1, 1);
		dim3 blockSizeStep1 = dim3(THREADS_PER_BLOCK, 1, 1);
			//call of kernel for step 1 of scalar mult
			multiplyVectorsAndPartialSumKernel<<< gridSizeStep1, blockSizeStep1 >>>(devQ, devP, devMetaScalarResult, N);
			cudaThreadSynchronize();
			//init grid parametres on Step 2
		dim3 gridSizeStep2 = dim3(1, 1, 1);
		dim3 blockSizeStep2 = dim3(THREADS_PER_BLOCK, 1, 1);
			//call of kernel for step 2 of scalar mult
			reductionSumAtSingleBlockKernel<<< gridSizeStep2, blockSizeStep2 >>>(devMetaScalarResult,devDividerScalar,DEV_META_RESULT_SIZE);
			cudaThreadSynchronize();
		// 2. devAlfaScalar = (R,P) / devDividerScalar
			multiplyVectorsAndPartialSumKernel<<< gridSizeStep1, blockSizeStep1 >>>(devR, devP, devMetaScalarResult, N);
			cudaThreadSynchronize();
			reductionSumAtSingleBlockSpecialKernelWithDivide<<< gridSizeStep2, blockSizeStep2 >>>(devMetaScalarResult,devAlfaScalar,DEV_META_RESULT_SIZE,devDividerScalar);
			cudaThreadSynchronize();
	//X = X + devAlfaScalar * P
		xPlusAlfaYKernel<<< gridSizeStandart, blockSizeStandart >>>(devVectorX,devP,devAlfaScalar,devVectorX,N);
		cudaThreadSynchronize();
	//R = R - devAlfaScalar * Q
		xMinusAlfaYKernel<<< gridSizeStandart, blockSizeStandart >>>(devR,devQ,devAlfaScalar,devR,N);
		cudaThreadSynchronize();
//Method step 3
	// |X| < eps ?
	// devBetaScalar = (R,Q) / (Q,P)
		// 1. devDividerScalar уже вычислен ранее (devDividerScalar == (Q,P))
		// 2. devBetaScalar = (R,Q) / devDividerScalar 
			multiplyVectorsAndPartialSumKernel<<< gridSizeStep1, blockSizeStep1 >>>(devR, devQ, devMetaScalarResult, N);
			cudaThreadSynchronize();
			reductionSumAtSingleBlockSpecialKernelWithDivide<<< gridSizeStep2, blockSizeStep2 >>>(devMetaScalarResult,devBetaScalar,DEV_META_RESULT_SIZE,devDividerScalar);
			cudaThreadSynchronize();
	// P = R - devBetaScalar * P (новое направление минимизации)
		xMinusAlfaYKernel<<< gridSizeStandart, blockSizeStandart >>>(devR,devP,devBetaScalar,devP,N);
		cudaThreadSynchronize();
}

	//point of end calculation
	cudaEventRecord(stop, 0);
	float time = 0;
	//sunc all threads
    cudaEventSynchronize(stop);
	//get execution time
    cudaEventElapsedTime(&time, start, stop);
	//copy from device
	CUDA_CHECK_ERROR(cudaMemcpy(vectorX, devVectorX, sizeof(rett)*N, cudaMemcpyDeviceToHost));
	//free allocated GPU memory
	CUDA_CHECK_ERROR(cudaFree(devMatrixA));
	CUDA_CHECK_ERROR(cudaFree(devVectorB));
	CUDA_CHECK_ERROR(cudaFree(devVectorX));
	CUDA_CHECK_ERROR(cudaFree(devAlfaScalar));
	CUDA_CHECK_ERROR(cudaFree(devBetaScalar));
	CUDA_CHECK_ERROR(cudaFree(devEps));
	CUDA_CHECK_ERROR(cudaFree(devR));
	CUDA_CHECK_ERROR(cudaFree(devP));
	CUDA_CHECK_ERROR(cudaFree(devQ));
	CUDA_CHECK_ERROR(cudaFree(devVectorTemp1));
	//destroy of events
	CUDA_CHECK_ERROR(cudaEventDestroy(start));
	CUDA_CHECK_ERROR(cudaEventDestroy(stop));
	return time;
}