#include "GlobalEntitys.h"

#pragma once

#define CUDA_DEBUG

#ifdef CUDA_DEBUG

#define CUDA_CHECK_ERROR(err)           \
if (err != cudaSuccess) {          \
 printf("Cuda error: %s\n", cudaGetErrorString(err));    \
 printf("Error in file: %s, line: %i\n", __FILE__, __LINE__);  \
 }                 \

#else

#define CUDA_CHECK_ERROR(err)

#endif

extern "C" __declspec(dllexport) void  __stdcall ups();
extern "C" __declspec(dllexport) float __stdcall methodSoprGrad(rett *A, rett *b, rett *x,const int col, rett e);
extern "C" __declspec(dllexport) float __stdcall skalarMult(rett *a, rett *b, rett &x, int col);
extern "C" __declspec(dllexport) float __stdcall matrixOnVectorMultiply(rett *A, rett *b, rett *result, const countt N);
extern "C" __declspec(dllexport) float __stdcall matrixPartOnVectorMultiply(rett *A, rett *b, rett *result,const countt N, const countt partNumber, const countt partSize);

extern "C" __declspec(dllexport) float __stdcall getDeviceInfo(DeviceInfo &dc);

