#include <cstdio>
#include <stdio.h>

#pragma once

#define rett float
#define countt unsigned long

//numeric constants
#define PI 3.14159265

//machine dependent constants
#define SHARED_MEMORY_SIZE_BYTE 48*1024
#define THREADS_PER_BLOCK 256 // only %2 == 0
#define MAX_BLOCKS 65536
#define TOTAL_RAM_BYTE 1.5*1024*1024*1024


class DeviceInfo{
public:
	DeviceInfo(){
		this->deviceCount = 0;
	};
	DeviceInfo(int deviceCount){
		this->deviceCount = deviceCount;
	}
	void toString(){
		if (this->deviceCount > 0){
			for (int i = 0; i < this->deviceCount; i++){
				printf("Device # %d\n",i);
				printf("TotalGlobalMem # %d\n",this->totalGlobalMem[i]);
				printf("SharedMemPerBlock # %d\n",this->sharedMemPerBlock[i]);
				printf("MaxThreadsPerBlock # %d\n",this->maxThreadsPerBlock[i]);
				printf("\n");
			}
		}else{
			printf("hasn't devises.\n");
		}
	}

	int deviceCount;
	int totalGlobalMem[10];
	int sharedMemPerBlock[10];
	int maxThreadsPerBlock[10];
};