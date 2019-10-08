
#include "cuda_runtime.h"			
#include "device_launch_parameters.h"
#include <random>
#include <curand_kernel.h>
#include "iostream"
#include "GeneralSharing.cuh"
using namespace std;
GeneralSharing *initSharing(int sketchName)
{
	int M = 1024 * 1024 * 4; 	// total memory space Mbits	
	/** parameters for counters **/
	const int mValueCounter = 128;
	const int counterSize = 32;

	/** parameters for bitmap **/
	const int bitmapSize = 1;	// sharing at bit level
	const int virtualArrayLength = 20000;

	/** parameters for FM sketch **/
	const int mValueFM = 128;
	const int FMsketchSize = 32;

	/** parameters for hyperLogLog **/
	const int mValueHLL = 128;
	const int HLLSize = 5;
	GeneralSharing GS[2];

	GS[0].sketchName = (int *)malloc(sizeof(int));
	GS[0].m = (int *)malloc(sizeof(int));
	GS[0].w = (int *)malloc(sizeof(int));
	GS[0].u = (int *)malloc(sizeof(int));
	*GS[0].sketchName = sketchName;
	cudaMalloc((void**)&GS[1].w, sizeof(int));
	cudaMalloc((void**)&GS[1].u, sizeof(int));
	cudaMalloc((void**)&GS[1].m, sizeof(int));
	cudaMalloc((void**)&GS[1].sketchName, sizeof(int));

	if (sketchName == 0) {
		*GS[0].m = mValueCounter;
		*GS[0].u = counterSize;
		*GS[0].w = M / counterSize;
		cudaMalloc((void **)&GS[1].C, sizeof(Counter *) * 1);
		Counter **tmpC = (Counter **)malloc(1 * sizeof(Counter*));
		tmpC[0] = newCounter(*GS[0].w, *GS[0].u);
		cudaMemcpy(GS[1].C, tmpC, sizeof(Counter *) * 1, cudaMemcpyHostToDevice);
		//	printf("%s\n", "\nGeneral VSketch-Counter Initialized!");
	}
	else if (sketchName == 1) {
		*GS->m = virtualArrayLength;
		*GS->u = bitmapSize;
		*GS->w = (M / *GS->u);
		cudaMalloc((void **)&GS[1].B, sizeof(Bitmap *) * 1);
		Bitmap **tmpB = (Bitmap **)malloc(1 * sizeof(Bitmap*));
		tmpB[0] = newBitmap(*GS[0].w, *GS[0].u);
		cudaMemcpy(GS[1].B, tmpB, sizeof(Bitmap *) * 1, cudaMemcpyHostToDevice);
		//printf("%s\n", "\nGeneral VSketch-Bitmap Initialized!");
	}
	else if (sketchName == 2) {
		*GS->m = mValueFM;
		*GS->u = FMsketchSize;
		*GS->w = M / *GS->u;
		cudaMalloc((void **)&GS[1].F, sizeof(FMsketch *) * 1);
		FMsketch **tmpF = (FMsketch **)malloc(1 * sizeof(FMsketch*));
		tmpF[0] = newFMsketch(*GS[0].w, *GS[0].u);
		cudaMemcpy(GS[1].F, tmpF, sizeof(FMsketch *) * 1, cudaMemcpyHostToDevice);
		//printf("%s\n", "\nGeneral VSketch-FMsketch Initialized!");
	}
	else if (sketchName == 3) {
		*GS->m = mValueHLL;
		*GS->u = HLLSize;
		*GS->w = M / *GS->u;
		cudaMalloc((void **)&GS[1].H, sizeof(HyperLogLog *) * 1);
		HyperLogLog **tmpH = (HyperLogLog **)malloc(1 * sizeof(HyperLogLog*));
		tmpH[0] = newHyperLogLog(*GS[0].w, *GS[0].u);
		cudaMemcpy(GS[1].H, tmpH, sizeof(HyperLogLog *) * 1, cudaMemcpyHostToDevice);
		//printf("%s\n", "\nGeneral VSketch-HyperLogLog Initialized!");
	}
	else {
		printf("%s\n", "Unsupported Data Structure!!");
	}
	generateSharingRandomSeeds(GS);
	GeneralSharing *res;
	cudaMalloc((void**)&res, sizeof(GeneralSharing));
	cudaMemcpy(res, &GS[1], sizeof(GeneralSharing), cudaMemcpyHostToDevice);
	cudaMemcpy(GS[1].w, GS[0].w, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(GS[1].u, GS[0].u, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(GS[1].m, GS[0].m, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(GS[1].sketchName, GS[0].sketchName, sizeof(int), cudaMemcpyHostToDevice);
	return res;
}

void generateSharingRandomSeeds(GeneralSharing *GS)
{
	GS[0].S = (int *)malloc(*GS[0].m * sizeof(int));
	for (int i = 0; i < *GS[0].m; i++) {
		*(GS[0].S + i) = rand();
	}
	cudaMalloc((void**)&GS[1].S, sizeof(int)* *GS[0].m);
	cudaMemcpy(GS[1].S, GS[0].S, sizeof(int)* *GS[0].m, cudaMemcpyHostToDevice);
}

