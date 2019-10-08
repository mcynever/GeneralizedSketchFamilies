
#include "cuda_runtime.h"			
#include "device_launch_parameters.h"
#include <random>
#include <curand_kernel.h>
#include "iostream"
#include "GeneralVSketch.cuh"
using namespace std;
GeneralVSketch *initVSketch(int sketchName)
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
	GeneralVSketch GVS[2];

	GVS[0].sketchName = (int *)malloc(sizeof(int));
	GVS[0].m = (int *)malloc(sizeof(int));
	GVS[0].w = (int *)malloc(sizeof(int));
	GVS[0].u = (int *)malloc(sizeof(int));
	*GVS[0].sketchName = sketchName;
	cudaMalloc((void**)&GVS[1].w, sizeof(int));
	cudaMalloc((void**)&GVS[1].u, sizeof(int));
	cudaMalloc((void**)&GVS[1].m, sizeof(int));
	cudaMalloc((void**)&GVS[1].sketchName, sizeof(int));

	if (sketchName == 0) {
		*GVS[0].m = mValueCounter;
		*GVS[0].u = counterSize;
		*GVS[0].w = M / counterSize;
		cudaMalloc((void **)&GVS[1].C, sizeof(Counter *) * 1);
		Counter **tmpC= (Counter **)malloc(1 * sizeof(Counter*));
		tmpC[0]= newCounter(*GVS[0].w, *GVS[0].u);
		cudaMemcpy(GVS[1].C, tmpC, sizeof(Counter *)*1, cudaMemcpyHostToDevice);
	//	printf("%s\n", "\nGeneral VSketch-Counter Initialized!");
	}
	else if (sketchName == 1) {
		*GVS->m = virtualArrayLength;
		*GVS->u = bitmapSize;
		*GVS->w = (M / *GVS->u);
		cudaMalloc((void **)&GVS[1].B, sizeof(Bitmap *) * 1);
		Bitmap **tmpB = (Bitmap **)malloc(1 * sizeof(Bitmap*));
		tmpB[0] = newBitmap(*GVS[0].w, *GVS[0].u);
		cudaMemcpy(GVS[1].B, tmpB, sizeof(Bitmap *) * 1, cudaMemcpyHostToDevice);
		//printf("%s\n", "\nGeneral VSketch-Bitmap Initialized!");
	}
	else if (sketchName == 2) {
		*GVS->m = mValueFM;
		*GVS->u = FMsketchSize;
		*GVS->w = M / *GVS->u;
		cudaMalloc((void **)&GVS[1].F, sizeof(FMsketch *) * 1);
		FMsketch **tmpF = (FMsketch **)malloc(1 * sizeof(FMsketch*));
		tmpF[0] = newFMsketch(*GVS[0].w, *GVS[0].u);
		cudaMemcpy(GVS[1].F, tmpF, sizeof(FMsketch *) * 1, cudaMemcpyHostToDevice);
		//printf("%s\n", "\nGeneral VSketch-FMsketch Initialized!");
	}
	else if (sketchName == 3) {
		*GVS->m = mValueHLL;
		*GVS->u = HLLSize;
		*GVS->w = M / *GVS->u;
		cudaMalloc((void **)&GVS[1].H, sizeof(HyperLogLog *) * 1);
		HyperLogLog **tmpH = (HyperLogLog **)malloc(1 * sizeof(HyperLogLog*));
		tmpH[0] = newHyperLogLog(*GVS[0].w, *GVS[0].u);
		cudaMemcpy(GVS[1].H, tmpH, sizeof(HyperLogLog *) * 1, cudaMemcpyHostToDevice);
		//printf("%s\n", "\nGeneral VSketch-HyperLogLog Initialized!");
	}
	else {
		printf("%s\n", "Unsupported Data Structure!!");
	}
	generateVSketchRandomSeeds(GVS);
	GeneralVSketch *res;
	cudaMalloc((void**)&res, sizeof(GeneralVSketch));
	cudaMemcpy(res, &GVS[1], sizeof(GeneralVSketch), cudaMemcpyHostToDevice);
	cudaMemcpy(GVS[1].w, GVS[0].w, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(GVS[1].u, GVS[0].u, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(GVS[1].m, GVS[0].m, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(GVS[1].sketchName, GVS[0].sketchName, sizeof(int), cudaMemcpyHostToDevice);
	return res;
}

void generateVSketchRandomSeeds(GeneralVSketch *GVS)
{
	GVS[0].S = (int *)malloc(*GVS[0].m * sizeof(int));
	for (int i = 0; i < *GVS[0].m; i++) {
		*(GVS[0].S + i) = rand();
	}
	cudaMalloc((void**)&GVS[1].S, sizeof(int)* *GVS[0].m);
	cudaMemcpy(GVS[1].S, GVS[0].S, sizeof(int)* *GVS[0].m, cudaMemcpyHostToDevice);
}

