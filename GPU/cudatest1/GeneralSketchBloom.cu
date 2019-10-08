#include "cuda_runtime.h"			//CUDA‘À–– ±API
#include "device_launch_parameters.h"
#include <random>
#include <curand_kernel.h>
#include "GeneralSketchBloom.cuh"
#include "iostream"
using namespace std;
/** parameters for count-min */
const int d = 4; 			// the nubmer of rows in Count Min

GeneralSketchBloom *initSketchBloom(int sketchName)
{
	int M = 1024 * 1024 * 4; 	// total memory space Mbits	


	/** parameters for counter */
	const int mValueCounter = 1;			// only one counter in the counter data structure
	const int counterSize = 32;				// size of each unit

	/** parameters for bitmap */
	const int bitArrayLength = 20000;

	/** parameters for FM sketch **/
	const int mValueFM = 128;
	const int FMsketchSize = 32;

	/** parameters for HLL sketch **/
	const int mValueHLL = 128;
	const int HLLSize = 5;
	
	GeneralSketchBloom GSB[2];


	GSB[0].sketchName = (int *)malloc(sizeof(int));
	GSB[0].m = (int *)malloc(sizeof(int));
	GSB[0].w = (int *)malloc(sizeof(int));
	GSB[0].u = (int *)malloc(sizeof(int));
	*GSB[0].sketchName = sketchName;

	cudaMalloc((void**)&GSB[1].w, sizeof(int));
	cudaMalloc((void**)&GSB[1].u, sizeof(int));
	cudaMalloc((void**)&GSB[1].m, sizeof(int));
	cudaMalloc((void**)&GSB[1].sketchName, sizeof(int));

	if (sketchName == 0) {
		*GSB[0].m = mValueCounter;
		*GSB[0].u = counterSize;
		int w = M / mValueCounter / counterSize;
		*GSB[0].w = w;

		cudaMalloc((void **)&GSB[1].C, sizeof(Counter **));
		Counter **tmpC[1];
		for (int i = 0; i < 1; i++) {
		
			cudaMalloc((void **)&(tmpC[0]), *GSB[0].w * sizeof(Counter *));
			Counter **tempC = (Counter **)malloc(*GSB[0].w * sizeof(Counter *));
		
			for (int j = 0; j < *GSB[0].w; j++) {
				tempC[j] = newCounter(*GSB[0].m, *GSB[0].u);
			}
			cudaMemcpy(tmpC[0], tempC, *GSB[0].w * sizeof(Counter *), cudaMemcpyHostToDevice);
		}
		cudaMemcpy(GSB[1].C, tmpC, sizeof(Counter **), cudaMemcpyHostToDevice);
		//printf("%s\n", "\nGeneral SketchBloom-Counter Initialized!");
	}
	else if (sketchName == 1) {
		*GSB[0].m = bitArrayLength;
		*GSB[0].u = bitArrayLength;
		*GSB[0].w = (M / *GSB[0].u) / 1;

		cudaMalloc((void **)&GSB[1].B, sizeof(Bitmap **));
		Bitmap **tmpB[1];
		for (int i = 0; i < 1; i++) {
			
			cudaMalloc((void **)&(tmpB[0]), *GSB[0].w * sizeof(Bitmap *));
			Bitmap **tempB = (Bitmap **)malloc(*GSB[0].w * sizeof(Bitmap *));
			
			for (int j = 0; j < *GSB[0].w; j++) {
				tempB[j] = newBitmap(*GSB[0].m, *GSB[0].u);
			}
			cudaMemcpy(tmpB[0], tempB, *GSB[0].w * sizeof(Bitmap *), cudaMemcpyHostToDevice);
		}
		cudaMemcpy(GSB[1].B, tmpB, sizeof(Bitmap **), cudaMemcpyHostToDevice);
		//printf("%s\n", "\nGeneral SketchBloom-Bitmap Initialized!");
	}
	else if (sketchName == 2) {
		*GSB[0].m = mValueFM;
		*GSB[0].u = FMsketchSize;
		*GSB[0].w = (M / *GSB[0].u / *GSB[0].m) / 1;
		cudaMalloc((void **)&GSB[1].F, sizeof(FMsketch **));
		FMsketch **tmpF[1];
		for (int i = 0; i < 1; i++) {
			cudaMalloc((void **)&(tmpF[0]), *GSB[0].w * sizeof(FMsketch *));
			FMsketch **tempF = (FMsketch **)malloc(*GSB[0].w * sizeof(FMsketch *));
			for (int j = 0; j < *GSB[0].w; j++) {
				tempF[j] = newFMsketch(*GSB[0].m, *GSB[0].u);
			}
			cudaMemcpy(tmpF[0], tempF, *GSB[0].w * sizeof(FMsketch *), cudaMemcpyHostToDevice);
		}
		cudaMemcpy(GSB[1].F, tmpF, sizeof(FMsketch **), cudaMemcpyHostToDevice);
		//printf("%s\n", "\nGeneral SketchBloom-FMsketch Initialized!");
	}
	else if (sketchName == 3) {
		*GSB->m = mValueHLL;
		*GSB->u = HLLSize;
		*GSB->w = (M / (*GSB->u * *GSB->m)) / 1;
		cudaMalloc((void **)&GSB[1].H, sizeof(HyperLogLog **));
		HyperLogLog **tmpH[1];
		for (int i = 0; i < 1; i++) {
			cudaMalloc((void **)&(tmpH[0]), *GSB[0].w * sizeof(HyperLogLog *));
			HyperLogLog **tempH = (HyperLogLog **)malloc(*GSB[0].w * sizeof(HyperLogLog *));
			for (int j = 0; j < *GSB[0].w; j++) {
				tempH[j] = newHyperLogLog(*GSB[0].m, *GSB[0].u);
			}
			cudaMemcpy(tmpH[0], tempH, *GSB[0].w * sizeof(HyperLogLog *), cudaMemcpyHostToDevice);
		}
		cudaMemcpy(GSB[1].H, tmpH, sizeof(HyperLogLog **), cudaMemcpyHostToDevice);
		//printf("%s\n", "\nGeneral SketchBloom-HyperLogLog Initialized!");
	}
	else {
		printf("%s\n", "Unsupported Data Structure!!");
	}
	GeneralSketchBloom *res;
	generateSketchBloomRandomSeeds(GSB);
	cudaMalloc((void**)&res, sizeof(GeneralSketchBloom));
	cudaMemcpy(res, &GSB[1], sizeof(GeneralSketchBloom), cudaMemcpyHostToDevice);
	cudaMemcpy(GSB[1].w, GSB[0].w, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(GSB[1].u, GSB[0].u, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(GSB[1].m, GSB[0].m, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(GSB[1].sketchName, GSB[0].sketchName, sizeof(int), cudaMemcpyHostToDevice);
	return res;
}

void generateSketchBloomRandomSeeds(GeneralSketchBloom *GSB)
{
	int num = d;
	GSB[0].S = (int *)malloc(d * sizeof(int));
	int i = 0;
	for (i = 0; i < d; i++) {
		*(GSB[0].S + i) = rand();
	}
	//cout << GSB[0].S[0] << endl;
	cudaMalloc((void**)&GSB[1].S, sizeof(int)*d);
	cudaMemcpy(GSB[1].S, GSB[0].S, sizeof(int)*d, cudaMemcpyHostToDevice);
}
