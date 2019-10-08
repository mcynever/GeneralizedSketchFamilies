#include <random>
#include "cuda_runtime.h"			//CUDA‘À–– ±API
#include "device_launch_parameters.h"
#include "Bitmap.cuh"
#include "iostream"
using namespace std;
Bitmap *newBitmap(int m, int size) {
	Bitmap b[2];
	b[0].arraySize = (int *)malloc(sizeof(int));
	b[0].m = (int *)malloc(sizeof(int));
	b[0].B = (bool *)malloc(m * sizeof(bool));
	cudaMalloc((void **)&b[1].m,sizeof(int));
	cudaMalloc((void **)&b[1].arraySize, sizeof(int));
	cudaMalloc((void **)&b[1].B, sizeof(bool)*m);
	//srand(time(NULL));
	*b[0].m = m;
	*b[0].arraySize = m;
	int i = 0;
	for (i = 0; i < m; i++) {
		b[0].B[i] = false;
	}
	cudaMemcpy(b[1].B, b[0].B, sizeof(bool)*m, cudaMemcpyHostToDevice);
	cudaMemcpy(b[1].m, b[0].m, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(b[1].arraySize, b[0].arraySize, sizeof(int),cudaMemcpyHostToDevice);
	// printf("%s\n", "This is a bitmap");
	Bitmap *res;
	cudaMalloc((void**)&res, sizeof(Bitmap));
	cudaMemcpy(res, &b[1], sizeof(Bitmap),cudaMemcpyHostToDevice);
	return res;
}

__device__ void encode(const Bitmap *b, int flowID, int *s) {
	int r = cudarand();
	int j=(intHash(r) % *b->m+ *b->m)% *b->m;
	int k = ((intHash(flowID)^s[j]) % *b->arraySize + *b->arraySize) % *b->arraySize;
	b->B[k] = true;
}

__device__ void encodeBitmap(Bitmap *b) {
	int r =  cudarand();
	int k = r % *b->arraySize;
	//if (k < 0) { printf("k less than zero!\n"); return; }
	b->B[k] = true;
	// printf("%s\n", "This is a bitmap");
}
__device__ void encodeBitmapEID(const Bitmap *b, int elementID) {
	int k = (intHash(elementID) % *b->arraySize + *b->arraySize) % *b->arraySize;
	b->B[k] = true;
}

__device__ void encodeBitmapSegment(const Bitmap *b, int flowID, int *s, int w) {
	int ms = *b->arraySize / w;
	int r = (int)cudarand();
	int j = r % ms;			// (GeneralUtil::intHash(flowID) % ms + ms) % ms;							// rand() % ms;
	int k = (intHash(flowID ^ s[j]) % w + w) % w;
	int i = j * w + k;
	b->B[i] = true;
}

__device__ void encodeBitmapSegmentEID(const Bitmap *b, int flowID, int elementID, int *s, int w) {
	int m = *b->arraySize / w;
	int j = (intHash(elementID^flowID) % m + m) % m;
	int k = (intHash(flowID ^ s[j]) % w + w) % w;
	int i = j * w + k;
	b->B[i] = true;
}
