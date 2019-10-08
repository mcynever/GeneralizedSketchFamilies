#include "FMsketch.cuh"
#include <random>
#include "cuda_runtime.h"			//CUDA‘À–– ±API
#include "device_launch_parameters.h"
#include "iostream"
using namespace std;
const double phi = 0.77351;

FMsketch *newFMsketch(int m, int size) {
	FMsketch f[2];
	//srand(time(NULL));
	f[0].FMsketchSize = (int *)malloc(sizeof(int));
	f[0].m = (int *)malloc(sizeof(int));
	cudaMalloc((void **)&f[1].FMsketchSize, sizeof(int));
	cudaMalloc((void **)&f[1].m, sizeof(int));
	*f[0].FMsketchSize = size;
	*f[0].m = m;
	cudaMalloc((void **)&f[1].FMsketchMatrix, m*sizeof(bool *));
	bool **tmpF= (bool **)malloc(m * sizeof(bool*));
	bool *tt= (bool *)malloc(size * sizeof(bool));
	for (int j = 0; j < size; j++) {
		tt[j] = false;
	}
	for (int i = 0; i < m; i++) {
		cudaMalloc((void **)&(tmpF[i]), size * sizeof(bool));
		cudaMemcpy(tmpF[i],tt,size * sizeof(bool),cudaMemcpyHostToDevice);
	}
	cudaMemcpy(f[1].FMsketchMatrix, tmpF, m * sizeof(bool *), cudaMemcpyHostToDevice);
	cudaMemcpy(f[1].FMsketchSize, f[0].FMsketchSize, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(f[1].m, f[0].m, sizeof(int), cudaMemcpyHostToDevice);
	// printf("%s\n", "This is an FMsketch");
	FMsketch *res;
	cudaMalloc((void **)&res, sizeof(FMsketch));
	cudaMemcpy(res, &f[1], sizeof(FMsketch), cudaMemcpyHostToDevice);
	return res;
}


__device__ void encodeFMsketch(const FMsketch *b) {
	int r = cudarand();
	int k = r % (*b->m);
	//unsigned hash_val = (unsigned) hash(distribution(generator));
	unsigned hash_val = uIntHash(r);
	int leadingZeros = cudamin(numberOfLeadingZeros(hash_val), *b->FMsketchSize - 1);
	(b->FMsketchMatrix)[k][leadingZeros] = 1;
	// printf("%s\n", "This is an FMsketch");
}

__device__ void encodeFMsketchEID(const FMsketch *b, int elementID) {
	unsigned hash_val = uIntHash((unsigned)elementID);
	int leadingZeros = cudamin(numberOfLeadingZeros(hash_val), *b->FMsketchSize - 1);
	int k = (hash_val % *b->m + *b->m) % *b->m;
	b->FMsketchMatrix[k][leadingZeros] = 1;
}

__device__ void encodeFMsketchSegment(const FMsketch *b, int flowID, int *s, int w) {
	int ms = *b->m / w;
	int r = cudarand();
	int j = r % ms;			// (GeneralUtil::intHash(flowID) % ms + ms) % ms;
	int k = (intHash(flowID ^ s[j]) % w + w) % w;
	int i = j * w + k;
	unsigned hash_val = uIntHash(r);							// (unsigned)
	int leadingZeros = cudamin(numberOfLeadingZeros(hash_val), *b->FMsketchSize - 1);
	b->FMsketchMatrix[i][leadingZeros] = 1;
}

__device__ void encodeFMsketchSegmentEID(const FMsketch *b, int flowID, int elementID, int *s, int w) {
	int m = *b->m / w;
	int j = (intHash(elementID^flowID) % m + m) % m;
	int k = (intHash(flowID ^ s[j]) % w + w) % w;
	int i = j * w + k;
	unsigned hash_val = uIntHash(elementID);						// (unsigned)
	int leadingZeros = cudamin(numberOfLeadingZeros(hash_val), *b->FMsketchSize - 1);
	b->FMsketchMatrix[i][leadingZeros] = 1;
}
