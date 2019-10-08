#include "HyperLogLog.cuh"
#include <random>
#include "cuda_runtime.h"			//CUDA‘À–– ±API
#include "device_launch_parameters.h"
#include "iostream"
using namespace std;
HyperLogLog *newHyperLogLog(int m, int size) {
	HyperLogLog h[2];
	//srand(time(NULL));
	h[0].HLLSize = (int *)malloc(sizeof(int));
	h[0].m = (int *)malloc(sizeof(int));
	h[0].maxRegisterValue = (int *)malloc(sizeof(int));
	cudaMalloc((void **)&h[1].HLLSize, sizeof(int));
	cudaMalloc((void **)&h[1].m, sizeof(int));
	cudaMalloc((void **)&h[1].maxRegisterValue, sizeof(int));
	*h[0].HLLSize = size;
	*h[0].m = m;
	*h[0].maxRegisterValue = (int)(31);
	cudaMalloc((void **)&h[1].HLL, m * sizeof(int *));
	int **tmpH = (int **)malloc(m * sizeof(int *));
	int *tt = (int *)malloc(1 * sizeof(int));
	for (int j = 0; j < 1; j++) {
		tt[j] = 0;
	}
	for (int i = 0; i < m; i++) {
		cudaMalloc((void **)&(tmpH[i]), 1 * sizeof(int));
		cudaMemcpy(tmpH[i], tt, 1 * sizeof(int), cudaMemcpyHostToDevice);
	}
	cudaMemcpy(h[1].HLL, tmpH, m * sizeof(int *), cudaMemcpyHostToDevice);
	cudaMemcpy(h[1].HLLSize, h[0].HLLSize, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(h[1].m, h[0].m, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(h[1].maxRegisterValue, h[0].maxRegisterValue, sizeof(int), cudaMemcpyHostToDevice);

	//b->alpha = getAlpha(m);
	// printf("%s\n", "This is a HyperLogLog");
	HyperLogLog *res;
	cudaMalloc((void **)&res, sizeof(HyperLogLog));
	cudaMemcpy(res, &h[1], sizeof(HyperLogLog), cudaMemcpyHostToDevice);
	return res;
}

// printf("%s\n", "This is an FMsketch");

__device__ void encodeHyperLogLogspread(const HyperLogLog *b,int flowID, int elementID, int *s,int w) {
	int ms = *b->m;
	int j = (intHash(elementID) % ms + ms) % ms;
	int i = (intHash(intHash(flowID) ^ s[j]) % w + w) % w;
	int hash_val = intHash(elementID);
	int leadingZeros = numberOfLeadingZeros(hash_val) + 1;
	if (leadingZeros > *b->maxRegisterValue)
		leadingZeros = *b->maxRegisterValue;
	//if (getBitsetValue(b->HLL[i]) < leadingZeros) {
		//setBitsetValue(b, i, leadingZeros);
	//}
	atomicMax(&b->HLL[i][0], leadingZeros);
}



__device__ void encodeHyperLogLog(const HyperLogLog *b) {
	int r = cudarand();
	int k = r % *b->m;
	//unsigned hash_val = (unsigned) hash(distribution(generator));
	unsigned hash_val = uIntHash(r);
	int leadingZeros = numberOfLeadingZeros(hash_val) + 1; // % hyperLogLog_size + hyperLogLog_size) % hyperLogLog_size;
	leadingZeros = cudamin(leadingZeros, *b->maxRegisterValue);
	//if (getBitsetValue(b->HLL[k]) < leadingZeros) {
		//setBitsetValue(b, k, leadingZeros);
	//}
	atomicMax(&b->HLL[k][0], leadingZeros);
	// printf("%s\n", "This is a HyperLogLog");
}

__device__ void encodeHyperLogLogEID(const HyperLogLog *b, int elementID) {
	unsigned hash_val = uIntHash((unsigned)elementID);
	int leadingZeros = numberOfLeadingZeros(hash_val) + 1; // % hyperLogLog_size + hyperLogLog_size) % hyperLogLog_size;
	int k = (hash_val % *b->m + *b->m) % *b->m;
	leadingZeros = cudamin(leadingZeros, *b->maxRegisterValue);
	//if (getBitsetValue(b->HLL[k]) < leadingZeros) {
		//setBitsetValue(b, k, leadingZeros);
	//}
	atomicMax(&b->HLL[k][0], leadingZeros);
}

__device__ void encodeHyperLogLogSegment(const HyperLogLog *b, int flowID, int *s, int w) {
	int ms = *b->m / w;
	int r = cudarand();
	int j = r % ms;			// (GeneralUtil::intHash(flowID) % ms + ms) % ms;
	int k = (intHash(flowID ^ s[j]) % w + w) % w;
	int i = j * w + k;
	unsigned hash_val = uIntHash(r);
	int leadingZeros = numberOfLeadingZeros(hash_val) + 1;
	leadingZeros = cudamin(leadingZeros, *b->maxRegisterValue);
	//if (getBitsetValue(b->HLL[i]) < leadingZeros) {
		//setBitsetValue(b, i, leadingZeros);
	//}
	atomicMax(&b->HLL[i][0], leadingZeros);
}

__device__ void encodeHyperLogLogSegmentEID(const HyperLogLog *b, int flowID, int elementID, int *s, int w) {
	int m = *b->m / w;
	//int j = (intHash(elementID) % m + m) % m;
	int j = (intHash(elementID^flowID) % m + m) % m;
	int k = (intHash(flowID ^ s[j]) % w + w) % w;
	int i = j * w + k;
	unsigned hash_val = uIntHash(elementID);						// (unsigned)
	int leadingZeros = numberOfLeadingZeros(hash_val) + 1;
	leadingZeros = cudamin(leadingZeros, *b->maxRegisterValue);
	//if (getBitsetValue(b->HLL[i]) < leadingZeros) {
		//setBitsetValue(b, i, leadingZeros);
	//}
	atomicMax(&b->HLL[i][0], leadingZeros);
}

// double getAlpha(int m)
// {
// 	double a;
// 	if (m == 16) {
// 		a = 0.673;
// 	}
// 	else if (m == 32) {
// 		a = 0.697;
// 	}
// 	else if (m == 64) {
// 		a = 0.709;
// 	}
// 	else {
// 		a = 0.7213 / (1 + 1.079 / m);
// 	}
// 	return a;
// }

__device__ int getBitsetValue(bool *b)
{
	int result = 0;
	int i = 0;
	for (i = 0; i < 5; i++) {
		if (b[i]) {
			result = result + (1 << i);
		}
	}
	return result;
}

__device__ void setBitsetValue(const HyperLogLog *b, int index, int value)
{
	int i = 0;
	while (value != 0 && i < *b->HLLSize) {
		if ((value & 1) != 0) {
			b->HLL[index][i] = 1;
		}
		else {
			b->HLL[index][i] = 0;
		}
		value = value >> 1;
		i++;
	}
}
