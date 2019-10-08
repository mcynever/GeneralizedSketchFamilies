#include "Counter.cuh"
#include <random>
#include "cuda_runtime.h"			//CUDA‘À–– ±API
#include "device_launch_parameters.h"
#include "iostream"
using namespace std;
Counter *newCounter(int m, int size) {
	Counter c[2];
	c[0].counterSize = (int *)malloc(sizeof(int));
	c[0].m = (int *)malloc(sizeof(int));
	c[0].maxValue = (int *)malloc(sizeof(int));
	c[0].counters = (int *)malloc(m * sizeof(int));
	cudaMalloc((void **)&c[1].counterSize, sizeof(int));
	cudaMalloc((void **)&c[1].m, sizeof(int));
	cudaMalloc((void **)&c[1].maxValue, sizeof(int));
	cudaMalloc((void **)&c[1].counters, sizeof(int)*m);
	*c[0].m = m;
	*c[0].counterSize = size;
	*c[0].maxValue = (int)(1 << size) - 1;
	int i = 0;
	for (i = 0; i < m; i++) {
		c[0].counters[i] = 0;
	}
	cudaMemcpy(c[1].counters, c[0].counters, sizeof(int)*m, cudaMemcpyHostToDevice);
	cudaMemcpy(c[1].counterSize, c[0].counterSize, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(c[1].m, c[0].m, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(c[1].maxValue, c[0].maxValue, sizeof(int), cudaMemcpyHostToDevice);
	//printf("%s\n", "This is a counter");
	Counter *res;
	cudaMalloc((void**)&res, sizeof(Counter));
	cudaMemcpy(res, &c[1], sizeof(Counter), cudaMemcpyHostToDevice);
	return res;
}





__device__ void encodeCounter(Counter *c) { // c is null
	//if (!c) { printk(KERN_WARNING"c is NULL!\n"); return; }
	int r = cudarand();
	int k = r % *c->m;
	//if (!c->counters) { printk(KERN_WARNING"c->counter is NULL!\n"); return; }
	atomicAdd(&c->counters[k], 1);
	// printf("%s\n", "This is a Counter");
}

__device__ void encodeCounterEID(const Counter *c, int elementID) {
	int k = (intHash(elementID) % *c->m + *c->m) % *c->m;
	atomicAdd(&c->counters[k], 1);
}

__device__ void encodeCounterSegment(const Counter *c, int flowID, int *s, int w) {
	int ms = *c->m / w;
	int r = cudarand();
	int j = r % ms;			// (GeneralUtil::intHash(flowID) % ms + ms) % ms;							// rand() % ms;
	int k = (intHash(flowID ^ s[j]) % w + w) % w;
	int i = j * w + k;
	atomicAdd(&c->counters[i], 1);
}

__device__ void encodeCounterSegmentEID(const Counter *c, int flowID, int elementID, int *s, int w) {
	int m = *c->m / w;
	int j = (intHash(elementID^flowID) % m + m) % m;
	int k = (intHash(flowID ^ s[j]) % w + w) % w;
	int i = j * w + k;
	atomicAdd(&c->counters[i], 1);
}
