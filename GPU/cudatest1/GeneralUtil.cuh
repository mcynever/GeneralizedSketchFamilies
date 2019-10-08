#ifndef GENERALUTIL_H
#define GENERALUTIL_H
#include "cuda_runtime.h"
#include <curand_kernel.h>

extern __device__ curandState st;

__global__ void initcurand();

__device__ int cudarand();

__device__ int cudamin(int x, int y);

__device__  int intHash(int key);

__device__ unsigned int uIntHash(unsigned int a);

__device__ void ip_str_to_num(unsigned int *src,unsigned int *dst, char *buf);

__device__ int numberOfLeadingZeros(unsigned x);

#endif