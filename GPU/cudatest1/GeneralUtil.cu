#include "stdio.h"
#include "cuda_runtime.h"
#include <curand_kernel.h>
#include "GeneralUtil.cuh"


__device__ curandState st;
__device__  int intHash(int key) {
	key += ~(key << 15);
	key ^= (key >> 10);
	key += (key << 3);
	key ^= (key >> 6);
	key += ~(key << 11);
	key ^= (key >> 16);
	return key;
}
__global__ void initcurand() {
	curand_init(0, 0, 0, &st);
}

__device__ int cudarand() {
	int t=(int)curand(&st);
	if (t < 0) t = -t;
	return t % 32768;
}

__device__ int cudamin(int x, int y) {
	if (x < y) return x;
	else
		return y;
}

__device__ void ip_str_to_num(unsigned int *src,unsigned int *dst, char *buf) {
	unsigned int tmpip[4] = { 0 };
	unsigned int tmpip1[4] = { 0 };
	char buf1[16];
	int i = 0;
	while (buf[i] == '.' || (buf[i] >= '0'&&buf[i] <= '9')) {
		buf1[i] = buf[i];
		i++;
	}
	buf1[i] = 0;
	int ll = 0;
	for (int l = 0; l < 4; i++) {
		while (buf1[ll] != 0 && buf1[ll] != '.') tmpip[l] = tmpip[l] * 10 + (buf1[ll++] - '0');
	}
	int j = 0;
	while (buf[i]<'0' || buf[i]>'9') i++;
	while (buf[i] == '.' || (buf[i] >= '0'&&buf[i] <= '9')) {
		buf1[j] = buf[i];
		i++;
		j++;
	}
	buf1[j] = 0;
	ll = 0;
	for (int l = 0; l < 4; i++) {
		while (buf1[ll] != 0 && buf1[ll] != '.') tmpip1[l] = tmpip1[l] * 10 + (buf1[ll++] - '0');
	}
	*src = (tmpip[3] << 24) | (tmpip[2] << 16) | (tmpip[1] << 8) | tmpip[0];
	*dst = (tmpip1[3] << 24) | (tmpip1[2] << 16) | (tmpip1[1] << 8) | tmpip1[0];
	free(tmpip);
	free(tmpip1);
	free(buf1);
}
__device__  unsigned int uIntHash(unsigned int a)
{
	a = (a + 0x7ed55d16) + (a << 12);
	a = (a ^ 0xc761c23c) ^ (a >> 19);
	a = (a + 0x165667b1) + (a << 5);
	a = (a + 0xd3a2646c) ^ (a << 9);
	a = (a + 0xfd7046c5) + (a << 3);
	a = (a ^ 0xb55a4f09) ^ (a >> 16);
	return a;
}

__device__ int numberOfLeadingZeros(unsigned x)
{
	int n = 0;
	if (x <= 0x0000ffff) n += 16, x <<= 16;
	if (x <= 0x00ffffff) n += 8, x <<= 8;
	if (x <= 0x0fffffff) n += 4, x <<= 4;
	if (x <= 0x3fffffff) n += 2, x <<= 2;
	if (x <= 0x7fffffff) n++;
	return n;
}
