#ifndef HYPERLOGLOG_H
#define HYPERLOGLOG_H
#include "GeneralUtil.cuh"

typedef struct
{
	int *HLLSize;			// size of HyperLogLog array
	int *m;						// number of HyperLogLoges
	int *maxRegisterValue;
	int **HLL;
	// double alpha;
}HyperLogLog;

HyperLogLog *newHyperLogLog(int m, int size);

__device__ void encodeHyperLogLogspread(const HyperLogLog *b, int flowID, int elementID, int *s, int w);

__device__ void encodeHyperLogLog(const HyperLogLog *b);

__device__ void encodeHyperLogLogEID(const HyperLogLog *b, int elementID);

__device__ void encodeHyperLogLogSegment(const HyperLogLog *b, int flowID, int *s, int w);

__device__ void encodeHyperLogLogSegmentEID(const HyperLogLog *b, int flowID, int elementID, int *s, int w);

// double getAlpha(int m);

__device__ int getBitsetValue(bool *b);

__device__ void setBitsetValue(const HyperLogLog *b, int index, int value);

#endif