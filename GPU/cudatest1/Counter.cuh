#ifndef COUNTER_H
#define COUNTER_H
#include "GeneralUtil.cuh"

typedef struct
{
	int *m;					// number of counters in a counter array
	int *counterSize;			// size of counter
	int *maxValue;			// maximum value of a counter
	int *counters;
}Counter;


Counter *newCounter(int m, int size);

//void deleteBitmap();

//char *getDataStructureName();
//
//int getUnitSize();

__device__ void encodeCounter(Counter *c);

__device__ void encodeCounterEID(const Counter *c, int elementID);

__device__ void encodeCounterSegment(const Counter *c, int flowID, int *s, int w);

__device__ void encodeCounterSegmentEID(const Counter *c, int flowID, int elementID, int *s, int w);

#endif