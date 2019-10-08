#ifndef FMSKETCH_H
#define FMSKETCH_H
#include "GeneralUtil.cuh"

typedef struct
{
	int *FMsketchSize;			// size of FMsketch array
	int *m;						// number of FMsketches
	bool **FMsketchMatrix;
}FMsketch;


FMsketch *newFMsketch(int m, int size);

__device__ void encodeFMsketch(const FMsketch *b);

__device__ void encodeFMsketchEID(const FMsketch *b, int elementID);

__device__ void encodeFMsketchSegment(const FMsketch *b, int flowID, int *s, int w);

__device__ void encodeFMsketchSegmentEID(const FMsketch *b, int flowID, int elementID, int *s, int w);

#endif
