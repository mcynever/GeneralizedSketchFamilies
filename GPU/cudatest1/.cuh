#ifndef GENERALSHARING_H
#define GENERALSHARING_H
#include "Bitmap.cuh"
#include "Counter.cuh"
#include "FMsketch.cuh"
#include "HyperLogLog.cuh"
#include "cuda_runtime.h"
typedef struct
{
	int *w;				// the number of columns in Count Min
	int *u;				// the size of each elementary data structure in Count Min.
	int *S;		// random seeds for Count Min
	int *m;				// number of bit/register in each unit (used for bitmap, FM sketch and HLL sketch)
	int *sketchName;		// sketch that is used in the algorithm
	union {
		Bitmap **B;			// bitmap-1
		Counter **C;		// counter-0
		FMsketch **F;		// FMskektch-2
		HyperLogLog **H;	// HLL-3
	};
}GeneralSharing;


GeneralSharing *initSharing(int sketchName);

void generateSharingRandomSeeds(GeneralSharing *GVS);

#endif