#ifndef GENERALSKETCHBLOOM_H
#define GENERALSKETCHBLOOM_H
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
		Bitmap ***B; // counter-0
		Counter ***C;// bitmap-1
		FMsketch ***F;		// FMskektch-2
		HyperLogLog ***H;	// HLL-3
	};
}GeneralSketchBloom;

GeneralSketchBloom *initSketchBloom(int sketchName);
void generateSketchBloomRandomSeeds(GeneralSketchBloom *GSB);
#endif