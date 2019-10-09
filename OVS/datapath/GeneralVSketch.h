#ifndef GENERALVSKETCH_H
#define GENERALVSKETCH_H
#include "GeneralUtil.h"
#include "Bitmap.h"
#include "Counter.h"
#include "FMsketch.h"
#include "HyperLogLog.h"

typedef struct
{
	int w;				// the number of columns in Count Min
	int u;				// the size of each elementary data structure in Count Min.
	int *S;		// random seeds for Count Min
	int m;				// number of bit/register in each unit (used for bitmap, FM sketch and HLL sketch)
	int sketchName;		// sketch that is used in the algorithm
	union{
	Bitmap **B;			// bitmap-1
	Counter **C;		// counter-0
	FMsketch **F;		// FMskektch-2
	HyperLogLog **H;	// HLL-3
	}
}GeneralVSketch;


GeneralVSketch *initVSketch(int sketchName);

void generateVSketchRandomSeeds(GeneralVSketch *GVS);

#endif