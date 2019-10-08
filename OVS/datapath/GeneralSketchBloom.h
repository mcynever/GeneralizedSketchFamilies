#ifndef GENERALSKETCHBLOOM_H
#define GENERALSKETCHBLOOM_H
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
	void *instance;
	Bitmap ***B;			// bitmap-1
	Counter ***C;		// counter-0
	FMsketch ***F;		// FMskektch-2
	HyperLogLog ***H;	// HLL-3
	};
}GeneralSketchBloom;


GeneralSketchBloom *initSketchBloom(int sketchName);

void generateSketchBloomRandomSeeds(GeneralSketchBloom *GSB);


#endif
