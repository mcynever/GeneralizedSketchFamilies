#ifndef FMSKETCH_H
#define FMSKETCH_H
#include "GeneralUtil.h"

typedef struct
{
	int FMsketchSize;			// size of FMsketch array
	int m;						// number of FMsketches
	bool **FMsketchMatrix;
}FMsketch;


FMsketch *newFMsketch(int m, int size);

void encodeFMsketch(const FMsketch *b);

void encodeFMsketchEID(const FMsketch *b, int elementID);

void encodeFMsketchSegment(const FMsketch *b, int flowID, int *s, int w);

void encodeFMsketchSegmentEID(const FMsketch *b, int flowID, int elementID, int *s, int w);

#endif
