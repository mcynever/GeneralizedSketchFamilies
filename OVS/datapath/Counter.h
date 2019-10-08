#ifndef COUNTER_H
#define COUNTER_H
#include "GeneralUtil.h"

typedef struct
{
	int m;					// number of counters in a counter array
	int counterSize;			// size of counter
	int maxValue;			// maximum value of a counter
	int *counters;
}Counter;


Counter *newCounter(int m, int size);

//void deleteBitmap();

//char *getDataStructureName();
//
//int getUnitSize();

void encodeCounter(const Counter *c);

void encodeCounterEID(const Counter *c, int elementID);

void encodeCounterSegment(const Counter *c, int flowID, int *s, int w);

void encodeCounterSegmentEID(const Counter *c, int flowID, int elementID, int *s, int w);

#endif