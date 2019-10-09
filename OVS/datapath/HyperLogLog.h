#ifndef HYPERLOGLOG_H
#define HYPERLOGLOG_H
#include "GeneralUtil.h"

typedef struct
{
	int HLLSize;			// size of HyperLogLog array
	int m;						// number of HyperLogLoges
	int maxRegisterValue;
	bool **HLL;
	// double alpha;
}HyperLogLog;

HyperLogLog *newHyperLogLog(int m, int size);

void encodeHyperLogLog(const HyperLogLog *b);

void encodeHyperLogLogEID(const HyperLogLog *b, int elementID);

void encodeHyperLogLogSegment(const HyperLogLog *b, int flowID, int *s, int w);

void encodeHyperLogLogSegmentEID(const HyperLogLog *b, int flowID, int elementID, int *s, int w);

// double getAlpha(int m);

int getBitsetValue(bool *b);

void setBitsetValue(const HyperLogLog *b, int index, int value);

#endif