#ifndef BITMAP_H
#define BITMAP_H
#include "GeneralUtil.h"

typedef struct
{
	int m;
	int arraySize;			// size of bitmap array
	bool *B;			// bit array
}Bitmap;


Bitmap *newBitmap(int m, int size);

//void deleteBitmap();

void encodeBitmap(const Bitmap *b);

void encodeBitmapEID(const Bitmap *b, int elementID);

void encodeBitmapSegment(const Bitmap *b, int flowID, int *s, int w);

void encodeBitmapSegmentEID(const Bitmap *b, int flowID, int elementID, int *s, int w);

#endif

