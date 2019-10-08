#ifndef BITMAP_H
#define BITMAP_H
#include "GeneralUtil.cuh"
typedef struct
{
	int *m;
	int *arraySize;			// size of bitmap array
	bool *B;			// bit array
}Bitmap;

Bitmap *newBitmap(int m, int size);

//void deleteBitmap();

__device__ void encodeBitmap(Bitmap *b);

__device__ void encodeBitmapEID(const Bitmap *b, int elementID);

__device__ void encodeBitmapSegment(const Bitmap *b, int flowID, int *s, int w);

__device__ void encodeBitmapSegmentEID(const Bitmap *b, int flowID, int elementID, int *s, int w);

#endif