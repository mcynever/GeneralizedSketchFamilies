#ifndef NEW_SKETCHES_H
#define NEW_SKETCHES_H
#include "stdio.h"
#include "GeneralSketchBloom.cuh"
#include "GeneralVSketch.cuh"

#define MY_SKETCH
const int number_of_test = 7;
static int sketch_name[number_of_test] =  {0,1,2,3,1,2,3};
static int GSB_or_GVS[number_of_test] =     {1,1,1,1,1,1,1};
static int size_or_spread[number_of_test] = {0,0,0,0,1,1,1};

typedef void(*GSB_size_t)(void* c);
typedef void(*GSB_spread_t)(void* c, int src);
typedef void(*GVS_size_t)(void *c, int src, int *s, int w);
typedef void(*GVS_spread_t)(void *c, int src, int dst, int *s, int w);
extern __device__ const GSB_size_t gsb_size_f[] = {
	(GSB_size_t)encodeCounter,
	(GSB_size_t)encodeBitmap,
	(GSB_size_t)encodeFMsketch,
	(GSB_size_t)encodeHyperLogLog,
};
extern __device__ const GSB_spread_t gsb_spread_f[] = {
	(GSB_spread_t)encodeCounterEID,
	(GSB_spread_t)encodeBitmapEID,
	(GSB_spread_t)encodeFMsketchEID,
	(GSB_spread_t)encodeHyperLogLogEID,
};
extern __device__ const GVS_size_t gvs_size_f[] = {
	(GVS_size_t)encodeCounterSegment,
	(GVS_size_t)encodeBitmapSegment,
	(GVS_size_t)encodeFMsketchSegment,
	(GVS_size_t)encodeHyperLogLogSegment,
};
extern __device__ const GVS_spread_t gvs_spread_f[] = {
	(GVS_spread_t)encodeCounterSegmentEID,
	(GVS_spread_t)encodeBitmapSegmentEID,
	(GVS_spread_t)encodeFMsketchSegmentEID,
	(GVS_spread_t)encodeHyperLogLogSegmentEID,
};
#endif
