#ifndef NEW_SKETCHES_H
#define NEW_SKETCHES_H

#include "GeneralSketchBloom.h"
#include "GeneralVSketch.h"
#define MY_SKETCH
static int sketch_name = 3;
static int GSB_or_GVS = 1;
static int size_or_spread = 1;

typedef void (*GSB_size_t)(void* c);
typedef void (*GSB_spread_t)(void* c, int src);
typedef void (*GVS_size_t)(void *c, int src, int *s, int w);
typedef void (*GVS_spread_t)(void *c, int src, int dst, int *s, int w);
static const GSB_size_t gsb_size_f[] = {
	(GSB_size_t)encodeCounter,
	(GSB_size_t)encodeBitmap,
	(GSB_size_t)encodeFMsketch,
	(GSB_size_t)encodeHyperLogLog,
};
static const GSB_spread_t gsb_spread_f[] = {
	(GSB_spread_t)encodeCounterEID,
	(GSB_spread_t)encodeBitmapEID,
	(GSB_spread_t)encodeFMsketchEID,
	(GSB_spread_t)encodeHyperLogLogEID,
};
static const GVS_size_t gvs_size_f[] = {
	(GVS_size_t)encodeCounterSegment,
	(GVS_size_t)encodeBitmapSegment,
	(GVS_size_t)encodeFMsketchSegment,
	(GVS_size_t)encodeHyperLogLogSegment,
};
static const GVS_spread_t gvs_spread_f[] = {
	(GVS_spread_t)encodeCounterSegmentEID,
	(GVS_spread_t)encodeBitmapSegmentEID,
	(GVS_spread_t)encodeFMsketchSegmentEID,
	(GVS_spread_t)encodeHyperLogLogSegmentEID,
};
static inline void print_sketch_mode(void){
	char* sname[] = {"0-counter", "1-bitmap", "2-FMsketch", "3-HLL"};
	char* smode[] = {"SketchBloom", "VSketch"};
	char* smetric[] = {"size", "spread"};
	printk("Sketch: %s, Mode: %s, Metric: %s", sname[sketch_name], smode[GSB_or_GVS], smetric[size_or_spread]);
}
#endif
