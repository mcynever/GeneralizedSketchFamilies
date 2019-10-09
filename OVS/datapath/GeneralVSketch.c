#include "GeneralVSketch.h"


GeneralVSketch *initVSketch(int sketchName)
{
	int M = 1024 * 1024 * 4; 	// total memory space Mbits	
	/** parameters for counters **/
	const int mValueCounter = 128;
	const int counterSize = 32;

	/** parameters for bitmap **/
	const int bitmapSize = 1;	// sharing at bit level
	const int virtualArrayLength = 20000;

	/** parameters for FM sketch **/
	const int mValueFM = 128;
	const int FMsketchSize = 32;

	/** parameters for hyperLogLog **/
	const int mValueHLL = 128;
	const int HLLSize = 5;

	GeneralVSketch *GVS = new(GeneralVSketch);
	GVS->sketchName = sketchName;
	if (sketchName == 0) {
		GVS->m = mValueCounter;
		GVS->u = counterSize;
		GVS->w = M / counterSize;
		GVS->C = (Counter **)malloc(1 * sizeof(Counter*));
		GVS->C[0] = newCounter(GVS->w, GVS->u);
		printf("%s\n", "\nGeneral VSketch-Counter Initialized!");
	}
	else if (sketchName == 1) {
		GVS->m = virtualArrayLength;
		GVS->u = bitmapSize;
		GVS->w = (M / GVS->u);
		GVS->B = (Bitmap **)malloc(1 * sizeof(Bitmap*));
		GVS->B[0] = newBitmap(GVS->w, GVS->u);
		printf("%s\n", "\nGeneral VSketch-Bitmap Initialized!");
	}
	else if (sketchName == 2) {
		GVS->m = mValueFM;
		GVS->u = FMsketchSize;
		GVS->w = M / GVS->u;
		GVS->F = (FMsketch **)malloc(1 * sizeof(FMsketch*));
		GVS->F[0] = newFMsketch(GVS->w, GVS->u);
		printf("%s\n", "\nGeneral VSketch-FMsketch Initialized!");
	}
	else if (sketchName == 3) {
		GVS->m = mValueHLL;
		GVS->u = HLLSize;
		GVS->w = M / GVS->u/6;
		GVS->H = (HyperLogLog **)malloc(1 * sizeof(HyperLogLog*));
		printf("1!\n");
		GVS->H[0] = newHyperLogLog(GVS->w, GVS->u);
		printf("%s\n", "\nGeneral VSketch-HyperLogLog Initialized!");
	}
	else {
		printf("%s\n", "Unsupported Data Structure!!");
	}
	generateVSketchRandomSeeds(GVS);
	return GVS;
}

void generateVSketchRandomSeeds(GeneralVSketch *GVS)
{
	GVS->S = newArr(int, GVS->m);
	int i = 0;
	for (i = 0; i < GVS->m; i++) {
		*(GVS->S + i) = rand();
	}
}
