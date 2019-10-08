#include "Bitmap.h"

Bitmap *newBitmap(int m, int size) {
	Bitmap *b = malloc(sizeof(Bitmap));
	//srand(time(NULL));
	b->m = m;
	b->arraySize = m;
	b->B = (bool *)malloc(m * sizeof(bool));
	int i = 0;
	for (i = 0; i < m; i++) {
		b->B[i] = false;
	}
	// printf("%s\n", "This is a bitmap");
	return b;
}

void encodeBitmap(const Bitmap *b) {
	if(!b){printk(KERN_EMERG"b is NULL!\n"); return;}
	int r = rand();
	int k = r % b->arraySize;
	if(k<0){printk(KERN_EMERG"k less than zero!\n"); return;}
	b->B[k] = true;
	// printf("%s\n", "This is a bitmap");
}

void encodeBitmapEID(const Bitmap *b, int elementID) {
	int k = (intHash(elementID) % b->arraySize + b->arraySize) % b->arraySize;
	b->B[k] = true;
}

void encodeBitmapSegment(const Bitmap *b, int flowID, int *s, int w) {
	int ms = b->arraySize / w;
	int r = rand();
	int j = r % ms;			// (GeneralUtil::intHash(flowID) % ms + ms) % ms;							// rand() % ms;
	int k = (intHash(flowID ^ s[j]) % w + w) % w;
	int i = j * w + k;
	b->B[i] = true;
}

void encodeBitmapSegmentEID(const Bitmap *b, int flowID, int elementID, int *s, int w) {
	int m = b->arraySize / w;
	int j = (intHash(elementID) % m + m) % m;
	int k = (intHash(flowID ^ s[j]) % w + w) % w;
	int i = j * w + k;
	b->B[i] = true;
}
