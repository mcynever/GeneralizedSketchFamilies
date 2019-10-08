#include "FMsketch.h"

const double phi = 0.77351;

FMsketch *newFMsketch(int m, int size) {
	FMsketch *b = malloc(sizeof(FMsketch));
	//srand(time(NULL));
	b->FMsketchSize = size;
	b->m = m;
	b->FMsketchMatrix = (bool **)malloc(m * sizeof(bool*));
	int i = 0;
	for (i = 0; i < m; i++) {
		b->FMsketchMatrix[i] = (bool *)malloc(size * sizeof(bool));
		int j;
		for (j = 0; j < size; j++) {
			b->FMsketchMatrix[i][j] = false;
		}
	}
	// printf("%s\n", "This is an FMsketch");
	return b;
}

void encodeFMsketch(const FMsketch *b) {
	int r = rand();
	int k = r % (b->m);
	//unsigned hash_val = (unsigned) hash(distribution(generator));
	unsigned hash_val = uIntHash(r);
	int leadingZeros = min(numberOfLeadingZeros(hash_val), b->FMsketchSize - 1);
	(b->FMsketchMatrix)[k][leadingZeros] = 1;
	// printf("%s\n", "This is an FMsketch");
}

void encodeFMsketchEID(const FMsketch *b, int elementID) {
	unsigned hash_val = uIntHash((unsigned)elementID);
	int leadingZeros = min(numberOfLeadingZeros(hash_val), b->FMsketchSize - 1);
	int k = (hash_val % b->m + b->m) % b->m;
	b->FMsketchMatrix[k][leadingZeros] = 1;
}

void encodeFMsketchSegment(const FMsketch *b, int flowID, int *s, int w) {
	int ms = b->m / w;
	int r = rand();
	int j = r % ms;			// (GeneralUtil::intHash(flowID) % ms + ms) % ms;
	int k = (intHash(flowID ^ s[j]) % w + w) % w;
	int i = j * w + k;
	unsigned hash_val = uIntHash(r);							// (unsigned)
	int leadingZeros = min(numberOfLeadingZeros(hash_val), b->FMsketchSize - 1);
	b->FMsketchMatrix[i][leadingZeros] = 1;
}

void encodeFMsketchSegmentEID(const FMsketch *b, int flowID, int elementID, int *s, int w) {
	int m = b->m / w;
	int j = (intHash(elementID) % m + m) % m;
	int k = (intHash(flowID ^ s[j]) % w + w) % w;
	int i = j * w + k;
	unsigned hash_val = uIntHash(elementID);						// (unsigned)
	int leadingZeros = min(numberOfLeadingZeros(hash_val), b->FMsketchSize - 1);
	b->FMsketchMatrix[i][leadingZeros] = 1;
}
