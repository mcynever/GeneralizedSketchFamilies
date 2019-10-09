#include "Counter.h"

Counter *newCounter(int m, int size) {
	Counter *c = malloc(sizeof(Counter));
	c->counterSize = size;
	c->m = m;
	c->maxValue = (int)(1 << size) - 1;
	c->counters = (int *)malloc(m * sizeof(int));
	int i = 0;
	for (i = 0; i < m; i++) {
		c->counters[i] = 0;
	}
	return c;
}

void encodeCounter(const Counter *c) { // c is null
	if(!c){printk(KERN_WARNING"c is NULL!\n"); return;}
	int r = rand();
	int k = r % c->m;
	if(!c->counters){printk(KERN_WARNING"c->counter is NULL!\n"); return;}	
	c->counters[k]++;
}

void encodeCounterEID(const Counter *c, int elementID) {
	int k = (intHash(elementID) % c->m + c->m) % c->m;
	c->counters[k]++;
}

void encodeCounterSegment(const Counter *c, int flowID, int *s, int w) {
	int ms = c->m / w;
	int r = rand();
	int j = r % ms;	
	int k = (intHash(flowID ^ s[j]) % w + w) % w;
	int i = j * w + k;
	c->counters[i]++;
}

void encodeCounterSegmentEID(const Counter *c, int flowID, int elementID, int *s, int w) {
	int m = c->m / w;
	int j = (intHash(elementID) % m + m) % m;
	int k = (intHash(flowID ^ s[j]) % w + w) % w;
	int i = j * w + k;
	c->counters[i]++;
}
