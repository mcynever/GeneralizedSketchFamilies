#include "GeneralUtil.h"


int intHash(int key) {
	key += ~(key << 15);
	key ^= (key >> 10);
	key += (key << 3);
	key ^= (key >> 6);
	key += ~(key << 11);
	key ^= (key >> 16);
	return key;
}


uint32_t uIntHash(uint32_t a)
{
	a = (a + 0x7ed55d16) + (a << 12);
	a = (a ^ 0xc761c23c) ^ (a >> 19);
	a = (a + 0x165667b1) + (a << 5);
	a = (a + 0xd3a2646c) ^ (a << 9);
	a = (a + 0xfd7046c5) + (a << 3);
	a = (a ^ 0xb55a4f09) ^ (a >> 16);
	return a;
}

int numberOfLeadingZeros(unsigned x)
{
	int n = 0;
	if (x <= 0x0000ffff) n += 16, x <<= 16;
	if (x <= 0x00ffffff) n += 8, x <<= 8;
	if (x <= 0x0fffffff) n += 4, x <<= 4;
	if (x <= 0x3fffffff) n += 2, x <<= 2;
	if (x <= 0x7fffffff) n++;
	return n;
}
