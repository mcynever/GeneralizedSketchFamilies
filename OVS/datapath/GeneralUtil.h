#ifndef GENERALUTIL_H
#define GENERALUTIL_H
// #include <stdlib.h>
// #include <time.h>
// #include <stdio.h>
// #include <stdbool.h>
// #include <stdint.h>
#include <linux/types.h>
#include <linux/slab.h>
#include <linux/random.h>
#include <linux/log2.h>
#include <linux/delay.h>
#include <linux/hash.h>

#define malloc(size) kzalloc(size, GFP_KERNEL)
#define printf printk
#define new(type) (type*)malloc(sizeof(type))
#define newArr(type, size) (type*)kzalloc(size * sizeof(type), GFP_KERNEL)
// #define min(a, b) ((a < b) ? a : b)

static inline uint32_t rand(void) {
    uint32_t i = 0;
    get_random_bytes(&i, sizeof(uint32_t));
    return i>>1;
}

int intHash(int key);

uint32_t uIntHash(uint32_t a);

int numberOfLeadingZeros(unsigned x);

#endif
