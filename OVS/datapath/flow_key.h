#ifndef FLOW_KEY_H
#define FLOW_KEY_H 1

#include "util.h"

struct flow_key {
    uint32_t srcip;
    uint32_t dstip;
    uint16_t srcport;
    uint16_t dstport;
    uint16_t protocol;
};

static struct flow_key empty_key;

static inline struct flow_key rand_flow_key(void) {
    struct flow_key key;
    key.srcip = rand_uint32();
    key.dstip = rand_uint32();
    key.srcport = rand_uint16();
    key.dstport = rand_uint16();
    key.protocol = rand_uint16();
    return key;
}


static inline uint32_t flow_key_hash_old(struct flow_key* key) {
    int hashCode = (int)key->srcip;
    hashCode = (hashCode * 397) ^ (int)key->dstip;
    hashCode = (hashCode * 397) ^ (int)key->srcport;
    hashCode = (hashCode * 397) ^ (int)key->dstport;
    hashCode = (hashCode * 397) ^ (int)key->protocol;
    return (uint32_t)hashCode;
}

static inline uint32_t flow_key_hash(struct flow_key* key, uint32_t bits) {
    uint32_t hash = sketch_hash_32(key->srcip, bits);
    hash ^= sketch_hash_32(key->dstip, bits);
    hash ^= sketch_hash_32(key->srcport, bits);
    hash ^= sketch_hash_32(key->dstport, bits);
    hash ^= sketch_hash_32(key->protocol, bits);
    return hash;
}

static inline int flow_key_equal(struct flow_key* lhs, struct flow_key* rhs) {
    return lhs->srcip == rhs->srcip && lhs->dstip == rhs->dstip && lhs->srcport == rhs->srcport &&
        lhs->dstport == rhs->dstport && lhs->protocol == rhs->protocol;
}

#endif