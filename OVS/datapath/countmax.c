#include "countmax.h"

struct countmax_line {
    size_t w;
    uint32_t mask;
    struct flow_key* keys;
    elemtype* counters;
};
static struct countmax_line* new_countmax_line(int w);
static void delete_countmax_line(struct countmax_line* this);
static void countmax_line_update(struct countmax_line* this, struct flow_key* key, elemtype value);
static elemtype countmax_line_query(struct countmax_line* this, struct flow_key* key);


struct countmax_sketch* new_countmax_sketch(int w, int d) {
    struct countmax_sketch* sketch = new(struct countmax_sketch);
    sketch->w = w;
    sketch->d = d;
    sketch->lines = newarr(struct countmax_line*, d);
    int i = 0;
    for (i = 0; i < d; i++) {
        sketch->lines[i] = new_countmax_line(w);
    }
    return sketch;
}

void delete_countmax_sketch(struct countmax_sketch* this) {
    int i = 0;
    for (i = 0; i < this->d; i++) {
        delete_countmax_line(this->lines[i]);
    }
    kfree(this->lines);
}

void countmax_sketch_update(struct countmax_sketch* this, struct flow_key* key,
    elemtype value) {
    int i = 0;
    for (i = 0; i < this->d; i++) {
        countmax_line_update(this->lines[i], key, value);
    }
}

elemtype countmax_sketch_query(struct countmax_sketch* this,
    struct flow_key* key) {
    elemtype max = 0;
    int i = 0;
    for (i = 0; i < this->d; i++) {
        elemtype q = countmax_line_query(this->lines[i], key);
        if (q > max) {
            max = q;
        }
    }
    return max;
}




static struct countmax_line* new_countmax_line(int w) {
    struct countmax_line* line = new(struct countmax_line);
    line->counters = newarr(elemtype, w);
    line->keys = newarr(struct flow_key, w);
    line->w = w;
    uint32_t rand = 0;
    get_random_bytes(&rand, sizeof(uint32_t));
    line->mask = rand;
    return line;
}

static void delete_countmax_line(struct countmax_line* this) {
    kfree(this->counters);
    kfree(this->keys);
    kfree(this);
}

static void countmax_line_update(struct countmax_line* this, struct flow_key* key, elemtype value) {
    size_t index = (uint32_t)flow_key_hash_old(key) % this->w;
    struct flow_key* current_key = &(this->keys[index]);
    if (flow_key_equal(key, current_key)) {
        this->counters[index] += value;
    }
    else {
        elemtype now = this->counters[index];
        if (value > now) {
            this->counters[index] = value - now;
            this->keys[index] = *key;
        }
        else {
            this->counters[index] -= value;
        }
    }
}

static elemtype countmax_line_query(struct countmax_line* this, struct flow_key* key) {
    size_t index = (uint32_t)flow_key_hash_old(key) % this->w;
    struct flow_key* current_key = &(this->keys[index]);
    if (flow_key_equal(key, current_key)) {
        return this->counters[index];
    }
    else {
        return 0;
    }
}
