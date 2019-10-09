#ifndef COUNTMAX_H
#define COUNTMAX_H
#include "flow_key.h"

struct countmax_sketch {
    size_t w;
    size_t d;
    struct countmax_line** lines;
};


struct countmax_sketch* new_countmax_sketch(int w, int d);

void countmax_sketch_update(struct countmax_sketch* this, struct flow_key* key, elemtype value);

elemtype countmax_sketch_query(struct countmax_sketch* this, struct flow_key* key);

void delete_countmax_sketch(struct countmax_sketch* this);

// unused code
#ifndef NULL
//struct countmax_manager {
//    size_t w;
//    size_t d;
//    size_t sw_count;
//    struct countmax_sketch** sketches;
//};
//struct countmax_manager* new_countmax_manager(int w, int d, int sw_count) {
//    struct countmax_manager* manager = new(struct countmax_manager);
//    manager->w = w;
//    manager->d = d;
//    manager->sw_count = sw_count;
//    manager->sketches = newarr(struct countmax_sketch*, sw_count);
//    int i = 0;
//    for (i = 0; i < sw_count; i++) {
//        manager->sketches[i] = new_countmax_sketch(w, d);
//    }
//    return manager;
//}
//
//static void delete_countmax_manager(struct countmax_manager* this) {
//    int i = 0;
//    for (i = 0; i < this->sw_count; i++) {
//        delete_countmax_sketch(this->sketches[i]);
//    }
//    kfree(this->sketches);
//}
//
//void countmax_manager_update(struct countmax_manager* this, int sw_id,
//    struct flow_key* key, elemtype value) {
//    if (sw_id < 0 || sw_id >= this->sw_count) {
//        return;
//    }
//    countmax_sketch_update(this->sketches[sw_id], key, value);
//}
//
//elemtype countmax_manager_query(struct countmax_manager* this,
//    struct flow_key* key) {
//    elemtype max = 0;
//    int i = 0;
//    for (i = 0; i < this->sw_count; i++) {
//        elemtype q = countmax_sketch_query(this->sketches[i], key);
//        if (q > max) {
//            max = q;
//        }
//    }
//    return max;
//}
static struct countmax_line* new_countmax_line(int w);
static void countmax_line_update(struct countmax_line* this, struct flow_key* key, elemtype value);
static elemtype countmax_line_query(struct countmax_line* this, struct flow_key* key);
static void delete_countmax_line(struct countmax_line* this);
static struct countmax_sketch* new_countmax_sketch(int w, int d);
static void countmax_sketch_update(struct countmax_sketch* this, struct flow_key* key, elemtype value);
static elemtype countmax_sketch_query(struct countmax_sketch* this, struct flow_key* key);
static void delete_countmax_sketch(struct countmax_sketch* this);
static struct countmax_manager* new_countmax_manager(int w, int d, int sw_count);
static void countmax_manager_update(struct countmax_manager* this, int sw_id, struct flow_key* key, elemtype value);
static elemtype countmax_manager_query(struct countmax_manager* this, struct flow_key* key);
static void delete_countmax_manager(struct countmax_manager* this);
#endif

#endif
