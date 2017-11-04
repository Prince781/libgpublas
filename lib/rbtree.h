#pragma once

#include <stdbool.h>
#include <stdio.h>

struct rbtree {
    const void *item;       /* generic item */
    bool red;               /* if false, then it is black */
    struct rbtree *parent;  /* parent of this node */
    struct rbtree *lchild;  /* left child */
    struct rbtree *rchild;  /* right child */
};

/**
 * Return values:
 *      negative    -> a < b
 *      zero        -> a == b
 *      positive    -> a > b
 */
typedef int (*compare_t)(const void *a, const void *b);

#ifdef __cplusplus
extern "C" {
#endif


/**
 * Given a pointer to a root pointer (the root pointer may be NULL), an item, and a comparator,
 * inserts an item into the tree in O(log n) time.
 *
 * Returns: a pointer to the newly-allocated node.
 */
struct rbtree *
rbtree_insert(struct rbtree **root, const void *item, compare_t comparator);

/**
 * Given a pointer to a root pointer (the root pointer may be NULL), a needle, and a comparator,
 * returns the first node whose content matches {needle}.
 *
 * Returns: a pointer to the matching node.
 */
struct rbtree *
rbtree_find(struct rbtree **root, const void *needle, compare_t comparator);

/**
 * Given a pointer to a root pointer (the root pointer may be NULL), and a node, 
 * removes the node from the tree and returns the data.
 *
 * Returns: a pointer to the parent of the deleted node, which may be NULL.
 */
struct rbtree *
rbtree_delete(struct rbtree **root, struct rbtree *node);

typedef void (*dtor_t)(void *elem, void *user_data);

/**
 * Given a pointer to a pointer to a tree (which may be NULL), removes all elements 
 * in the tree and calls destructor(elem, user_data). Either destructor or user_data
 * may be NULL.
 */
void
rbtree_destroy(struct rbtree **root, dtor_t destructor, void *user_data);

typedef void (*print_t)(const void *item, int n, char buf[n]);

/**
 * Print out to a dotfile.
 */
void
rbtree_print(struct rbtree *root, print_t printer, FILE *stream);

#ifdef __cplusplus
};
#endif
