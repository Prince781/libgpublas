/**
 * rbtree.c
 *
 * See http://www.geeksforgeeks.org/c-program-red-black-tree-insertion/
 * See http://www.geeksforgeeks.org/red-black-tree-set-3-delete-2/
 */
#include "rbtree.h"
#include <stdio.h>

void abort(void);

/* [gparn] 
 *    |
 * [node2]
 *    |
 * [node ]
 *
 * set [node] to be the child of [gparn]
 * [node] may be NULL
 */
static inline void reparent(struct rbtree *node, struct rbtree *node2) {
    if (node2->parent) {
        if (node2->parent->lchild == node2)
            node2->parent->lchild = node;
        else if (node2->parent->rchild == node2)
            node2->parent->rchild = node;
    }
    if (node)
        node->parent = node2->parent;
}

// (2) and (3)
static void rbtree_recolor(struct rbtree *node)
{
    while (node) {
        if (!node->parent) { /* (2) */
            node->red = false;
            node = NULL;
        } else if (node->parent->red) { /* (3) */
            struct rbtree *grandparent;
            struct rbtree *parent;
            struct rbtree *uncle;

            parent = node->parent;
            grandparent = parent->parent;
            
            if (parent == grandparent->lchild)
                uncle = grandparent->rchild;
            else
                uncle = grandparent->lchild;

            if (uncle && uncle->red) {   /* (a) */
                parent->red = false;
                uncle->red = false;
                grandparent->red = true;
                node = grandparent;
            } else {    /* (b) uncle is black */
                struct rbtree *t2, *t3, *t4;
                /* left-left */
                if (grandparent->lchild == parent && parent->lchild == node) {
                    t3 = parent->rchild;

                    reparent(parent, grandparent);
                    parent->rchild = grandparent;
                    grandparent->lchild = t3;
                    if (t3)
                        t3->parent = grandparent;
                    grandparent->parent = parent;

                    parent->red = false;
                    grandparent->red = true;
                } 
                /* left-right */
                else if (grandparent->lchild == parent && parent->rchild == node) {
                    t2 = node->lchild;

                    node->parent = grandparent;
                    node->lchild = parent;
                    parent->parent = node;
                    parent->rchild = t2;
                    if (t2)
                        t2->parent = parent;

                    node = parent;
                }
                /* right-right */
                else if (grandparent->rchild == parent && parent->rchild == node) {
                    t3 = parent->lchild;

                    grandparent->rchild = t3;
                    if (t3)
                        t3->parent = grandparent;
                    grandparent->red = true;

                    reparent(parent, grandparent);
                    parent->lchild = grandparent;
                    parent->red = false;
                    grandparent->parent = parent;
                }
                /* right->left */
                else if (grandparent->rchild == parent && parent->lchild == node) {
                    t4 = node->rchild;

                    node->parent = grandparent;
                    node->rchild = parent;
                    parent->parent = node;
                    parent->lchild = t4;
                    if (t4)
                        t4->parent = parent;

                    node = parent;
                } else {
                    fprintf(stderr, "%s: corrupted tree\n", __func__);
                    abort();
                }
            }
        } else
            break;
    }
}

struct rbtree *
rbtree_insert(struct rbtree **root, void *item, compare_t comparator)
{
    extern void *(*real_malloc)(size_t);
    struct rbtree **old_root = root;
    struct rbtree *node = NULL;

    if (*root == NULL) {
        *root = real_malloc(sizeof(**root));
        (*root)->red = false;
        (*root)->item = item;
        (*root)->lchild = NULL;
        (*root)->rchild = NULL;
        (*root)->parent = NULL;
        node = *root;
    } else {
        struct rbtree *parent = NULL;

        while (*root) {
            int res;
            if ((res = comparator(item, (*root)->item)) < 0) {
                parent = *root;
                root = &(*root)->lchild;
            } else if (res > 0) {
                parent = *root;
                root = &(*root)->rchild;
            } else {
                return NULL;
            }
        }

        // (1)
        node = rbtree_insert(root, item, comparator);
        node->parent = parent;
        node->red = true;

        rbtree_recolor(node);        

        /* if (*old_root) was updated, change (*old_root) to 
         * point to the new root of the tree */
        while ((*old_root)->parent)
            *old_root = (*old_root)->parent;
    }

    return node;
}

struct rbtree *
rbtree_find(struct rbtree **root, const void *needle, compare_t comparator)
{
    while (*root) {
        int res;

        if ((res = comparator(needle, (*root)->item)) == 0)
            break;
        else if (res < 0)
            root = &(*root)->lchild;
        else 
            root = &(*root)->rchild;
    }

    return *root;
}

struct rbtree *
rbtree_delete(struct rbtree **root, struct rbtree *node)
{
    /* TODO */
    abort();
}

void
rbtree_destroy(struct rbtree **root, dtor_t destructor, void *user_data)
{
    extern void (*real_free)(void *);
    if (!*root)
        return;

    rbtree_destroy(&(*root)->lchild, destructor, user_data);
    rbtree_destroy(&(*root)->rchild, destructor, user_data);

    if (destructor)
        destructor((void *) (*root)->item, user_data);
    real_free(*root);
    *root = NULL;
}

static void
rbtree_print2(FILE *stream, struct rbtree *root, print_t printer)
{
    char buf[1024];

    printer(root->item, sizeof(buf), buf);
    buf[sizeof(buf)-1] = '\0';
    fprintf(stream, "node%p [%s,color=%s,shape=box];\n", root, buf, root->red ? "red" : "black");

    if (root->lchild) {
        fprintf(stream, "node%p -> node%p;\n", root, root->lchild);
        rbtree_print2(stream, root->lchild, printer);
    }

    if (root->rchild) {
        fprintf(stream, "node%p -> node%p;\n", root, root->rchild);
        rbtree_print2(stream, root->rchild, printer);
    }
}

static void
default_printer(const void *item, int n, char buf[n])
{
    snprintf(buf, n, "label=%p", item);
}

void
rbtree_print(struct rbtree *root, print_t printer, FILE *stream)
{
    bool stream_null = stream == NULL;
    
    if (stream_null)
        stream = tmpfile();
    fprintf(stream, "digraph G {\n");
    if (root)
        rbtree_print2(stream, root, printer ? printer : default_printer);
    fprintf(stream, "}\n");

    if (stream_null)
        fclose(stream);
}
