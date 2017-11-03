/**
 * rbtree.c
 *
 * See http://www.geeksforgeeks.org/c-program-red-black-tree-insertion/
 * See http://www.geeksforgeeks.org/red-black-tree-set-3-delete-2/
 */
#include "rbtree.h"
#include <stdio.h>
#include <stdlib.h>

// (2) and (3)
static void rbtree_recolor(struct rbtree *node)
{
    extern void abort(void);

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
            
            if (parent == parent->parent->lchild)
                uncle = parent->parent->rchild;
            else
                uncle = parent->parent->lchild;

            if (uncle->red) {   /* (a) */
                parent->red = false;
                uncle->red = false;

                node = parent->parent;
            } else {    /* (b) uncle is black */
                struct rbtree *t2, *t3, *t4;
                /* left-left */
                if (grandparent->lchild == parent && parent->lchild == node) {
                    t3 = parent->rchild;

                    parent->parent = NULL;
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
                    t3 = node->rchild;

                    grandparent->lchild = node;
                    node->lchild = parent;
                    parent->rchild = t2;

                    node->parent = NULL;
                    node->rchild = grandparent;
                    grandparent->lchild = t3;

                    if (t2)
                        t2->parent = parent;
                    if (t3)
                        t3->parent = grandparent;
                    parent->parent = node;
                    grandparent->parent = node;

                    parent->red = true;
                    node->red = false;
                    grandparent->red = true;
                }
                /* right-right */
                else if (grandparent->rchild == parent && parent->rchild == node) {
                    t3 = parent->lchild;

                    grandparent->rchild = t3;
                    if (t3)
                        t3->parent = grandparent;
                    grandparent->red = true;

                    parent->parent = NULL;
                    parent->lchild = grandparent;
                    parent->red = false;
                    
                    parent->parent = NULL;
                    grandparent->parent = parent;
                }
                /* right->left */
                else if (grandparent->rchild == parent && parent->lchild == node) {
                    t3 = node->lchild;
                    t4 = node->rchild;

                    grandparent->rchild = node;
                    node->rchild = parent;
                    parent->lchild = t4;
                    if (t4)
                        t4->parent = parent;

                    grandparent->rchild = t3;
                    if (t3)
                        t3->parent = grandparent;
                    grandparent->red = true;
                    node->lchild = grandparent;
                    node->red = false;

                    grandparent->parent = node;
                    node->parent = NULL;
                    parent->parent = node;
                } else {
                    fprintf(stderr, "%s: corrupted tree\n", __func__);
                    abort();
                }

                node = NULL;
            }
        }
    }
}

struct rbtree *
rbtree_insert(struct rbtree **root, const void *item, compare_t comparator)
{
    struct rbtree **old_root = root;
    struct rbtree *node = NULL;

    if (*root == NULL) {
        *root = malloc(sizeof(**root));
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
    extern void abort(void);
    struct rbtree *successor;
    const void *data;
    struct rbtree *old_node, *old_parent;

    if (!node)
        return NULL;

    old_node = node;

    if (node->lchild && node->rchild) { /* two children */
        /* find inorder successor (smallest item in right subtree) */
        successor = node->rchild;
        while (successor->lchild)
            successor = successor->lchild;

        /* swap successor <=> node; delete successor */
        data = node->item;
        node->item = successor->item;
        successor->item = data;

        return rbtree_delete(root, successor);
    } else if (node->lchild || node->rchild) {  /* one child */
        successor = node->lchild ? node->lchild : node->rchild;
    } else {    /* no children */
        successor = NULL;
    }

    if (node->red || (successor && successor->red)) {
        /* case (2) */
        if (successor) {
            if (node->red && successor->red) {
                fprintf(stderr, "%s: malformed tree\n", __func__);
                abort();
            }
            successor->red = false; 
        }
        /* if NULL, then it is already black */

        if (node == node->parent->lchild)
            node->parent->lchild = successor;
        else
            node->parent->rchild = successor;

        if (node == *root)
            root = &successor;
    } else {
        /* case (3) - both node and successor are black */
        /* successor is NULL */
        struct rbtree *sibling;
        struct rbtree *redchild;    /* child of sibling */
        struct rbtree *parent;
        bool dblack = true;     /* is current node double-black? */

        while (dblack || node != *root) {
            /* (3.1) successor node is double-black */
            parent = node->parent;
            /* (3.2) get sibling; do while current node is double-black */
            if (node == parent->lchild)
                sibling = parent->rchild;
            else
                sibling = parent->lchild;

            if (sibling->lchild && sibling->lchild->red)
                redchild = sibling->lchild;
            else if (sibling->rchild && sibling->rchild->red)
                redchild = sibling->rchild;
            else
                redchild = NULL;

            if (!sibling->red && redchild) {
                /* case (a) */
                if (sibling == parent->lchild
                 && redchild == sibling->lchild) {
                    /* case (i) - left-left */
                    parent->lchild = sibling->rchild;
                    if (sibling->rchild)
                        sibling->rchild->parent = parent;
                    sibling->parent = parent->parent;
                    parent->parent = sibling;
                    sibling->rchild = parent;
                    redchild->red = false;
                } else if (sibling == parent->lchild
                        && redchild == sibling->rchild) {
                    /* case (ii) - left-right */
                    redchild->parent = parent->parent;
                    parent->lchild = NULL;
                    sibling->rchild = NULL;
                    redchild->lchild = sibling;
                    sibling->parent = redchild;
                    redchild->rchild = parent;
                    parent->parent = redchild;
                    redchild->red = false;
                    /* sibling->red = false; */
                } else if (sibling == parent->rchild
                        && redchild == sibling->rchild) {
                    /* case (iii) - right-right */
                    parent->rchild = sibling->lchild;
                    if (sibling->lchild)
                        sibling->lchild->parent = parent;
                    sibling->parent = parent->parent;
                    parent->parent = sibling;
                    sibling->lchild = parent;
                    redchild->red = false;
                } else if (sibling == parent->rchild
                        && redchild == sibling->lchild) {
                    /* case (iv) - right-left */
                    redchild->parent = parent->parent;
                    parent->rchild = NULL;
                    sibling->lchild = NULL;
                    redchild->rchild = sibling;
                    sibling->parent = redchild;
                    redchild->lchild = parent;
                    parent->parent = redchild;
                    redchild->red = false;
                } else {
                    fprintf(stderr, "%s: malformed tree\n", __func__);
                    abort();
                }
            } else if (!sibling->red && !redchild) {
                /* case (b) - sibling is black and both children are black */

                /*
                if (node == parent->lchild)
                    parent->lchild = NULL;
                else
                    parent->rchild = NULL;
                */

                if (!parent->red) {
                    /* parent is now double-black
                     * set node = parent and redo this algorithm for node */
                    node = parent;      /* recur for parent node */
                } else {
                    /* parent was red, but is now black */
                    parent->red = false;
                    /* we have solved the problem; stop */
                    dblack = false;
                }

                /* in either case, set sibling to be red */
                sibling->red = true;
            } else if (sibling->red) {
                /* case (c) - sibling is red */
                if (sibling == parent->lchild) {
                    /* (i) left case */
                    parent->lchild = sibling->rchild;
                    if (parent->lchild)
                        parent->lchild->parent = parent;
                    sibling->rchild = parent;
                    parent->parent = sibling;
                } else {    /* sibling is parent->rchild */
                    /* (ii) right case */
                    parent->rchild = sibling->lchild;
                    if (parent->rchild)
                        parent->rchild->parent = parent;
                    sibling->lchild = parent;
                    parent->parent = sibling;
                }
                sibling->red = false;
                parent->red = true;
            } else {
                fprintf(stderr, "%s: malformed tree\n", __func__);
                abort();
            }
        }
    }

    if (node == *root && node)
        node->red = false;  /* set root to be black */

    old_parent = old_node->parent;

    free(old_node);

    return old_parent;
}

void
rbtree_destroy(struct rbtree **root, dtor_t destructor, void *user_data)
{
    if (!*root)
        return;

    rbtree_destroy(&(*root)->lchild, destructor, user_data);
    rbtree_destroy(&(*root)->rchild, destructor, user_data);

    if (destructor)
        destructor((void *) (*root)->item, user_data);
    free(*root);
    *root = NULL;
}

static void
rbtree_print2(FILE *stream, struct rbtree *root, print_t printer)
{
    char buf[1024];

    printer(root->item, sizeof(buf), buf);
    buf[sizeof(buf)-1] = '\0';
    fprintf(stream, "node%p [%s];\n", root, buf);

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
rbtree_print(struct rbtree *root, print_t printer)
{
    FILE *stream;

    stream = tmpfile();
    fprintf(stream, "digraph G {\n");
    rbtree_print2(stream, root, printer ? default_printer : printer);
    fprintf(stream, "}\n");

    fclose(stream);
}
