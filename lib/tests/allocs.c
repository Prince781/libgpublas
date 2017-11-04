#include <stdlib.h>

#define N 128

int main() {
    void *ptrs[N];

    for (int i=0; i<N; ++i)
        ptrs[i] = malloc((i+1) * 10);

    for (int i=0; i<N; ++i)
        free(ptrs[i]);

    return 0;
}
