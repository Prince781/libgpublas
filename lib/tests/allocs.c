#include <stdlib.h>

#define N 512

int main() {
    void *ptrs[N];

    for (int i=0; i<N; ++i)
        ptrs[i] = malloc((i+1) * 10);

    for (int i=0; i<N*100; i += 99) {
        free(ptrs[(i % N)]);
        ptrs[(i % N)] = NULL;
    }

    return 0;
}
