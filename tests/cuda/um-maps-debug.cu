#include <stdlib.h>
#include <stdio.h>

static void copy_file(const char *filename1, const char *filename2) {
    FILE *file_in, *file_out;
    char buf[4096];
    size_t nbytes = 0;

    file_in = fopen(filename1, "r");
    file_out = fopen(filename2, "w");

    while (fread(buf, sizeof buf, 1, file_in) == 1)
        fwrite(buf, sizeof buf, 1, file_out);

    while ((nbytes = fread(buf, 1, sizeof buf, file_in)) > 0)
        fwrite(buf, nbytes, 1, file_out);

    fclose(file_in);
    fclose(file_out);
}

static void dump_maps(unsigned lineno) {
    char fname_buf[64];

    snprintf(fname_buf, sizeof fname_buf, "maps-ln%u.txt", lineno);
    copy_file("/proc/self/maps", fname_buf);

    snprintf(fname_buf, sizeof fname_buf, "smaps-ln%u.txt", lineno);
    copy_file("/proc/self/smaps", fname_buf);
}

__global__ void do_stuff(float *dev_A, size_t size) {
    /* blockDim.{x,y} is a constant = the dimension of the grid */
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < size) {
        dev_A[i] = 13;
    }
}

int main() {
    float *managed;
    float *hostbuf;
    unsigned nelems = 1024;
    size_t sz = nelems * nelems * sizeof *managed;

    dump_maps(__LINE__);

    cudaMallocManaged(&managed, sz);
    dump_maps(__LINE__);

    hostbuf = (float *) malloc(sz);
    dump_maps(__LINE__);

    // access the data on the host
    memcpy(hostbuf, managed, sz);
    dump_maps(__LINE__);
    
    // access the data on the device
    do_stuff<<<(nelems * nelems + 255)/256 /*blocks*/, 256/*threads*/>>>(managed, nelems * nelems);
    dump_maps(__LINE__);

    // access the data on the host again
    memcpy(hostbuf, managed, sz);
    dump_maps(__LINE__);

    cudaFree(managed);
    dump_maps(__LINE__);
    free(hostbuf);
    dump_maps(__LINE__);
}
