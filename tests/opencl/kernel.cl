kernel void add(global float *A, global float *B, global float *C) {
    const int i = get_global_id(0);

    C[i] = A[i] + B[i];
}
