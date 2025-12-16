#include <stdio.h>

#define N 0x100000
#define PARALLEL_THREADS 8
#define NUM_BLOCKS (N + PARALLEL_THREADS - 1) / PARALLEL_THREADS

#define Type_t float

template <typename T>
__global__ void add(T* a, T* b, T* c) {

    size_t index = threadIdx.x + (blockIdx.x * blockDim.x);

    if (index < N) {
        c[index] = a[index] + b[index];
    }
}

int main() {

    Type_t* d_a;
    Type_t* d_b;
    Type_t* d_c;

    // Allocate Device memory
    cudaMalloc((void **) &d_a, N * sizeof(Type_t));
    cudaMalloc((void **) &d_b, N * sizeof(Type_t));
    cudaMalloc((void **) &d_c, N * sizeof(Type_t));

    // Allocate Host memory
    Type_t* a = new Type_t[N];
    Type_t* b = new Type_t[N];
    Type_t* c = new Type_t[N];

    // Random values for array
    size_t value = 0;
    for (size_t ii = 0; ii < N; ++ii) {
        a[ii] = value;
        b[ii] = value++;
    }

    // Host -> Dev
    cudaMemcpy(static_cast<void*>(d_a), static_cast<void*>(a), N * sizeof(Type_t), cudaMemcpyHostToDevice);
    cudaMemcpy(static_cast<void*>(d_b), static_cast<void*>(b), N * sizeof(Type_t), cudaMemcpyHostToDevice);

    // kernel execution
    add<Type_t><<<NUM_BLOCKS, PARALLEL_THREADS>>>(d_a, d_b, d_c);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Dev -> Host
    cudaMemcpy(c, d_c, N * sizeof(Type_t), cudaMemcpyDeviceToHost);

    printf("Last sum[%lu] = %d\n", N - 1, static_cast<int>(c[N-1]));

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
