#include <stdio.h>
#include <cassert>

using Type_t = int;

static constexpr size_t NUM_PROCS = 0x100000;
static constexpr size_t N_IN_BYTES = NUM_PROCS * sizeof(Type_t);
static constexpr size_t PARALLEL_THREADS  = 8;
static constexpr size_t NUM_BLOCKS = (NUM_PROCS + PARALLEL_THREADS - 1) / PARALLEL_THREADS;

namespace MatrixOperations {

    template <typename T>
    __global__ void add(T* a, T* b, T* c, size_t NUM_PROCS) {
    
        const size_t index = threadIdx.x + (blockIdx.x * blockDim.x);

        if (index < NUM_PROCS) {
            c[index] = a[index] + b[index];
        }
    }

    template <typename T>
    __global__ void substract(T* a, T* b, T* c, size_t NUM_PROCS) {
    
        const size_t index = threadIdx.x + (blockIdx.x * blockDim.x);

        if (index < NUM_PROCS) {
            c[index] = a[index] - b[index];
        }
    }
}

void checkForCudaError() {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(error));
        exit(1);
    }
}

int main() {

    Type_t* d_a, * d_b, * d_c;

    // Allocate Device memory
    cudaMalloc((void **) &d_a, N_IN_BYTES);
    cudaMalloc((void **) &d_b, N_IN_BYTES);
    cudaMalloc((void **) &d_c, N_IN_BYTES);
    checkForCudaError();

    // Allocate Host memory
    Type_t* a = new Type_t[NUM_PROCS];
    Type_t* b = new Type_t[NUM_PROCS];
    Type_t* c = new Type_t[NUM_PROCS];

    // Random values for array
    size_t value = 0;
    for (size_t ii = 0; ii < NUM_PROCS; ++ii) {
        a[ii] = value;
        b[ii] = value++;
    }

    // Host -> Dev
    cudaMemcpy(static_cast<void*>(d_a), static_cast<void*>(a), N_IN_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(static_cast<void*>(d_b), static_cast<void*>(b), N_IN_BYTES, cudaMemcpyHostToDevice);
    checkForCudaError();

    // kernel execution
    MatrixOperations::substract<Type_t><<<NUM_BLOCKS, PARALLEL_THREADS>>>(d_a, d_b, d_c, NUM_PROCS);
    checkForCudaError();

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Dev -> Host
    cudaMemcpy(c, d_c, N_IN_BYTES, cudaMemcpyDeviceToHost);
    checkForCudaError();

    printf("Last sum[%ll] = %d\n", NUM_PROCS - 1, static_cast<int>(c[NUM_PROCS-1]));

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
