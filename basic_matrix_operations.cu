#include <stdio.h>
#include <cassert>
#include "matrix_operations_v1.cuh"

using Type_t = int;

static constexpr size_t NUM_PROCS = 0x100000;
static constexpr size_t N_IN_BYTES = NUM_PROCS * sizeof(Type_t);
static constexpr size_t PARALLEL_THREADS  = 8;
static constexpr size_t NUM_BLOCKS = (NUM_PROCS + PARALLEL_THREADS - 1) / PARALLEL_THREADS;

int main() {

    Type_t* d_firstMatrix, * d_secondMatrix, * d_result;

    // Allocate Device memory
    cudaMalloc((void **) &d_firstMatrix, N_IN_BYTES);
    cudaMalloc((void **) &d_secondMatrix, N_IN_BYTES);
    cudaMalloc((void **) &d_result, N_IN_BYTES);
    CudaTools::checkForError();

    // Allocate Host memory
    Type_t* firstMatrix = new Type_t[NUM_PROCS];
    Type_t* secondMatrix  = new Type_t[NUM_PROCS];
    Type_t* result = new Type_t[NUM_PROCS];

    // Random values for array
    size_t value = 0;
    for (size_t ii = 0; ii < NUM_PROCS; ++ii) {
        firstMatrix[ii] = value;
        secondMatrix[ii] = value++;
    }

    // Host -> Dev
    cudaMemcpy(static_cast<void*>(d_firstMatrix), static_cast<void*>(firstMatrix), N_IN_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(static_cast<void*>(d_secondMatrix), static_cast<void*>(secondMatrix), N_IN_BYTES, cudaMemcpyHostToDevice);
    CudaTools::checkForError();

    // kernel execution
    MatrixOperations::substract<Type_t><<<NUM_BLOCKS, PARALLEL_THREADS>>>(d_firstMatrix, d_secondMatrix, d_result, NUM_PROCS);
    CudaTools::checkForError();

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Dev -> Host
    cudaMemcpy(result, d_result, N_IN_BYTES, cudaMemcpyDeviceToHost);
    CudaTools::checkForError();

    cudaFree(d_firstMatrix);
    cudaFree(d_secondMatrix);
    cudaFree(d_result);

    return 0;
}
