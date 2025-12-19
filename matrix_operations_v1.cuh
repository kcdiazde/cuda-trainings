
#pragma once

namespace MatrixOperations {

    template <typename T>
    __global__ void add(T* firstMatrix, T* secondMatrix, T* result, const size_t NUM_PROCS) {
    
        const size_t index = threadIdx.x + (blockIdx.x * blockDim.x);

        if (index < NUM_PROCS) {
            result[index] = firstMatrix[index] + secondMatrix[index];
        }
    }

    template <typename T>
    __global__ void substract(T* firstMatrix, T* secondMatrix, T* result, const size_t NUM_PROCS) {
    
        const size_t index = threadIdx.x + (blockIdx.x * blockDim.x);

        if (index < NUM_PROCS) {
            result[index] = firstMatrix[index] - secondMatrix[index];
        }
    }

    template <typename T>
    __global__ void scalarMultiplication(T* matrix, T scalarValue, T* result, const size_t NUM_PROCS) {
    
        const size_t index = threadIdx.x + (blockIdx.x * blockDim.x);

        if (index < NUM_PROCS) {
            result[index] = scalarValue * matrix[index];
        }
    }

}

namespace CudaTools {

    void checkForError() {
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("Error: %s\n", cudaGetErrorString(error));
            exit(1);
        }
    }

}


