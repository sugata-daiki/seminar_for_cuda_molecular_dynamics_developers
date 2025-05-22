#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <iostream>

typedef struct __align__(8) {
    __half x;
    __half y;
    __half z;
    __half w;
} half4;

__forceinline__ __device__ half4 operator+(const half4& a, const half4& b) {
    half4 c;
    c.x = __hadd(a.x, b.x);
    c.y = __hadd(a.y, b.y);
    c.z = __hadd(a.z, b.z);
    c.w = __hadd(a.w, b.w);
    return c;
}

template <typename T>
__global__ void add_arr(
        const T * __restrict__ dA, 
        const T * __restrict__ dB, 
        T * __restrict__ dC, 
        size_t size) {

    unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;

    if (index > size) return;

    dC[index] = dA[index] + dB[index];

    return;
}

int main(void) {

    size_t num_iterations = 100000;
    size_t size = 100000;
    size_t numThreads = 32;
    size_t numBlocks = (size + 32 - 1)/32;


    thrust::device_vector<half4> arr_A(size);
    thrust::device_vector<half4> arr_B(size);
    thrust::device_vector<half4> arr_C(size);

    half4 * dA = thrust::raw_pointer_cast(arr_A.data());
    half4 * dB = thrust::raw_pointer_cast(arr_B.data());
    half4 * dC = thrust::raw_pointer_cast(arr_C.data());

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (size_t i = 0; i < num_iterations; i++) {
        add_arr<<<numBlocks, numThreads>>>(
                dA,
                dB, 
                dC, 
                size);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Elapsed time: " << milliseconds << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
