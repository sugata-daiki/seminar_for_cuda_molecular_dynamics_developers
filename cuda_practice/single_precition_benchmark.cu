#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <iostream>

__forceinline__ __device__ float4 operator+(const float4& a, const float4& b) {
    float4 c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    c.z = a.z + b.z;
    c.w = a.w + b.w;
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


    thrust::device_vector<float4> arr_A(size);
    thrust::device_vector<float4> arr_B(size);
    thrust::device_vector<float4> arr_C(size);

    float4 * dA = thrust::raw_pointer_cast(arr_A.data());
    float4 * dB = thrust::raw_pointer_cast(arr_B.data());
    float4 * dC = thrust::raw_pointer_cast(arr_C.data());

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
