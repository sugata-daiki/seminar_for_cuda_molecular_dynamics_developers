#ifndef __OPERATIONS_CUH__
#define __OPERATIONS_CUH__

#include <cuda_runtime.h>

// 加算
__host__ __device__ inline float3 operator+(const float3 &a, const float3 &b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

// 減算
__host__ __device__ inline float3 operator-(const float3 &a, const float3 &b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

// スカラー倍（float3 * float）
__host__ __device__ inline float3 operator*(const float3 &a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

// スカラー倍（float * float3）
__host__ __device__ inline float3 operator*(float s, const float3 &a) {
    return a * s;
}

// スカラー除算
__host__ __device__ inline float3 operator/(const float3 &a, float s) {
    float inv = 1.0f / s;
    return a * inv;
}

// 内積
__host__ __device__ inline float dot(const float3 &a, const float3 &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

#endif
