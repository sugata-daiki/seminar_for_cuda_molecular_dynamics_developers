#ifndef __CONTAINERS_CUH__
#define __CONTAINERS_CUH__

#include "Forcefields.h"

#include <curand_kernel.h>

typedef struct {

    thrust::device_vector<float4> posq;
    thrust::device_vector<float3> force;
    thrust::device_vector<float> r0;
    thrust::device_vector<float> kb;
    thrust::device_vector<int2> pair;
    thrust::device_vector<float> thermalFluctuation;
    thrust::device_vector<curandState> states;
} GpuData;

typedef struct {
    thrust::host_vector<float4> posq;
    thrust::host_vector<float> thermalFluctuation;
    EnableBondedForceFields EnableBondedForceFields_;
    float timestep = 0.0;
    int seed = 0;
    int num_particles = 0;
    int num_bonds;
} CpuData;

#endif 
