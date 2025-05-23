#ifndef __CONTAINERS_CUH__
#define __CONTAINERS_CUH__

#include "Forcefields.h"

typedef struct {

    thrust::device_vector<float4> posq;
    thrust::device_vector<float3> force;
    thrust::device_vector<float> r0;
    thrust::device_vector<float> kb;
    thrust::device_vector<int2> pair;
} GpuData;

typedef struct {
    thrust::host_vector<float4> posq;
    EnableBondedForceFields EnableBondedForceFields_;
    int num_particles = 0;
    int num_bonds;
} CpuData;

#endif 
