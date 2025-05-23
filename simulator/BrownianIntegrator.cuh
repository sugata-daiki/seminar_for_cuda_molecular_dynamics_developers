#include <curand_kernel.h>
#include "Constants.cuh"

__global__ void init_random_statesD(
        curandState * __restrict__ d_states, 
        int seed, 
        int num_particles) {
    int particle_idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (particle_idx >= num_particles) return;

    curand_init(seed, particle_idx, 0, &d_states[particle_idx]);
    return;
}

__global__ void BrownianIntegratorD(
        float4 * __restrict__ d_posq, 
        float3 * __restrict__ d_force, 
        float * __restrict__ d_thermalFluctuation, 
        curandState * __restrict__ d_states, 
        int num_particles) {

    int particle_idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (particle_idx >= num_particles) return;

    curandState localState = d_states[particle_idx];

    float4 posq = d_posq[particle_idx];
    float3 pos = params_.inv_kbT*d_force[particle_idx] + d_thermalFluctuation[particle_idx]*make_float3(
            curand_normal(&localState), 
            curand_normal(&localState), 
            curand_normal(&localState))
        + make_float3(posq.x, posq.y, posq.z);

    d_states[particle_idx] = localState;

    float4 new_posq = make_float4(pos.x, pos.y, pos.z, posq.w);
    d_posq[particle_idx] = new_posq;
    d_force[particle_idx] = make_float3(0.0f, 0.0f, 0.0f);
}
