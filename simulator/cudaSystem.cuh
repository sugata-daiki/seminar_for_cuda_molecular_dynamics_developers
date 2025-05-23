#ifndef __CUDASYSTEM_CUH__
#define __CUDASYSTEM_CUH__

#include "Bonded_force_kernels.cuh"
#include "BrownianIntegrator.cuh"
#include "Containers.cuh"

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)            \
                      << " in file " << __FILE__ << " at line " << __LINE__   \
                      << std::endl;                                           \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

void calculateBondedForcesH(GpuData & gpu_data, CpuData & cpu_data) {
        unsigned int numThreads = 256;
        unsigned int numBlocks = (cpu_data.num_bonds + numThreads - 1) / numThreads;
        
        float4 * d_posq = thrust::raw_pointer_cast(gpu_data.posq.data());
        float3 * d_force = thrust::raw_pointer_cast(gpu_data.force.data());
        float * d_r0 = thrust::raw_pointer_cast(gpu_data.r0.data());
        float * d_kb = thrust::raw_pointer_cast(gpu_data.kb.data());
        int2 * d_pair = thrust::raw_pointer_cast(gpu_data.pair.data());

        if (cpu_data.EnableBondedForceFields_.EnableHarmonicBond) {
                calculateBondedForcesD<true, float3, float *, float *, int2 *, float4* ><<<numBlocks, numThreads>>>(
                    d_force, 
                    cpu_data.num_bonds, 
                    d_r0, 
                    d_kb, 
                    d_pair, 
                    d_posq);
        }

        CUDA_CHECK(cudaPeekAtLastError());

}

void init_random_statesH(GpuData & gpu_data, CpuData & cpu_data) {
        unsigned int numThreads = 256;
        unsigned int numBlocks = (cpu_data.num_particles + numThreads - 1) / numThreads;
        
        curandState * d_states = thrust::raw_pointer_cast(gpu_data.states.data());
        init_random_statesD<<<numBlocks, numThreads>>>(
                d_states, 
                cpu_data.seed, 
                cpu_data.num_particles);
        CUDA_CHECK(cudaPeekAtLastError());
}

void BrownianIntegratorH(GpuData & gpu_data, CpuData & cpu_data) {
        unsigned int numThreads = 256;
        unsigned int numBlocks = (cpu_data.num_particles + numThreads - 1) / numThreads;
        float4 * d_posq = thrust::raw_pointer_cast(gpu_data.posq.data());
        float3 * d_force = thrust::raw_pointer_cast(gpu_data.force.data());
        
        float * d_diffusion_coefficient = thrust::raw_pointer_cast(gpu_data.diffusion_coefficient.data());
        curandState * d_states = thrust::raw_pointer_cast(gpu_data.states.data());
        
        BrownianIntegratorD<<<numBlocks, numThreads>>>(
                d_posq, 
                d_force, 
                d_diffusion_coefficient, 
                d_states, 
                cpu_data.num_particles);
}
#endif // __CUDASYSTEN_CUH__
