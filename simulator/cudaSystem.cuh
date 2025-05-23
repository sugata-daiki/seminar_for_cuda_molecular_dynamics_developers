#ifndef __CUDASYSTEM_CUH__
#define __CUDASYSTEM_CUH__

#include "Bonded_force_kernels.cuh"
#include "Containers.cuh"

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

}

#endif // __CUDASYSTEN_CUH__
