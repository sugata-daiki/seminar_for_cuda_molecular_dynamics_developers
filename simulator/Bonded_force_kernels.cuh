#ifndef __BONDED_FORCE_KERNELS_CUH__
#define __BONDED_FORCE_KERNELS_CUH__

#include <cuda_runtime.h>
#include "Forcefields.h"
#include "Operations.cuh"
#include "Force_functors.cuh"

template <bool EnableHarmonicBond, typename T, typename... Args>
__global__ void calculateBondedForcesD(
		T * __restrict__ d_force, 
		const int num_bonds, 
		Args... args) {

	int bond_idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (bond_idx >= num_bonds) return;

	int2 pair = make_int2(0, 0);
	float3 force = make_float3(0.0f, 0.0f, 0.0f);

	if constexpr (EnableHarmonicBond) {
        
        HarmonicBondForceField harmonicForceField_(args...);
        HarmonicBondInfo harmonicBondInfo_ = harmonicForceField_(bond_idx);
		pair = harmonicBondInfo_.p;
		force = harmonicBondInfo_.f;
        
		atomicAdd(&d_force[pair.x].x, force.x);
		atomicAdd(&d_force[pair.x].y, force.y);
		atomicAdd(&d_force[pair.x].z, force.z);

		atomicAdd(&d_force[pair.y].x, - force.x);
		atomicAdd(&d_force[pair.y].y, - force.y);
		atomicAdd(&d_force[pair.y].z, - force.z);
	}
	return;
}

#endif // __BONDED_FORCE_KERNELS_CUH__
