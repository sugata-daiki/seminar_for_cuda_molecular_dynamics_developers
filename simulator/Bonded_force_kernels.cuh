#ifndef __BONDED_FORCE_KERNELS_CUH__
#define __BONDED_FORCE_KERNELS_CUH__

#include "Forcefields.h"
#include "Force_functors.h"

template <typename EnableBondedForceField_T, typename T, typename... Args>
__global__ void calculateBondedForces(
		T * __restrict__ d_forces, 
		const int num_bonds, 
		Args... args) {

	int bond_idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (bond_idx >= num_bonds) return;

	int2 pair = make_int2(0, 0);
	float3 force = make_float3(0.0f, 0.0f, 0.0f);

	if constexpr (EnableBondedForceField_T::EnableHarmonicBond) {
		HarmonicBondInfo harmonicBondInfo_ = HarmonicBondForceField(args);
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
