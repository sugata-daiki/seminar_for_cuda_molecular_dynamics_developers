#ifndef __FORCE_FUNCTORS_H__
#define __FORCE_FUNCTORS_H__

#include <cuda_runtime.h>
#include "Simulation.cuh"

struct HarmonicBondInfo {
	int2 p;
	float3 f;
};

struct HarmonicBondForceField {
	const float * __restrict__ device_r0; 
	const float * __restrict__ device_kb; 
	const int2 * __restrict__ device_pair;
	const float4 * __restrict__ device_posq; 

	___host__ __device__ HarmonicBondForceField(
			const float * __restrict__ d_r0, 
			const float * __restrict__ d_kb, 
			const int2 * __restrict__ d_pair, 
			const float4 * __restrict__ d_posq, 

		d_r0(device_r0),
		d_kb(device_kb), 
		d_pair(device_pair), 
		d_posq(device_posq) {}

	__forceinline__ __device__ HarmonicBondInfo operator()(int bond_idx) const {
       		int2 pair_index = d_pair[bond_idx];
		float r0 = d_r0[bond_idx];
		float kb = d_kb[bond_idx];

		float4 posq_i = d_posq[pair_index.x];
		float4 posq_j = d_posq[pair_index.y];
		
		float3 pos_i = {posq_i.x, posq_i.y, posq_i.z};
		float3 pos_j = {posq_j.x, posq_j.y, posq_j.z};

		float3 rel_coord = pos_i - pos_j;
		float r2 = rel_coord.x*rel_coord.x + rel_coord.y*rel_coord.y + rel_coord.z*rel_coord.z;
		float inv_r = rsqrtf(r2);
		
		HarmonicBondInfo harmonicBondInfo_;
		harmonicBondInfo_.p = pair;
	       	harmonicBondInfo_.f = (r0*inv_r - 1.0)*kb*rel_coord;
		return harmonicBondInfo_;
	}
};

#endif
