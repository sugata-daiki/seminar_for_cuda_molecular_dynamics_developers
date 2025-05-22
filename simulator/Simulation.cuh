#ifndef __SIMULATION_CUH__
#define __SIMULATION_CUH__

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "System.h"

typedef struct {

    thrust::device_vector<float4> posq;
    thrust::device_vector<float3> force;
    thrust::device_vector<float> r0;
    thrust::device_vector<float> kb;
    thrust::device_vector<int2> pair;
    
} GpuData;

typedef struct {
    thrust::host_vector<float4> posq;
    int num_particles = 0;
    int num_bonds;
} CpuData;

class Simulation {
    public:
        Simulation() {};
        ~Simulation() {};

        void run(System system, int num_steps);

        void set_positions(float x, float y, float z, float q);

        int get_num_particles() {return cpuData_.num_particles;}

	void set_num_bonds(int num_bonds) {
		cpuData_.num_bonds = num_bonds;
	}	

    private:
        GpuData gpuData_;
        CpuData cpuData_;
};

void Simulation::run(System system, int num_steps) {

    // H to D memory transfer (position)
    cudaMemcpy(
            thrust::raw_pointer_cast(gpuData_.posq.data()),
            thrust::raw_pointer_cast(cpuData_.posq.data()),
            cpuData_.num_particles*sizeof(float4),
            cudaMemcpyDefault);

    // H to D memory transfer (bond infomation)
    cudaMemcpy(
	    thrust::raw_pointer_cast(gpuData_.r0.data()),
	    thrust::raw_pointer_cast(system.get_bondList().r0.data()), 
	    sizeof(float)*system.get_bondList().r0.size(),
	    cudaMemcpyDefault);

    cudaMemcpy(
	    thrust::raw_pointer_cast(gpuData_.kb.data()), 
	    thrust::raw_pointer_cast(system.get_bondList().kb.data()), 
	    sizeof(float)*system.get_bondList().kb.size(),
	    cudaMemcpyDefault);

    cudaMemcpy(
	    thrust::raw_pointer_cast(gpuData_.pair.data()),
	    thrust::raw_pointer_cast(system.get_bondList().pair.data()), 
	    sizeof(int2)*system.get_bondList().pair.size(), 
	    cudaMemcpyDefault);

    set_num_bonds(system.get_bondList().r0.size());


    for (int i = 0; i < num_steps; i++) {
        // empty process
    }
}

void Simulation::set_positions(float x, float y, float z, float q) {
    float4 posq = {x, y, z, q};
    cpuData_.posq.push_back(posq);
    cpuData_.num_particles++;
}

#endif // __SIMULATION_CUH__
