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
} CpuData;

class Simulation {
    public:
        Simulation() {};
        ~Simulation() {};

        void run(int num_steps);

        void set_positions(float x, float y, float z, float q);

        int get_num_particles() {return cpuData_.num_particles;}

    private:
        GpuData gpuData_;
        CpuData cpuData_;
};

void Simulation::run(System system, int num_steps) {

    cudaMemcpy(
            thrust::raw_pointer_cast(gpuData_.posq.data()),
            thrust::raw_pointer_cast(cpuData_.posq.data()),
            cpuData_.num_particles*sizeof(float4),
            cudaMemcpyDefault);

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
