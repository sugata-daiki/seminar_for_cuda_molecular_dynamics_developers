#ifndef __SIMULATION_CUH__
#define __SIMULATION_CUH__

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <map>
#include <stdexcept>
#include <algorithm>

#include "System.h"
#include "Observer.cuh"
#include "Containers.cuh"
#include "cudaSystem.cuh"


class Simulation {
    public:
        Simulation() {};
        ~Simulation() {};

        void run(System system, Observer & observer, int num_steps);

        void set_seed(int seed) {
            cpuData_.seed = seed;
        }

        void set_positions(float x, float y, float z, float q);

        int get_num_particles() {return cpuData_.num_particles;}

        void read_forcefields(std::string & input_filename);    

    private:
        void set_num_bonds(int num_bonds) {
            cpuData_.num_bonds = num_bonds;
        }

        void load_positions();

        void load_diffusion_coefficients(const std::vector<float> & diffusion_coefficients);

        void load_bond_params(System system);

        void load_constant_params(System system);

        void set_random_states();

        GpuData gpuData_;
        CpuData cpuData_;
};

void Simulation::run(System system, Observer & observer, int num_steps) {

    load_positions();

    load_bond_params(system);

    set_num_bonds(system.get_bondList().r0.size());

    load_diffusion_coefficients(system.get_diffusion_coefficients());

    load_constant_params(system);

    set_random_states();

    cudaEvent_t start, stop;

    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    
    std::cout << "simulation start!" << std::endl;
    // run simulation
    for (int i = 0; i < num_steps; i++) {
        if (i % observer.get_output_period() == 0) {
            observer.CoordObserver(gpuData_, cpuData_);
        }
        // bonded interaction
        calculateBondedForcesH(gpuData_, cpuData_);
        BrownianIntegratorH(gpuData_, cpuData_);

    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);


    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "execution time: " << 1e-3*elapsedTime << " [sec]" << std::endl;


}

void Simulation::set_positions(float x, float y, float z, float q) {
    float4 posq = {x, y, z, q};
    cpuData_.posq.push_back(posq);
    cpuData_.num_particles++;
}

void Simulation::set_random_states() {
    gpuData_.states.resize(cpuData_.num_particles);

    init_random_statesH(gpuData_, cpuData_);
}

void Simulation::load_diffusion_coefficients(const std::vector<float> & diffusion_coefficients) {

    gpuData_.diffusion_coefficient.resize(diffusion_coefficients.size());
    thrust::copy(
        diffusion_coefficients.begin(),
        diffusion_coefficients.end(), 
        gpuData_.diffusion_coefficient.begin());
}

void Simulation::load_positions() {
    gpuData_.posq.resize(cpuData_.posq.size());
    gpuData_.force.resize(cpuData_.posq.size());
    thrust::copy(
        cpuData_.posq.begin(),
        cpuData_.posq.end(), 
        gpuData_.posq.begin());
}

void Simulation::load_bond_params(System system) {
    gpuData_.r0.resize(system.get_bondList().r0.size());
    gpuData_.kb.resize(system.get_bondList().kb.size());
    gpuData_.pair.resize(system.get_bondList().pair.size());

    bond_SOA b = system.get_bondList();

    thrust::copy(
        b.r0.begin(), 
        b.r0.end(), 
        gpuData_.r0.begin());

    thrust::copy(
        b.kb.begin(), 
        b.kb.end(), 
        gpuData_.kb.begin());

    thrust::copy(
        b.pair.begin(), 
        b.pair.end(), 
        gpuData_.pair.begin());
}

void Simulation::load_constant_params(System system) {
    SimParams p = system.get_params();

    if (p.timestep == 0.0) {
        std::cerr << "[ error ]: Please set timestep." << std::endl;
    }

    cudaMemcpyToSymbol(params_, &p, sizeof(SimParams));

}

void Simulation::read_forcefields(std::string & input_filename) {

    std::ifstream ifs(input_filename);
    if (!ifs) {
        std::cerr << "[ error ]: Could not open file." << std::endl;
        exit(-1);
    }

    std::string line;
    std::vector<std::string> BondedForceFieldNameList;

    /*
     * input file example
     * 
     * # bonded force field setup
     * [bondedFF] HarmonicBond 
     * 
    */

    int line_num = 0;
    while (std::getline(ifs, line)) {
        line_num++;

        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::stringstream ss(line);
        std::string ff;
        ss >> ff;
        if (ff == "[bondedFF]") {
            if (BondedForceFieldNameList.size() != 0) {
                std::cerr << "[ error ]: Definition conflict detected: [bondedFF]" << std::endl;
                exit(-1);
            }
            std::string bonded_ff;
            while (ss >> bonded_ff) {
                BondedForceFieldNameList.push_back(bonded_ff);
            }
        }
    }

    // output enabled force fields
    std::cout << "[ bonded force field ]: ";

    for (int i = 0; i < BondedForceFieldNameList.size(); i++) {
        std::cout << BondedForceFieldNameList[i] << " ";
    }
    std::cout << std::endl;

    cpuData_.EnableBondedForceFields_.EnableHarmonicBond = (std::find(
               BondedForceFieldNameList.begin(), 
               BondedForceFieldNameList.end(), 
               "HarmonicBond"
               )
            != BondedForceFieldNameList.end());

    cpuData_.EnableBondedForceFields_.EnableCosineAngle = (std::find(
               BondedForceFieldNameList.begin(), 
               BondedForceFieldNameList.end(), 
               "CosineAngle"
               )
            != BondedForceFieldNameList.end());

    cpuData_.EnableBondedForceFields_.EnableDihedral = (std::find(
                BondedForceFieldNameList.begin(), 
                BondedForceFieldNameList.end(), 
                "Dihedral"
                )
            != BondedForceFieldNameList.end());
}


#endif // __SIMULATION_CUH__
