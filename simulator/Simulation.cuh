#ifndef __SIMULATION_CUH__
#define __SIMULATION_CUH__

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <map>
#include <stdexcept>
#include <algorithm>

#include "System.h"
#include "Containers.cuh"
#include "cudaSystem.cuh"


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

        void read_forcefields(std::string & input_filename);    

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

    // run simulation
    for (int i = 0; i < num_steps; i++) {
        // bonded interaction
        calculateBondedForcesH(gpuData_, cpuData_);
    }
}

void Simulation::set_positions(float x, float y, float z, float q) {
    float4 posq = {x, y, z, q};
    cpuData_.posq.push_back(posq);
    cpuData_.num_particles++;
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
