#include <iostream>
#include <vector>
#include <string>

#include "Simulation.cuh"

int main(int argc, char* argv[]) {

    System system;
    Simulation simulation;
    
    if (argc < 2) {
        std::cerr << "[ error ]: No input file specified." << std::endl;
        exit(-1);
    }

    std::string ff_input = argv[1];

    simulation.read_forcefields(ff_input);

    int N_particles = 10000;
    int num_steps = 10000;
    float temperature = 300.0;
    float timestep = 0.01;
    int seed = 12345;
    float diffusion_coefficient = 10.0;

    simulation.set_timestep(timestep);
    simulation.set_seed(seed);
    system.set_params(temperature);

    for (int i = 0; i < N_particles; i++) {
        system.add_particle("DNA", diffusion_coefficient, i);
        simulation.set_positions(i*0.9f, 0.0f, 0.0f, 0.0f);
    }

    int numBonds = N_particles - 1;

    for (int i = 0; i < numBonds; i++) {
        system.add_bond(1.0, 1.0, i, i+1);
    }

    simulation.run(system, num_steps);
    
    cudaDeviceSynchronize();

}
