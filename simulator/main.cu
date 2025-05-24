#include <iostream>
#include <vector>
#include <string>

#include "Simulation.cuh"

int main(int argc, char* argv[]) {

    if (argc < 2) {
        std::cerr << "[ error ]: No input file specified." << std::endl;
        exit(-1);
    }

    std::string ff_input = argv[1];

    System system;
    Simulation simulation;
    std::string output_file = "./output/output.dat";
    int output_period = 100;

    Observer observer(output_file, output_period);

    simulation.read_forcefields(ff_input);

    int N_particles = 10000;
    int num_steps = 10000;
    float temperature = 300.0; // K
    float timestep = 0.001; // us
    int seed = 12345;
    float diffusion_coefficient = 10.0; // [(um**2)/s)]
    float r0 = 10.0;    
    float kbond = 2.5;

    simulation.set_seed(seed);
    system.set_params(temperature, timestep);

    for (int i = 0; i < N_particles; i++) {
        system.add_particle("DNA", diffusion_coefficient, i);
        simulation.set_positions(i*9.0f, 0.0f, 0.0f, 0.0f);
    }

    int numBonds = N_particles - 1;

    for (int i = 0; i < numBonds; i++) {
        system.add_bond(r0, kbond, i, i+1);
    }

    simulation.run(system, observer, num_steps);
    
    cudaDeviceSynchronize();

}
