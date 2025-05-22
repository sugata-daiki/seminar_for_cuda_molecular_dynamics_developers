#include <iostream>
#include "Simulation.cuh"

int main(void) {

    System system;
    Simulation simulation;

    int N_particles = 10000;

    for (int i = 0; i < N_particles; i++) {
        system.add_particle("DNA", i);
        simulation.set_positions(i*0.9f, 0.0f, 0.0f, 0.0f);
    }

    int numBonds = N_particles - 1;

    for (int i = 0; i < numBonds; i++) {
        system.add_bond(1.0, 1.0, i, i+1);
    }

    std::cout << simulation.get_num_particles() << std::endl;
    simulation.run(system, 1000);

}
