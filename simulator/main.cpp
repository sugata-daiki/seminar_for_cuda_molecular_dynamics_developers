#include "System.h"

int main(void) {

    System system;

    int N_particles = 10000;

    for (int i = 0; i < N_particles; i++) {
        system.add_particle("DNA", i);
    }

}
