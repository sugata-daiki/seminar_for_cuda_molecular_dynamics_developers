#ifndef __PARTICLEDATAMANAGER_H__
#define __PARTICLEDATAMANAGER_H__

#include <vector>
#include <string>
#include "ParameterManager.cuh"

struct Particle {
    int id;
    int particleType_id;
};

class ParticleDataManager {
    public:

        const std::vector<int> & get_particle_ids() const { return particle_ids_;}
        const std::vector<int> & get_particle_type_ids() const { return particle_type_ids_;}

        void set_particle_id(int id) {
            particle_ids_.push_back(id);
        }

        void set_particle_type_id(int particle_type_id) {
            particle_type_ids_.push_back(particle_type_id);
        }

        void set_bond(bond b) {
            bonds_.push_back(b);
        }

    private:
        std::vector<int> particle_ids_;
        std::vector<int> particle_type_ids_;
        std::vector<bond> bonds_;
};

#endif // __PARTICLEDATAMANAGER_H__
