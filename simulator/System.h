#ifndef __SYSTEM_H__
#define __SYSTEM_H__

#include <vector>
#include <memory>

#include "ParticleDataManager.h"

class System : public ParameterManager{
    public:
        System();
        ~System();

        void add_particle(const std::string& particle_type, float diffusion_coefficient, int id) {

            auto particle_type_id = get_or_assign_particle_type_id(particle_type);

            particleDataManager_.set_particle_id(id);
            particleDataManager_.set_particle_type_id(particle_type_id);
            particleDataManager_.set_diffusion_coefficient(diffusion_coefficient);

        }

        void add_bond(float r0, float kb, int id_i, int id_j);
        void set_temperature(float temperature) {
            temperature_ = temperature;
        }


    private:
        ParticleDataManager particleDataManager_;
        float temperature_;
};


System::System() {
}

System::~System() {
}

void System::add_bond(float r0, float kb, int id_i, int id_j) {
    bond b = {r0, kb, id_i, id_j};
    
    store_bondInfo(b);
    particleDataManager_.set_bond(b);
}

#endif // __SYSTEM_H__
