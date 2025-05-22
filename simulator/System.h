#include <vector>
#include <memory>

#include "ParticleDataManager.h"
#include "ParameterManager.h"

class System {
    public:
        System();
        ~System();

        void add_particle(const std::string& particle_type, int id) {

            auto particle_type_id = parameterManager_.get_or_assign_particle_type_id(particle_type);

            particleDataManager_.set_particle_id(id);
            particleDataManager_.set_particle_type_id(particle_type_id);

        }


    private:
        ParticleDataManager particleDataManager_;
        ParameterManager parameterManager_;
};


System::System() {
}

System::~System() {
}
