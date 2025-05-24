#ifndef __PARAMETERMANAGER_CUH__
#define __PARAMETERMANAGER_CUH__

#include <string>
#include <vector>
#include <map>

#include <thrust/host_vector.h>

#include "Constants.cuh"

typedef struct {
    float r0;
    float kb;
    int id_i;
    int id_j;
} bond;

typedef struct {
    thrust::host_vector<float> r0;
    thrust::host_vector<float> kb;
    thrust::host_vector<int2> pair;
} bond_SOA;

class ParameterManager {
    public:
        ParameterManager() : next_type_id_(0) {}

    int get_or_assign_particle_type_id(const std::string& type_name) {
        auto it = type_name_to_id_.find(type_name);
        if (it != type_name_to_id_.end()) {
            return it->second; // return existing id
        }
        else {
            // assign new id to particle type name
            int new_id = next_type_id_++;
            type_name_to_id_[type_name] = new_id;
            type_id_to_name_.push_back(type_name);

            return new_id;
        }
    }

    void store_bondInfo(bond b) {
        bondList.r0.push_back(b.r0);
        bondList.kb.push_back(b.kb);
        bondList.pair.push_back({b.id_i, b.id_j});
    }

    void store_diffusionInfo(float diffusion_coefficient) {
        diffusion_coefficients_.push_back(diffusion_coefficient);
    }

    bond_SOA get_bondList() const {
	    return bondList;
    }

    std::vector<float> get_diffusion_coefficients() {
        return diffusion_coefficients_;
    }

    void set_params(float temperature) {
        simParams_.inv_kbT = 1.0/(1.0e-3*k_b*temperature*NA/4.184);
    }

    SimParams get_params() {
        return simParams_;
    }

    private:
        std::map<std::string, int> type_name_to_id_;
        std::vector<std::string> type_id_to_name_;
        int next_type_id_;
        bond_SOA bondList;
        std::vector<float> diffusion_coefficients_;

        SimParams simParams_;
};

#endif // __PARAMETERMANAGER_CUH__
