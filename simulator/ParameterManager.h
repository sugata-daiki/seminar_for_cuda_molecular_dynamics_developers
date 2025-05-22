#include <string>
#include <vector>
#include <map>

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


    private:
        std::map<std::string, int> type_name_to_id_;
        std::vector<std::string> type_id_to_name_;
        int next_type_id_;
};
