#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <map>
#include <stdexcept>
#include <algorithm>

enum class BondedForceField {
	HarmonicBond, 
	CosineAngle,
	Dihedral, 
	Unknown
};

enum class NonbondedForceField {
	LennardJones, 
	DebyeHuckel, 
	Unknown
};

std::map<std::string, BondedForceField> BondedForceField_map = {
	{"HarmonicBond", BondedForceField::HarmonicBond}, 
	{"CosineAngle", BondedForceField::CosineAngle}, 
	{"Dihedral", BondedForceField::Dihedral}
};

std::map<std::string, NonbondedForceField> NonbondedForceField_map = {
	{"LennardJones", NonbondedForceField::LennardJones}, 
	{"DebyeHuckel", NonbondedForceField::DebyeHuckel}
};

BondedForceField string_to_BondedForceField(const std::string & bonded_str) {
	auto it = BondedForceField_map.find(bonded_str);
	if (it != BondedForceField_map.end()) {
		return it->second;
	}
	return BondedForceField::Unknown;
}

NonbondedForceField string_to_NonbondedForceField(const std::string & nonbonded_str) {
	auto it = NonbondedForceField_map.find(nonbonded_str);
	if (it != NonbondedForceField_map.end()) {
		return it->second;
	}
	return NonbondedForceField::Unknown;
}

template <bool EnableHarmonicBond_T, bool EnableCosineAngle_T, bool EnableDihedral_T>
void calculate_bonded_forces() {
	if constexpr(EnableHarmonicBond_T) {
		std::cout << "HarmonicBond calculated" << std::endl;
	}
	if constexpr(EnableCosineAngle_T) {
		std::cout << "CosineAngle calculated" << std::endl;
	}
	if constexpr(EnableDihedral_T) {
		std::cout << "Dihedral calculated" << std::endl;
	}
}

int main(int argc, char* argv[]) {
	if (argc < 2) {
		std::cerr << "[ error ]: No input file specified." << std::endl;
	}

	std::string input_file = argv[1];

	std::ifstream ifs(input_file);

	if (!ifs) {
		std::cerr << "[ error ]: Could not open file." << std::endl;
		return 1;
	}

	std::string line;
	std::vector<std::string> BondedForceFieldList;
	std::vector<std::string> NonbondedForceFieldList;
	/*
	 * input file example:
	 *
	 * # bonded force field setup
	 * [bondedFF] HarmonicBond CosineAngle
	 *
	 * # nonbonded force field setup
	 * [nonbondedFF] LennardJones DebyeHuckel
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
			if (BondedForceFieldList.size() != 0) {
				std::cerr << "[ error ] Definition conflict detected: [bondedFF]" << std::endl;
				return 1;
			}
			std::string bonded_ff;
			while (ss >> bonded_ff) {
				BondedForceFieldList.push_back(bonded_ff);
			}
		}
		if (ff == "[nonbondedFF]") {
			if (NonbondedForceFieldList.size() != 0) {
				std::cerr << "[ error ] Definition conflict detected: [nonbondedFF]" << std::endl;
				return 1;
			}
			std::string nonbonded_ff;
			while (ss >> nonbonded_ff) {
				NonbondedForceFieldList.push_back(nonbonded_ff);
			}
		}
	}

	// output enabled force fields
	std::cout << "[ bonded force field ]: ";

	for (int i = 0; i < BondedForceFieldList.size(); i++) {
		std::cout << BondedForceFieldList[i] << " ";
	}
	std::cout << std::endl;

	std::cout << "[ nonbonded force field ]: ";

	for (int i = 0; i < NonbondedForceFieldList.size(); i++) {
		std::cout << NonbondedForceFieldList[i] << " ";
	}
	std::cout << std::endl;
	
	// run dummy simulation
	constexpr bool EnableHarmonicBond = (std::find(
							BondedForceFieldList.begin(), 
							BondedForceFieldList.end(), 
							"HarmonicBond"
						  )
					!= BondedForceFieldList.end());

	constexpr bool EnableCosineAngle = (std::find(
							BondedForceFieldList.begin(), 
							BondedForceFieldList.end(), 
							"CosineAngle"
						)
					!= BondedForceFieldList.end());

	constexpr bool EnableDihedral = (std::find(
							BondedForceFieldList.begin(), 
							BondedForceFieldList.end(), 
							"Dihedral"
						)
					!= BondedForceFieldList.end());

	if (EnableHarmonicBond) {
		if (EnableCosineAngle) {
			if (EnableDihedral) {
				calculate_bonded_forces<true, true, true>();
			}
			else {
				calculate_bonded_forces<true, true, false>();
			}
		}
		else {
			if (EnableDihedral) {
				calculate_bonded_forces<true, false, true>();
			}
			else {
				calculate_bonded_forces<true, false, false>();
			}
		}
	}
	else {
	       	if (EnableCosineAngle) {
	       		if (EnableDihedral) {
		 		calculate_bonded_forces<false, true, true>();
			}
			else {
				calculate_bonded_forces<false, true, false>();
			}
		}	
		else {
			if (EnableDihedral) {
				calculate_bonded_forces<false, false, true>();
			}
			else {
				calculate_bonded_forces<false, false, false>();
			}
		}
	}


	return 0;
}
