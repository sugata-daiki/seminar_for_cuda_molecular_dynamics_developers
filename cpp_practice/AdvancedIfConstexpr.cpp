#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <map>
#include <stdexcept>


enum class Operation {
	Add,
	Subtract,
	Multiply,
	Divide,
        Unknown
};

std::map<std::string, Operation> op_map = {
	{"Add", Operation::Add}, 
	{"Subtract", Operation::Subtract}, 
	{"Multiply", Operation::Multiply}, 
	{"Divide", Operation::Divide}
};

Operation string_to_operation(const std::string & op_str) {
	auto it = op_map.find(op_str);
	if (it != op_map.end()) {
		return it->second;
	}
	return Operation::Unknown;
}

void perform_operation(Operation Op, auto a, auto b) {
	if (Op == Operation::Add) {
		std::cout << "Addition: " << a << " + " << b << " = " << (a + b) << std::endl;
	}

	else if (Op == Operation::Subtract) {
		std::cout << "Subtraction: " << a << " - " << b << " = " << (a - b) << std::endl;
	}

	else if (Op == Operation::Multiply) {
		std::cout << "Multiplication: " << a << " * " << b << " = " << a*b << std::endl;
	}

	else if (Op == Operation::Divide) {
		if (b != 0) {
			std::cout << "Division: " << a << " / " << b << " = " << a/b << std::endl;
		}
		else {
			std::cout << "[ error ] Cannot divide by zero." << std::endl;
		}
	}
	else {
		std::cout << "[ warning ] Unknown operation." << std::endl;
	}
}


int main(int argc, char* argv[]) {

	if (argc < 2) {
		std::cerr << "[ error ]: No input file specified." << std::endl;
		return 1;
	}

	std::string input_file = argv[1];

	std::ifstream ifs(input_file);
	
	if (!ifs) {
		std::cerr << "[ error ]: Could not open file." << std::endl;
		return 1;
	}

	std::string line;
	int line_num = 0;
	while (std::getline(ifs, line)) {
		line_num++;

		if (line.empty() || line[0] == '#') {
			continue;
		}

		std::stringstream ss(line);
		std::string op_type_str;
		int operand1, operand2;

		if (!(ss >> op_type_str >> operand1 >> operand2)) {
			std::cerr << "[ warning ]: Invalid line format at line " << line_num << ": '" << line << "'" << std::endl;
			continue;
		}

		Operation current_op = string_to_operation(op_type_str);
		perform_operation(current_op, operand1, operand2);
	}

	ifs.close();

	return 0;
}
