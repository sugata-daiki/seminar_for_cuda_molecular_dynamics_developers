#include <iostream>
#include <string>


enum class Operation {
	Add,
	Subtract,
	Multiply,
	Divide,
        Unknown
};

template <Operation Op>
void perform_operation(auto a, auto b) {
	if constexpr (Op == Operation::Add) {
		std::cout << "Addition: " << a << " + " << b << " = " << (a + b) << std::endl;
	}

	else if constexpr (Op == Operation::Subtract) {
		std::cout << "Subtraction: " << a << " - " << b << " = " << (a - b) << std::endl;
	}

	else if constexpr (Op == Operation::Multiply) {
		std::cout << "Multiplication: " << a << " * " << b << " = " << a*b << std::endl;
	}

	else if constexpr (Op == Operation::Divide) {
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


int main(void) {
	perform_operation<Operation::Add>(10, 5);
	return 0;
}
