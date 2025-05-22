#include <iostream>
#include <functional>

template<typename FunctionObject, typename... Args>
auto FunctionObjectUser(FunctionObject&& func, Args&&... args) -> decltype(std::forward<FunctionObject>(func)(std::forward<Args>(args)...)) {
	return std::forward<FunctionObject>(func)(std::forward<Args>(args)...);
}

void Use_FunctionObjectUser()
{
	int i;

	i = FunctionObjectUser( std::plus<int>(), 100, 200);

	std::cout << i << std::endl;

}

int main(void) {
	Use_FunctionObjectUser();

	return 0;
}
