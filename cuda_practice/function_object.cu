#include <cstdio>
#include <thrust/device_vector.h>

// abstruct kernel function
template <typename Function, typename T, typename... Args>
__global__ void AbstructKernel(Function func, T* d_result, Args... args)
{
	unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;
	d_result[index] += func(args...);
}



// user-defined potential function
template <typename T>
struct HarmonicBondPotential {
	__forceinline__ __device__ T operator()(T a, T b, T c) const {
		T r = a - b;
		return c*r*r;
	}
};

template <typename T>
struct DebyeHuckelPotential {
	__forceinline__ __device__ T operator()(T a, T b, T c, T d) const {
		T r = a - b;
		T r2 = r*r;
		T inv_r = rsqrtf(r2);
		return c*expf(-d*r2*inv_r)*inv_r;
	}
};


// user-defined C++ kernel wrapper
void calculate_energy(
		thrust::device_vector<float> &energy)
{
	float * d_energy = thrust::raw_pointer_cast(energy.data());

	AbstructKernel<<<32, 512>>>(HarmonicBondPotential<float>(), d_energy, 10.0, 20.0, 1.0);
	AbstructKernel<<<32, 512>>>(DebyeHuckelPotential<float>(), d_energy, 10.0, 20.0, 1.0, 1.0);
}

// user defined main function
int main(void) {
	thrust::device_vector<float> energy(16384);

	for (int i = 0; i < 100000; i++) {
		calculate_energy(energy);
	}

	cudaDeviceSynchronize();
}
