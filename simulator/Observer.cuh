#ifndef __OBSERVER_H__
#define __OBSERVER_H__

#include <string>
#include <fstream>
#include <ostream>
#include <iomanip>
#include <algorithm>
#include <iterator>
#include <iostream>

#include "Containers.cuh"

class Observer
{
    public:
        Observer(std::string outputFile, int output_period);
        ~Observer() {};

        void CoordObserver(GpuData & gpu_data, CpuData & cpu_data);

        int get_output_period() {return output_period_;}

    private:
        std::string outputFile_;
        int output_period_;
};


Observer::Observer(std::string outputFile, int output_period) : 
    outputFile_(outputFile),
    output_period_(output_period) {
}


void Observer::CoordObserver(GpuData & gpu_data, CpuData & cpu_data) {

    std::ofstream ofs(outputFile_, std::ios::app);
    
    cpu_data.posq.resize(gpu_data.posq.size());

    cudaMemcpy(
            thrust::raw_pointer_cast(cpu_data.posq.data()), 
            thrust::raw_pointer_cast(gpu_data.posq.data()), 
            cpu_data.posq.size()*sizeof(float4), 
            cudaMemcpyDefault);

    std::vector<float> tmp_x(cpu_data.posq.size());
    std::vector<float> tmp_y(cpu_data.posq.size());
    std::vector<float> tmp_z(cpu_data.posq.size());

    float4 tmp_posq = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    for (int i = 0; i < cpu_data.posq.size(); i++) {
        tmp_posq = cpu_data.posq[i];
        tmp_x[i] = tmp_posq.x;
        tmp_y[i] = tmp_posq.y;
        tmp_z[i] = tmp_posq.z;
    }

    std::copy(
            std::cbegin(tmp_x), 
            std::cend(tmp_x), 
            std::ostream_iterator<const decltype(tmp_x)::value_type&>(ofs, " "));

    std::copy(
            std::cbegin(tmp_y), 
            std::cend(tmp_y), 
            std::ostream_iterator<const decltype(tmp_y)::value_type&>(ofs, " "));

    std::copy(
            std::cbegin(tmp_z), 
            std::cend(tmp_z), 
            std::ostream_iterator<const decltype(tmp_y)::value_type&>(ofs, " "));

    ofs << std::endl;
    ofs.close();
}
#endif
