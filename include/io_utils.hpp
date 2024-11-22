#pragma once

#include <string>
#include <vector>
#include <H5Cpp.h>
#include <stdexcept>

namespace IOUtils {

class HDF5Writer {
public:
    // Only declare the constructor here, don't define it
    HDF5Writer(const std::string& filename, 
               int nx, int ny, 
               int num_species,
               float dx, float dy);
    
    ~HDF5Writer();
    
    void writeTimestep(const std::vector<float>& concentrations, float time);
    void createXDMF(const std::string& xdmf_filename);
    
private:
    std::string filename_;
    int nx_, ny_, num_species_;
    float dx_, dy_;
    int timestep_;
    std::vector<float> times_;
    H5::H5File file_;
    H5::DataSpace dataspace_;
};

} // namespace IOUtils