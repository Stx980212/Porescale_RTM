#pragma once

#include <string>
#include <vector>
#include <H5Cpp.h>
#include <stdexcept>

namespace IOUtils {

class HDF5Writer {
public:
    HDF5Writer(const std::string& filename, 
               int nx, int ny, 
               int num_species,
               float dx, float dy)
        : filename_(filename)
        , nx_(nx), ny_(ny)
        , num_species_(num_species)
        , dx_(dx), dy_(dy)
        , timestep_(0) {
        
        file_ = H5::H5File(filename_, H5F_ACC_TRUNC);
        
        hsize_t dims[4] = {
            static_cast<hsize_t>(nx_),
            static_cast<hsize_t>(ny_),
            static_cast<hsize_t>(1),
            static_cast<hsize_t>(num_species_)
        };
        dataspace_ = H5::DataSpace(4, dims);
    }
    
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