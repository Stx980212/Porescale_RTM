#pragma once

#include <string>
#include <vector>
#include <H5Cpp.h>
#include <stdexcept>

namespace IOUtils {

class MaskReader {
public:
    struct MaskData {
        std::vector<int> active_cells;      // 1 for active cells, 0 for inactive
        std::vector<int> interface_cells;   // 1 for water cells at CO2 interface, 0 otherwise
        std::vector<int> clay_cells;
        std::vector<unsigned char> raw_labels; // Store original labels
    };
    
    static MaskData loadRawMask(const std::string& filename, 
                               int nx, int ny,
                               int clay_label = 0,
                               int water_label = 1,    // Label for water phase
                               int co2_label = 2);     // Label for sc-CO2 phase
};

class HDF5Writer {
public:
    HDF5Writer(const std::string& filename, 
               int nx, int ny, 
               int num_species,
               float dx, float dy);  // Just declare the constructor
    
    ~HDF5Writer();
    
    void writeTimestep(const std::vector<float>& concentrations, 
                      const std::vector<float>& cell_volumes,
                      const std::vector<float>& porosity,
                      float time);
    void createXDMF(const std::string& xdmf_filename);
    void writeMask(const std::vector<int>& mask, const std::string& name);
    
private:
    std::string filename_;
    int nx_, ny_, num_species_;
    float dx_, dy_;
    int timestep_;
    std::vector<float> times_;
    std::vector<float> cell_volumes_;  
    std::vector<float> porosity_;     
    H5::H5File file_;
    H5::DataSpace dataspace_;
};

} // namespace IOUtils