#pragma once
#include <string>
#include <vector>
#include <H5Cpp.h>

class IOUtils {
public:
    class HDF5Writer {
    public:
        HDF5Writer(const std::string& filename, int nx, int ny, int num_species,
                  float dx = 1.0f, float dy = 1.0f);
        ~HDF5Writer();
        void writeTimestep(const std::vector<float>& concentrations, float time);
        void createXDMF(const std::string& xdmf_filename);
    
    private:
        H5::H5File file_;
        H5::DataSpace dataspace_;
        int nx_, ny_, num_species_;
        float dx_, dy_;
        int timestep_;
        std::vector<float> times_;
        std::string filename_;
    };
};