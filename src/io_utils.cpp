#include "io_utils.hpp"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>  
#include <stdexcept>

IOUtils::HDF5Writer::HDF5Writer(
    const std::string& filename, int nx, int ny, int num_species,
    float dx, float dy)
    : nx_(nx), ny_(ny), num_species_(num_species), dx_(dx), dy_(dy),
      timestep_(0), filename_(filename) {
    
    try {
        // Create HDF5 file
        file_ = H5::H5File(filename, H5F_ACC_TRUNC);
        
        // Create dataspace for concentration field
        hsize_t dims[3] = {static_cast<hsize_t>(ny_),
                          static_cast<hsize_t>(nx_),
                          static_cast<hsize_t>(num_species_)};
        dataspace_ = H5::DataSpace(3, dims);
        
    } catch (const H5::Exception& e) {
        throw std::runtime_error("Failed to create HDF5 file: " + 
                               std::string(e.getCDetailMsg()));
    }
}

IOUtils::HDF5Writer::~HDF5Writer() {
    try {
        file_.close();
    } catch (const H5::Exception& e) {
        // Just log error on destruction
        std::cerr << "Error closing HDF5 file: " << e.getCDetailMsg() << std::endl;
    }
}

void IOUtils::HDF5Writer::writeTimestep(
    const std::vector<float>& concentrations, float time) {
    
    try {
        // Create dataset name
        std::stringstream ss;
        ss << "concentrations_" << std::setw(6) << std::setfill('0') << timestep_;
        
        // Create and write dataset
        H5::DataSet dataset = file_.createDataSet(ss.str(),
                                                H5::PredType::NATIVE_FLOAT,
                                                dataspace_);
        dataset.write(concentrations.data(), H5::PredType::NATIVE_FLOAT);
        
        // Add time attribute
        H5::DataSpace attr_space(H5S_SCALAR);
        H5::Attribute attr = dataset.createAttribute("time",
                                                   H5::PredType::NATIVE_FLOAT,
                                                   attr_space);
        attr.write(H5::PredType::NATIVE_FLOAT, &time);
        
        times_.push_back(time);
        timestep_++;
        
    } catch (const H5::Exception& e) {
        throw std::runtime_error("Failed to write timestep: " + 
                               std::string(e.getCDetailMsg()));
    }
}

void IOUtils::HDF5Writer::createXDMF(const std::string& xdmf_filename) {
    std::ofstream xdmf(xdmf_filename);
    if (!xdmf.is_open()) {
        throw std::runtime_error("Failed to create XDMF file");
    }
    
    xdmf << "<?xml version=\"1.0\" ?>\n"
         << "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n"
         << "<Xdmf Version=\"3.0\">\n"
         << "  <Domain>\n"
         << "    <Grid GridType=\"Collection\" CollectionType=\"Temporal\">\n";
    
    for (int i = 0; i < timestep_; ++i) {
        xdmf << "      <Grid Name=\"TimeSeries_" << i << "\" GridType=\"Uniform\">\n"
             << "        <Time Value=\"" << times_[i] << "\"/>\n"
             << "        <Topology TopologyType=\"2DCoRectMesh\" "
             << "Dimensions=\"" << ny_ << " " << nx_ << "\"/>\n"
             << "        <Geometry GeometryType=\"ORIGIN_DXDY\">\n"
             << "          <DataItem Dimensions=\"2\" Format=\"XML\">\n"
             << "            0.0 0.0\n"
             << "          </DataItem>\n"
             << "          <DataItem Dimensions=\"2\" Format=\"XML\">\n"
             << "            " << dx_ << " " << dy_ << "\n"
             << "          </DataItem>\n"
             << "        </Geometry>\n";
        
        // Write data for each species
        for (int s = 0; s < num_species_; ++s) {
            xdmf << "        <Attribute Name=\"Species_" << s 
                 << "\" AttributeType=\"Scalar\" Center=\"Node\">\n"
                 << "          <DataItem Dimensions=\"" << ny_ << " " << nx_ 
                 << "\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">\n"
                 << "            " << filename_ << ":/concentrations_"
                 << std::setw(6) << std::setfill('0') << i << "\n"
                 << "          </DataItem>\n"
                 << "        </Attribute>\n";
        }
        
        xdmf << "      </Grid>\n";
    }
    
    xdmf << "    </Grid>\n"
         << "  </Domain>\n"
         << "</Xdmf>\n";
}