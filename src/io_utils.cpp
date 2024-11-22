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
        
        // For FVM, dimensions are based on number of cells
        hsize_t dims[3] = {static_cast<hsize_t>(ny_),    // Number of cells in y
                          static_cast<hsize_t>(nx_),      // Number of cells in x
                          static_cast<hsize_t>(num_species_)};
        dataspace_ = H5::DataSpace(3, dims);
        
        // Add mesh attributes
        H5::Group mesh_info = file_.createGroup("/mesh");
        H5::DataSpace scalar_space(H5S_SCALAR);
        
        // Add cell size information
        H5::Attribute dx_attr = mesh_info.createAttribute("dx", 
            H5::PredType::NATIVE_FLOAT, scalar_space);
        dx_attr.write(H5::PredType::NATIVE_FLOAT, &dx_);
        
        H5::Attribute dy_attr = mesh_info.createAttribute("dy", 
            H5::PredType::NATIVE_FLOAT, scalar_space);
        dy_attr.write(H5::PredType::NATIVE_FLOAT, &dy_);
        
        // Add mesh type attribute
        H5::StrType str_type(H5::PredType::C_S1, 256);
        H5::Attribute type_attr = mesh_info.createAttribute("type", 
            str_type, scalar_space);
        const char* mesh_type = "finite_volume";
        type_attr.write(str_type, mesh_type);
        
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

void IOUtils::HDF5Writer::writeTimestep(const std::vector<float>& concentrations, float time) {
    try {
        std::stringstream ss;
        ss << "concentrations_" << std::setw(6) << std::setfill('0') << timestep_;
        
        // For FVM, we're storing cell-averaged values
        H5::DataSet dataset = file_.createDataSet(ss.str(),
                                                H5::PredType::NATIVE_FLOAT,
                                                dataspace_);
        dataset.write(concentrations.data(), H5::PredType::NATIVE_FLOAT);
        
        // Add time and mesh type attributes
        H5::DataSpace attr_space(H5S_SCALAR);
        H5::Attribute time_attr = dataset.createAttribute("time",
                                                        H5::PredType::NATIVE_FLOAT,
                                                        attr_space);
        time_attr.write(H5::PredType::NATIVE_FLOAT, &time);
        
        // Add attribute to indicate cell-centered data
        H5::StrType str_type(H5::PredType::C_S1, 256);
        H5::Attribute center_attr = dataset.createAttribute("centering",
                                                          str_type,
                                                          attr_space);
        const char* center_val = "cell";
        center_attr.write(str_type, center_val);
        
        times_.push_back(time);
        timestep_++;
    } catch (const H5::Exception& e) {
        throw std::runtime_error("Failed to write timestep: " + 
                               std::string(e.getCDetailMsg()));
    }
}

void IOUtils::HDF5Writer::createXDMF(const std::string& xdmf_filename) {
    std::ofstream xdmf(xdmf_filename);
    
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
             << "          <DataItem Dimensions=\"2\" Format=\"XML\">0.0 0.0</DataItem>\n"
             << "          <DataItem Dimensions=\"2\" Format=\"XML\">" << dx_ << " " << dy_ << "</DataItem>\n"
             << "        </Geometry>\n";

        for (int s = 0; s < num_species_; ++s) {
            xdmf << "        <Attribute Name=\"Species_" << s 
                 << "\" AttributeType=\"Scalar\" Center=\"Node\">\n"
                 << "          <DataItem Dimensions=\"" << ny_ << " " << nx_ 
                 << "\" NumberType=\"Float\" Format=\"HDF\">\n"
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