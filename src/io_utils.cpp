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
        for (int s = 0; s < num_species_; ++s) {
            // Create dataset name for each species
            std::stringstream ss;
            ss << "concentrations_" << std::setw(6) << std::setfill('0') << timestep_ 
               << "_species_" << s;
            
            // Create 3D dataspace for cell-centered data
            hsize_t dims[3] = {static_cast<hsize_t>(ny_),
                              static_cast<hsize_t>(nx_),
                              static_cast<hsize_t>(1)};  // Added third dimension
            H5::DataSpace dataspace(3, dims);
            
            // Create and write dataset
            H5::DataSet dataset = file_.createDataSet(ss.str(),
                                                    H5::PredType::NATIVE_FLOAT,
                                                    dataspace);
            
            // Create temporary buffer for 3D data
            std::vector<float> data_3d(nx_ * ny_);
            
            // Extract data for this species
            for (int i = 0; i < nx_ * ny_; ++i) {
                data_3d[i] = concentrations[i * num_species_ + s];
            }
            
            dataset.write(data_3d.data(), H5::PredType::NATIVE_FLOAT);
            
            // Add time attribute
            H5::DataSpace attr_space(H5S_SCALAR);
            H5::Attribute attr = dataset.createAttribute("time",
                                                       H5::PredType::NATIVE_FLOAT,
                                                       attr_space);
            attr.write(H5::PredType::NATIVE_FLOAT, &time);
        }
        
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
             << "        <Topology TopologyType=\"3DRectMesh\" "
             << "Dimensions=\"" << (ny_ + 1) << " " << (nx_ + 1) << " 2\"/>\n"
             << "        <Geometry GeometryType=\"VXVYVZ\">\n"
             // X coordinates
             << "          <DataItem Name=\"X\" Dimensions=\"" << (nx_ + 1) << "\" NumberType=\"Float\" Format=\"XML\">\n            ";
        
        // Write X coordinates
        for (int x = 0; x <= nx_; x++) {
            xdmf << x * dx_ << " ";
        }
        
        // Y coordinates
        xdmf << "\n          </DataItem>\n"
             << "          <DataItem Name=\"Y\" Dimensions=\"" << (ny_ + 1) << "\" NumberType=\"Float\" Format=\"XML\">\n            ";
        
        // Write Y coordinates
        for (int y = 0; y <= ny_; y++) {
            xdmf << y * dy_ << " ";
        }
        
        // Z coordinates (just 0 and dz for single layer)
        xdmf << "\n          </DataItem>\n"
             << "          <DataItem Name=\"Z\" Dimensions=\"2\" NumberType=\"Float\" Format=\"XML\">\n"
             << "            0.0 1.0\n"
             << "          </DataItem>\n"
             << "        </Geometry>\n";
        
        // Write data for each species
        for (int s = 0; s < num_species_; ++s) {
            xdmf << "        <Attribute Name=\"Species_" << s 
                 << "\" AttributeType=\"Scalar\" Center=\"Cell\">\n"
                 << "          <DataItem Dimensions=\"" << ny_ << " " << nx_ << " 1"
                 << "\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF5\">\n"
                 << "            " << filename_ << ":/concentrations_"
                 << std::setw(6) << std::setfill('0') << i << "_species_" << s << "\n"
                 << "          </DataItem>\n"
                 << "        </Attribute>\n";
        }
        
        xdmf << "      </Grid>\n";
    }
    
    xdmf << "    </Grid>\n"
         << "  </Domain>\n"
         << "</Xdmf>\n";
}