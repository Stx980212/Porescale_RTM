#include "io_utils.hpp"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>  
#include <stdexcept>
#include <algorithm>

IOUtils::HDF5Writer::HDF5Writer(
    const std::string& filename, int nx, int ny, int num_species,
    float dx, float dy)
    : nx_(nx), ny_(ny), num_species_(num_species), dx_(dx), dy_(dy),
      timestep_(0), filename_(filename) {
    
    try {
        // Create HDF5 file
        file_ = H5::H5File(filename, H5F_ACC_TRUNC);
        
        // For FVM, dimensions are based on number of cells
        // In HDF5Writer constructor:
        hsize_t dims[4] = {static_cast<hsize_t>(nx_),     // x dimension first
                        static_cast<hsize_t>(ny_),      // y dimension second
                        static_cast<hsize_t>(1),        // Unit thickness in z
                        static_cast<hsize_t>(num_species_)};
        dataspace_ = H5::DataSpace(4, dims);
        
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
        
        H5::DataSet dataset = file_.createDataSet(ss.str(),
                                                H5::PredType::NATIVE_FLOAT,
                                                dataspace_);
        
        // Write the data
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
             // Topology matches the data dimensions for cells
             << "        <Topology TopologyType=\"3DRectMesh\" "
             << "Dimensions=\"" << nx_+1 << " " << ny_+1 << " 2\"/>\n"
             << "        <Geometry GeometryType=\"ORIGIN_DXDYDZ\">\n"
             << "          <DataItem Dimensions=\"3\" NumberType=\"Float\" Format=\"XML\">\n"
             << "            0.0 0.0 0.0\n"
             << "          </DataItem>\n"
             << "          <DataItem Dimensions=\"3\" NumberType=\"Float\" Format=\"XML\">\n"
             << "            " << dx_ << " " << dy_ << " " << dx_ << "\n"
             << "          </DataItem>\n"
             << "        </Geometry>\n";
        
        // Write data for each species
        for (int s = 0; s < num_species_; ++s) {
            xdmf << "        <Attribute Name=\"Species_" << s 
                 << "\" AttributeType=\"Scalar\" Center=\"Cell\">\n"
                 << "          <DataItem ItemType=\"HyperSlab\" Dimensions=\"" 
                 << nx_ << " " << ny_ << " 1\">\n"
                 << "            <DataItem Dimensions=\"3 4\" Format=\"XML\">\n"
                 << "              0 0 0 " << s << "\n"
                 << "              1 1 1 1\n"
                 << "              " << nx_ << " " << ny_ << " 1 1\n"
                 << "            </DataItem>\n"
                 // This matches our HDF5 data structure exactly
                 << "            <DataItem Dimensions=\"" << nx_ << " " << ny_ 
                 << " 1 " << num_species_ << "\" NumberType=\"Float\" Format=\"HDF5\">\n"
                 << "              " << filename_ << ":/concentrations_"
                 << std::setw(6) << std::setfill('0') << i << "\n"
                 << "            </DataItem>\n"
                 << "          </DataItem>\n"
                 << "        </Attribute>\n";
        }
        
        xdmf << "      </Grid>\n";
    }
    
    xdmf << "    </Grid>\n"
         << "  </Domain>\n"
         << "</Xdmf>\n";
}

void IOUtils::HDF5Writer::writeMask(const std::vector<int>& mask, const std::string& name) {
    try {
        // Create dataspace for the mask
        hsize_t dims[2] = {static_cast<hsize_t>(nx_), static_cast<hsize_t>(ny_)};
        H5::DataSpace mask_space(2, dims);
        
        // Create the dataset
        H5::DataSet dataset = file_.createDataSet(
            "/" + name,
            H5::PredType::NATIVE_INT,
            mask_space);
        
        // Write the mask data
        dataset.write(mask.data(), H5::PredType::NATIVE_INT);
        
        // Add attribute to indicate this is a mask
        H5::DataSpace attr_space(H5S_SCALAR);
        H5::StrType str_type(H5::PredType::C_S1, 256);
        H5::Attribute type_attr = dataset.createAttribute(
            "type",
            str_type,
            attr_space);
        const char* data_type = "domain_mask";
        type_attr.write(str_type, data_type);
        
    } catch (const H5::Exception& e) {
        throw std::runtime_error(
            "Failed to write mask: " + std::string(e.getCDetailMsg()));
    }
}

IOUtils::MaskReader::MaskData IOUtils::MaskReader::loadRawMask(
    const std::string& filename, 
    int nx, int ny,
    int clay_label,
    int water_label,
    int co2_label) {
    
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open mask file: " + filename);
    }
    
    // Read the raw 8-bit data
    std::vector<unsigned char> raw_data(nx * ny);
    file.read(reinterpret_cast<char*>(raw_data.data()), nx * ny);
    
    if (file.gcount() != nx * ny) {
        throw std::runtime_error(
            "Invalid mask file size. Expected " + std::to_string(nx * ny) + 
            " bytes, got " + std::to_string(file.gcount()));
    }
    
    MaskData result;
    result.raw_labels = raw_data;
    result.active_cells.resize(nx * ny, 0);
    result.interface_cells.resize(nx * ny, 0);
    result.clay_cells.resize(nx * ny, 0);
    
    // Process the mask
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            int idx = i + j * nx;
            unsigned char val = raw_data[idx];
            
            // Set active cells (both water and clay)
            if (val == water_label || val == clay_label) {
                result.active_cells[idx] = 1;
                
                // Mark clay cells specifically
                if (val == clay_label) {
                    result.clay_cells[idx] = 1;
                }
                
                // Check for water-CO2 interface only if it's a water cell
                if (val == water_label) {
                    // Check neighboring cells for CO2 (4-connectivity)
                    if ((i > 0 && raw_data[idx-1] == co2_label) ||
                        (i < nx-1 && raw_data[idx+1] == co2_label) ||
                        (j > 0 && raw_data[idx-nx] == co2_label) ||
                        (j < ny-1 && raw_data[idx+nx] == co2_label)) {
                        result.interface_cells[idx] = 1;
                    }
                }
            }
        }
    }
    
    return result;
}