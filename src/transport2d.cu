#include "transport2d.cuh"
#include "fvm_utils.cuh"
#include "cuda_utils.cuh" 
#include <iostream>
#include <cstring>

// Calculate fluxes at cell interfaces
__global__ void calculateFluxesKernel(
    const double* concentrations,    // Cell-averaged concentrations
    double* fluxes_x,               // Fluxes at x-interfaces
    double* fluxes_y,               // Fluxes at y-interfaces
    const int* mask,      
    const double* modified_diffusion,  // Add modified_diffusion parameter
    bool has_modified_diffusion,      // Add flag for modified diffusion

    int nx, int ny,
    double dx, double dy,
    double dt,
    int num_species,
    double2 velocity,
    double2 diffusion
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Calculate x-direction fluxes
    if (i < nx+1 && j < ny) {  // +1 for interfaces
        for (int s = 0; s < num_species; s++) {
            // no-flux boundary
            if (i == 0 || i == nx || 
                (i > 0 && i < nx && (!mask[(j * nx + i-1)] || !mask[(j * nx + i)]))) {
                fluxes_x[(j * (nx+1) + i) * num_species + s] = 0.0f;
                continue;
            }

            double cL = concentrations[(j * nx + i-1) * num_species + s];
            double cR = concentrations[(j * nx + i) * num_species + s];
            
            // MUSCL reconstruction
            //double cL_interface, cR_interface;
            //FVMUtils::Reconstruction::muscl(cL, cR, dx, cL_interface, cR_interface);
            
            // Calculate advective flux using upwind scheme
            double flux_adv = FVMUtils::NumericalFlux::upwind(cL, cR, velocity.x);

            // Get appropriate diffusion coefficient
            double diff_coef;
            if (has_modified_diffusion) {
                // Average diffusion coefficients of adjacent cells
                double diff_L = modified_diffusion[j * nx + i-1];
                double diff_R = modified_diffusion[j * nx + i];
                diff_coef = 2.0f * (diff_L * diff_R) / (diff_L + diff_R + 1e-20f);
            } else {
                diff_coef = diffusion.x;
            }

            // Calculate diffusive flux with modified coefficient
            double flux_diff = FVMUtils::NumericalFlux::diffusive(cL, cR, dx, diff_coef);
            
            // Store total flux
            fluxes_x[(j * (nx+1) + i) * num_species + s] = flux_adv + flux_diff;
        }
    }
    
    // Calculate y-direction fluxes
    if (i < nx && j < ny+1) {
        for (int s = 0; s < num_species; s++) {
            // At domain boundaries, set flux to zero
            if (j == 0 || j == ny || 
                (j > 0 && j < ny && (!mask[((j-1) * nx + i)] || !mask[(j * nx + i)]))) {
                fluxes_y[(j * nx + i) * num_species + s] = 0.0f;
                continue;
            }
            
            double cB = concentrations[((j-1) * nx + i) * num_species + s];
            double cT = concentrations[(j * nx + i) * num_species + s];
            
            // Calculate advective and diffusive fluxes for interior points
            double flux_adv = FVMUtils::NumericalFlux::upwind(cB, cT, velocity.y);

            // Get appropriate diffusion coefficient
            double diff_coef;
            if (has_modified_diffusion) {
                // Average diffusion coefficients of adjacent cells
                double diff_B = modified_diffusion[(j-1) * nx + i];
                double diff_T = modified_diffusion[j * nx + i];
                diff_coef = 2.0f * (diff_B * diff_T) / (diff_B + diff_T + 1e-20f);
            } else {
                diff_coef = diffusion.y;
            }

            // Calculate diffusive flux with modified coefficient
            double flux_diff = FVMUtils::NumericalFlux::diffusive(cB, cT, dy, diff_coef);
            
            fluxes_y[(j * nx + i) * num_species + s] = flux_adv + flux_diff;
        }
    }
}

// Update cell averages using calculated fluxes
__global__ void updateConcentrationsKernel(
    double* concentrations_new,
    const double* concentrations,
    const double* fluxes_x,
    const double* fluxes_y,
    const double* cell_volumes,
    int nx, int ny,
    double dx, double dy,
    double dt,
    int num_species
) {
    const double UNDER_RELAX = 1.0f;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i >= nx || j >= ny) return;
    
    for (int s = 0; s < num_species; s++) {
        int idx = (j * nx + i) * num_species + s;
        
        // Get fluxes at all cell interfaces
        double flux_left = fluxes_x[(j * (nx+1) + i) * num_species + s];
        double flux_right = fluxes_x[(j * (nx+1) + i + 1) * num_species + s];
        double flux_bottom = fluxes_y[(j * nx + i) * num_species + s];
        double flux_top = fluxes_y[((j+1) * nx + i) * num_species + s];

        // Calculate mass change
        double dmass = -dt * (
            (flux_right - flux_left) * dy +  // Mass flux through x-faces
            (flux_top - flux_bottom) * dx    // Mass flux through y-faces
        );

        // Update concentration based on mass change and cell volume
        double new_mass = concentrations[idx] * cell_volumes[j * nx + i] + dmass;
        double new_conc = new_mass / cell_volumes[j * nx + i];

        // Apply underrelaxation
        concentrations_new[idx] = UNDER_RELAX * new_conc + 
                                 (1.0f - UNDER_RELAX) * concentrations[idx];

        // Ensure positivity
        concentrations_new[idx] = fmaxf(0.0f, concentrations_new[idx]);
    }
}


TransportSolver2D::TransportSolver2D(
    int nx, int ny, double dx, double dy, double dt, int num_species)
    : nx_(nx), ny_(ny), dx_(dx), dy_(dy), dt_(dt), num_species_(num_species), d_mask_(nullptr), d_modified_diffusion_(nullptr), 
      has_modified_diffusion_(false) {
    // Allocate device memory
    checkCudaErrors(cudaMalloc(&d_concentrations_, nx * ny * num_species * sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_concentrations_new_, nx * ny * num_species * sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_fluxes_x_, (nx+1) * ny * num_species * sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_fluxes_y_, nx * (ny+1) * num_species * sizeof(double)));
     
    // Allocate mask memory
    checkCudaErrors(cudaMalloc(&d_mask_, nx * ny * sizeof(int)));
    // Initialize mask to all valid (1)
    checkCudaErrors(cudaMemset(d_mask_, 1, nx * ny * sizeof(int)));

    // Allocate modified diffusion coefficients memory
    checkCudaErrors(cudaMalloc(&d_modified_diffusion_, nx * ny * sizeof(double)));

    // Initialize with default diffusion values
    std::vector<double> default_diffusion(nx * ny, diffusion_.x);
    checkCudaErrors(cudaMemcpy(d_modified_diffusion_, default_diffusion.data(),
                              nx * ny * sizeof(double), cudaMemcpyHostToDevice));

    // Allocate memory for cell volumes
    checkCudaErrors(cudaMalloc(&d_cell_volumes_, nx * ny * sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_porosity_, nx_ * ny_ * sizeof(double)));

    
    // Initialize with unit volumes
    std::vector<double> unit_volumes(nx * ny, 1.0f);
    checkCudaErrors(cudaMemcpy(d_cell_volumes_, unit_volumes.data(),
                              nx * ny * sizeof(double), cudaMemcpyHostToDevice));

    // Initialize parameters
    velocity_ = make_double2(0.0f, 0.0f);
    diffusion_ = make_double2(0.001f, 0.001f);
}

TransportSolver2D::~TransportSolver2D() {
    checkCudaErrors(cudaFree(d_concentrations_));
    checkCudaErrors(cudaFree(d_concentrations_new_));
    checkCudaErrors(cudaFree(d_fluxes_x_));
    checkCudaErrors(cudaFree(d_fluxes_y_));
    if (d_mask_) {
        checkCudaErrors(cudaFree(d_mask_));
    }
    if (d_modified_diffusion_) {
        checkCudaErrors(cudaFree(d_modified_diffusion_));
    }
    if (d_cell_volumes_) {
        checkCudaErrors(cudaFree(d_cell_volumes_));
    }
    if (d_porosity_) {
        cudaFree(d_porosity_);
        d_porosity_ = nullptr;
    }
}

std::vector<double> TransportSolver2D::getDiffusionCoefficients() const {
    if (!has_modified_diffusion_) {
        // Return uniform diffusion coefficients
        return std::vector<double>(nx_ * ny_, diffusion_.x);  // Assuming isotropic diffusion
    }

    // Return modified diffusion coefficients
    std::vector<double> host_diffusion(nx_ * ny_);
    checkCudaErrors(cudaMemcpy(host_diffusion.data(), d_modified_diffusion_,
                              nx_ * ny_ * sizeof(double),
                              cudaMemcpyDeviceToHost));
    return host_diffusion;
}

void TransportSolver2D::setModifiedDiffusion(const std::vector<double>& modified_diffusion) {
    if (modified_diffusion.size() != nx_ * ny_) {
        throw std::runtime_error("Modified diffusion array size does not match domain dimensions");
    }

    // Allocate device memory if not already done
    if (!has_modified_diffusion_) {
        checkCudaErrors(cudaMalloc(&d_modified_diffusion_, nx_ * ny_ * sizeof(double)));
    }

    checkCudaErrors(cudaMemcpy(d_modified_diffusion_, modified_diffusion.data(),
                              nx_ * ny_ * sizeof(double),
                              cudaMemcpyHostToDevice));
    has_modified_diffusion_ = true;
}

void TransportSolver2D::setCellVolumes(const std::vector<double>& volumes) {
    if (volumes.size() != nx_ * ny_) {
        throw std::runtime_error("Cell volumes array size does not match domain dimensions");
    }
    
    checkCudaErrors(cudaMemcpy(d_cell_volumes_, volumes.data(),
                              nx_ * ny_ * sizeof(double),
                              cudaMemcpyHostToDevice));
}

void TransportSolver2D::setPorosity(const std::vector<double>& porosity) {
    if (porosity.size() != nx_ * ny_) {
        throw std::runtime_error("Porosity vector size mismatch");
    }
    checkCudaErrors(cudaMemcpy(d_porosity_, porosity.data(),
                              nx_ * ny_ * sizeof(double),
                              cudaMemcpyHostToDevice));
}

std::vector<double> TransportSolver2D::getCellVolumes() const {
    std::vector<double> host_volumes(nx_ * ny_);
    checkCudaErrors(cudaMemcpy(host_volumes.data(), d_cell_volumes_,
                              nx_ * ny_ * sizeof(double),
                              cudaMemcpyDeviceToHost));
    return host_volumes;
}

std::vector<double> TransportSolver2D::getPorosity() const {
    std::vector<double> host_porosity(nx_ * ny_);
    checkCudaErrors(cudaMemcpy(host_porosity.data(), d_porosity_,
                              nx_ * ny_ * sizeof(double),
                              cudaMemcpyDeviceToHost));
    return host_porosity;
}

double TransportSolver2D::getTotalMass() const {
    std::vector<double> host_concentrations(nx_ * ny_ * num_species_);
    std::vector<double> host_volumes(nx_ * ny_);
    
    checkCudaErrors(cudaMemcpy(host_concentrations.data(), d_concentrations_,
                              nx_ * ny_ * num_species_ * sizeof(double),
                              cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(host_volumes.data(), d_cell_volumes_,
                              nx_ * ny_ * sizeof(double),
                              cudaMemcpyDeviceToHost));
    
    double total_mass = 0.0f;
    for (int i = 0; i < nx_ * ny_; ++i) {
        for (int s = 0; s < num_species_; ++s) {
            total_mass += host_concentrations[i * num_species_ + s] * host_volumes[i];
        }
    }
    return total_mass;
}

void TransportSolver2D::solve(std::vector<double>& concentrations) {
    checkCFLCondition();
    // Copy data to device
    checkCudaErrors(cudaMemcpy(d_concentrations_, concentrations.data(),
               nx_ * ny_ * num_species_ * sizeof(double),
               cudaMemcpyHostToDevice));
    
    // Set up grid and block dimensions
    dim3 block_size(16, 16);
    dim3 num_blocks_fluxes(
        (nx_ + 1 + block_size.x - 1) / block_size.x,
        (ny_ + 1 + block_size.y - 1) / block_size.y
    );
    dim3 num_blocks_update(
        (nx_ + block_size.x - 1) / block_size.x,
        (ny_ + block_size.y - 1) / block_size.y
    );
    
    // Step 1: Calculate fluxes at cell interfaces
    calculateFluxesKernel<<<num_blocks_fluxes, block_size>>>(
        d_concentrations_,
        d_fluxes_x_,
        d_fluxes_y_,
        d_mask_,
        d_modified_diffusion_,  // Add modified diffusion array
        has_modified_diffusion_, // Add flag for using modified diffusion
        nx_, ny_,
        dx_, dy_,
        dt_,
        num_species_,
        velocity_,
        diffusion_
    );
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    checkBoundaryFluxes();
        
    // Step 2: Update cell averages using computed fluxes
    updateConcentrationsKernel<<<num_blocks_update, block_size>>>(
        d_concentrations_new_,
        d_concentrations_,
        d_fluxes_x_,
        d_fluxes_y_,
        d_cell_volumes_,
        nx_, ny_,
        dx_, dy_,
        dt_,
        num_species_
    );
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
        
    // Copy results back to host
    checkCudaErrors(cudaMemcpy(concentrations.data(), d_concentrations_new_,
               nx_ * ny_ * num_species_ * sizeof(double),
               cudaMemcpyDeviceToHost));
    
    // Swap pointers for next iteration
    double* temp = d_concentrations_;
    d_concentrations_ = d_concentrations_new_;
    d_concentrations_new_ = temp;
}

void TransportSolver2D::setVelocity(double vx, double vy) {
    velocity_ = make_double2(vx, vy);
}

void TransportSolver2D::setDiffusion(double dx, double dy) {
    diffusion_ = make_double2(dx, dy);
}

void TransportSolver2D::checkCFLCondition() {
    double max_velocity = fmax(abs(velocity_.x), abs(velocity_.y));
    double diff_cfl = (diffusion_.x/(dx_*dx_) + diffusion_.y/(dy_*dy_)) * dt_;
    double adv_cfl = max_velocity * dt_ / fmin(dx_, dy_);
    
    if (diff_cfl > 0.25 || adv_cfl > 1.0) {  // Changed from 0.5 to 0.25 for diffusion
        std::cerr << "Warning: CFL condition might be violated\n"
                  << "Diffusive CFL: " << diff_cfl << "\n"
                  << "Advective CFL: " << adv_cfl << std::endl;
    }
}

void TransportSolver2D::checkBoundaryFluxes() {
    std::vector<double> h_fluxes_x((nx_+1) * ny_ * num_species_);
    std::vector<double> h_fluxes_y(nx_ * (ny_+1) * num_species_);
    
    // Copy fluxes back to host for checking
    cudaMemcpy(h_fluxes_x.data(), d_fluxes_x_, 
               h_fluxes_x.size() * sizeof(double), 
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fluxes_y.data(), d_fluxes_y_, 
               h_fluxes_y.size() * sizeof(double), 
               cudaMemcpyDeviceToHost);
    
    // Check x-direction boundary fluxes
    for (int j = 0; j < ny_; j++) {
        for (int s = 0; s < num_species_; s++) {
            double left_flux = h_fluxes_x[(j * (nx_+1)) * num_species_ + s];
            double right_flux = h_fluxes_x[(j * (nx_+1) + nx_) * num_species_ + s];
            if (std::abs(left_flux) > 1e-10 || std::abs(right_flux) > 1e-10) {
                std::cout << "Warning: Non-zero boundary flux detected in x-direction\n";
                std::cout << "j=" << j << ", species=" << s 
                         << ", left=" << left_flux 
                         << ", right=" << right_flux << std::endl;
            }
        }
    }
    
    // Check y-direction boundary fluxes
    for (int i = 0; i < nx_; i++) {
        for (int s = 0; s < num_species_; s++) {
            double bottom_flux = h_fluxes_y[i * num_species_ + s];
            double top_flux = h_fluxes_y[(ny_ * nx_ + i) * num_species_ + s];
            if (std::abs(bottom_flux) > 1e-10 || std::abs(top_flux) > 1e-10) {
                std::cout << "Warning: Non-zero boundary flux detected in y-direction\n";
                std::cout << "i=" << i << ", species=" << s 
                         << ", bottom=" << bottom_flux 
                         << ", top=" << top_flux << std::endl;
            }
        }
    }
}

void TransportSolver2D::setMask(const std::vector<int>& mask) {
    if (mask.size() != nx_ * ny_) {
        throw std::runtime_error("Mask size does not match domain dimensions");
    }
    
    // Copy mask to device
    checkCudaErrors(cudaMemcpy(d_mask_, mask.data(), 
                              nx_ * ny_ * sizeof(int), 
                              cudaMemcpyHostToDevice));
}

std::vector<int> TransportSolver2D::getMask() const {
    std::vector<int> host_mask(nx_ * ny_);
    checkCudaErrors(cudaMemcpy(host_mask.data(), d_mask_,
                             nx_ * ny_ * sizeof(int),
                             cudaMemcpyDeviceToHost));
    return host_mask;
}