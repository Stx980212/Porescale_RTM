#include "transport2d.cuh"
#include "fvm_utils.cuh"
#include "cuda_utils.cuh"  // Include the new header
#include <iostream>

// Calculate fluxes at cell interfaces
__global__ void calculateFluxesKernel(
    const float* concentrations,    // Cell-averaged concentrations
    float* fluxes_x,               // Fluxes at x-interfaces
    float* fluxes_y,               // Fluxes at y-interfaces
    int nx, int ny,
    float dx, float dy,
    float dt,
    int num_species,
    float2 velocity,
    float2 diffusion
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Calculate x-direction fluxes
    if (i < nx+1 && j < ny) {  // +1 for interfaces
        for (int s = 0; s < num_species; s++) {
            float cL, cR;  // Left and right states
            
            if (i == 0) {  // Left boundary
                cL = concentrations[(j * nx) * num_species + s];
                cR = cL;  // Zero gradient
            }
            else if (i == nx) {  // Right boundary
                cR = concentrations[(j * nx + nx-1) * num_species + s];
                cL = cR;  // Zero gradient
            }
            else {  // Interior interfaces
                cL = concentrations[(j * nx + i-1) * num_species + s];
                cR = concentrations[(j * nx + i) * num_species + s];
            }
            
            // MUSCL reconstruction
            //float cL_interface, cR_interface;
            //FVMUtils::Reconstruction::muscl(cL, cR, dx, cL_interface, cR_interface);
            
            float cL_interface = cL;
            float cR_interface = cR;
            
            // Calculate advective flux using upwind scheme
            float flux_adv = FVMUtils::NumericalFlux::upwind(
                cL_interface, cR_interface, velocity.x);
            
            // Calculate diffusive flux
            float flux_diff = FVMUtils::NumericalFlux::diffusive(
                cL, cR, dx, diffusion.x);
            
            // Store total flux
            fluxes_x[(j * (nx+1) + i) * num_species + s] = flux_adv + flux_diff;
        }
    }
    
    // Calculate y-direction fluxes
    if (i < nx && j < ny+1) {
        for (int s = 0; s < num_species; s++) {
            float cB, cT;  // Bottom and top states
            
            if (j == 0) {  // Bottom boundary
                cB = concentrations[i * num_species + s];
                cT = cB;  // Zero gradient
            }
            else if (j == ny) {  // Top boundary
                cT = concentrations[((ny-1) * nx + i) * num_species + s];
                cB = cT;  // Zero gradient
            }
            else {  // Interior interfaces
                cB = concentrations[((j-1) * nx + i) * num_species + s];
                cT = concentrations[(j * nx + i) * num_species + s];
            }
            
            // MUSCL reconstruction
            float cB_interface, cT_interface;
            FVMUtils::Reconstruction::muscl(cB, cT, dy, cB_interface, cT_interface);
            
            // Calculate advective flux using upwind scheme
            float flux_adv = FVMUtils::NumericalFlux::upwind(
                cB_interface, cT_interface, velocity.y);
            
            // Calculate diffusive flux
            float flux_diff = FVMUtils::NumericalFlux::diffusive(
                cB, cT, dy, diffusion.y);
            
            // Store total flux
            fluxes_y[(j * nx + i) * num_species + s] = flux_adv + flux_diff;
        }
    }
}

// Update cell averages using calculated fluxes
__global__ void updateConcentrationsKernel(
    float* concentrations_new,
    const float* concentrations,
    const float* fluxes_x,
    const float* fluxes_y,
    int nx, int ny,
    float dx, float dy,
    float dt,
    int num_species
) {
    const float UNDER_RELAX = 0.8f;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i >= nx || j >= ny) return;
    
    for (int s = 0; s < num_species; s++) {
        int idx = (j * nx + i) * num_species + s;
        
        // Get fluxes at all cell interfaces
        float flux_left = fluxes_x[(j * (nx+1) + i) * num_species + s];
        float flux_right = fluxes_x[(j * (nx+1) + i + 1) * num_species + s];
        float flux_bottom = fluxes_y[(j * nx + i) * num_species + s];
        float flux_top = fluxes_y[((j+1) * nx + i) * num_species + s];
        
        // Update cell average using flux differencing
        // This ensures conservation
        concentrations_new[idx] = concentrations[idx] -
            (dt/dx) * (flux_right - flux_left) -
            (dt/dy) * (flux_top - flux_bottom);
            
        concentrations_new[idx] = UNDER_RELAX * concentrations_new[idx] + 
                         (1.0f - UNDER_RELAX) * concentrations[idx];

        // Ensure positivity
        concentrations_new[idx] = fmaxf(0.0f, concentrations_new[idx]);
    }
}

TransportSolver2D::TransportSolver2D(
    int nx, int ny, float dx, float dy, float dt, int num_species)
    : nx_(nx), ny_(ny), dx_(dx), dy_(dy), dt_(dt), num_species_(num_species) {
    // Allocate device memory
    checkCudaErrors(cudaMalloc(&d_concentrations_, nx * ny * num_species * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_concentrations_new_, nx * ny * num_species * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fluxes_x_, (nx+1) * ny * num_species * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fluxes_y_, nx * (ny+1) * num_species * sizeof(float)));
     
    // Initialize parameters
    velocity_ = make_float2(0.0f, 0.0f);
    diffusion_ = make_float2(0.001f, 0.001f);
}

TransportSolver2D::~TransportSolver2D() {
    checkCudaErrors(cudaFree(d_concentrations_));
    checkCudaErrors(cudaFree(d_concentrations_new_));
    checkCudaErrors(cudaFree(d_fluxes_x_));
    checkCudaErrors(cudaFree(d_fluxes_y_));
}

void TransportSolver2D::solve(std::vector<float>& concentrations) {
    // Copy data to device
    checkCudaErrors(cudaMemcpy(d_concentrations_, concentrations.data(),
               nx_ * ny_ * num_species_ * sizeof(float),
               cudaMemcpyHostToDevice));
    
    // Set up grid and block dimensions
    dim3 block_size(16, 16);
    dim3 num_blocks_fluxes(
        (nx_ + 2 + block_size.x - 1) / block_size.x,
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
        nx_, ny_,
        dx_, dy_,
        dt_,
        num_species_,
        velocity_,
        diffusion_
    );
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
        
    // Step 2: Update cell averages using computed fluxes
    updateConcentrationsKernel<<<num_blocks_update, block_size>>>(
        d_concentrations_new_,
        d_concentrations_,
        d_fluxes_x_,
        d_fluxes_y_,
        nx_, ny_,
        dx_, dy_,
        dt_,
        num_species_
    );
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
        
    // Copy results back to host
    checkCudaErrors(cudaMemcpy(concentrations.data(), d_concentrations_new_,
               nx_ * ny_ * num_species_ * sizeof(float),
               cudaMemcpyDeviceToHost));
    
    // Swap pointers for next iteration
    float* temp = d_concentrations_;
    d_concentrations_ = d_concentrations_new_;
    d_concentrations_new_ = temp;
}

void TransportSolver2D::setVelocity(float vx, float vy) {
    velocity_ = make_float2(vx, vy);
}

void TransportSolver2D::setDiffusion(float dx, float dy) {
    diffusion_ = make_float2(dx, dy);
}
