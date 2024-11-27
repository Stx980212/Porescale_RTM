#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <memory>
#include "transport2d.cuh"
#include "reactions2d.cuh"
#include "io_utils.hpp"
#include "cuda_utils.cuh"

class ReactiveTransportSolver {
public:
    struct ConvergenceParams {
        float relative_tol;     // Relative tolerance for convergence
        float absolute_tol;     // Absolute tolerance for convergence
        int max_iterations;     // Maximum iterations per timestep
        float mass_tol;         // Tolerance for mass conservation
        
        ConvergenceParams()
            : relative_tol(1e-5f)
            , absolute_tol(1e-7f)
            , max_iterations(50)
            , mass_tol(1e-5f)
        {}
    };
    
    struct ConvergenceStatus {
        bool converged;
        float max_residual;
        float mass_error;
        int iterations;
        std::string divergence_reason;
        
        ConvergenceStatus()
            : converged(false)
            , max_residual(0.0f)
            , mass_error(0.0f)
            , iterations(0)
        {}
    };

    struct InterfaceFluxInfo {
        float accumulated_mass;
        std::vector<float> local_flux;
    };

    InterfaceFluxInfo getInterfaceFluxInfo() const {
        InterfaceFluxInfo info;
        info.accumulated_mass = accumulated_interface_mass_;
        info.local_flux = interface_mass_flux_;
        return info;
    }

    ReactiveTransportSolver(
        int nx, int ny,
        float dx, float dy,
        float dt,
        float total_time,
        int num_species, 
        float Hs_co2,
        float clay_porosity = 0.3f,    // Default clay porosity
        float clay_tortuosity = 0.5f  // Default clay tortuosity
    ): nx_(nx), ny_(ny), dx_(dx), dy_(dy), dt_(dt), initial_dt_(dt),
        total_time_(total_time), num_species_(num_species),
        transport_solver_(nx, ny, dx, dy, dt, num_species),
        Hs_co2_(Hs_co2) {
        
        // Initialize concentrations
        concentrations_.resize(nx * ny * num_species);
        previous_concentrations_.resize(nx * ny * num_species);
        interface_cells_.resize(nx * ny, 0);
        clay_cells_.resize(nx * ny, 0);

        const float P_co2 = 1.0f; // Partial pressure of CO2 in atmospheres
        co2_saturation_conc_ = Hs_co2_ * P_co2;

        // Set default convergence parameters
        conv_params_ = ConvergenceParams();
        
        // Set up reaction parameters
        reaction_params_.k_forward = 1.0f;
        reaction_params_.k_backward = 0.1f;
        reaction_params_.equilibrium_K = 10.0f;

        interface_mass_flux_.resize(nx * ny * num_species_, 0.0f);
        accumulated_interface_mass_ = 0.0f;
        
        // Initialize HDF5 writer
        writer_.reset(new IOUtils::HDF5Writer(
            "reactive_transport.h5", nx, ny, num_species, dx, dy));
    }

    void setConvergenceParams(const ConvergenceParams& params) {
        if (params.relative_tol <= 0.0f || params.relative_tol >= 1.0f) {
            throw std::invalid_argument("Relative tolerance must be between 0 and 1");
        }
        if (params.absolute_tol <= 0.0f) {
            throw std::invalid_argument("Absolute tolerance must be positive");
        }
        if (params.max_iterations <= 0) {
            throw std::invalid_argument("Maximum iterations must be positive");
        }
        if (params.mass_tol <= 0.0f || params.mass_tol >= 1.0f) {
            throw std::invalid_argument("Mass tolerance must be between 0 and 1");
        }
        
        conv_params_ = params;
        std::cout << "Setting convergence parameters:\n"
                  << "  Relative tolerance: " << params.relative_tol << "\n"
                  << "  Absolute tolerance: " << params.absolute_tol << "\n"
                  << "  Maximum iterations: " << params.max_iterations << "\n"
                  << "  Mass tolerance: " << params.mass_tol << std::endl;
    }

    
    void initialize() {
    /*
    // Initializing the concentration field with Gaussian field
    // Get the mask from transport solver (add a getMask() method to TransportSolver2D first)
    const std::vector<int>& mask = transport_solver_.getMask();
 
    // Initialize with Gaussian pulses for species A and B
    const float cx = nx_ / 2.0f;
    const float cy = ny_ / 2.0f;
    const float radius = nx_ / 10.0f;
    
    for (int j = 0; j < ny_; j++) {
        for (int i = 0; i < nx_; i++) {
            // Check if cell is masked (invalid)
            if (!mask[j * nx_ + i]) {
                // Zero all species in masked cells
                for (int s = 0; s < num_species_; s++) {
                    concentrations_[(j * nx_ + i) * num_species_ + s] = 0.0f;
                }
                continue;
            }
            
            // For valid cells, set initial concentrations
            float dx = (i - cx);
            float dy = (j - cy);
            float r2 = (dx*dx + dy*dy)/(radius*radius);
            
            // Species A: Central Gaussian
            concentrations_[(j * nx_ + i) * num_species_ + 0] = 
                std::exp(-r2);
            
            // Species B: Offset Gaussian
            concentrations_[(j * nx_ + i) * num_species_ + 1] = 
                0.5f * std::exp(-(dx*dx + (dy-radius)*(dy-radius))/(radius*radius));
            
            // Species C: Initially zero
            concentrations_[(j * nx_ + i) * num_species_ + 2] = 0.0f;
        }
    }
    */
    std::fill(concentrations_.begin(), concentrations_.end(), 0.0f);
    
    // Save initial state
    writer_->writeTimestep(concentrations_, 0.0f);
    
    // Create XDMF file after first timestep is written
    writer_->createXDMF("reactive_transport.xmf");
    }
    
    void solve() {
        float current_time = 0.0f;
        int step = 0;
        const int save_interval = 100;
        const float MIN_DT = 5e-5f;  // minimum dt
        bool flag = true; // to test if there is any error reported in the simulation
        
        // Store initial mass for conservation checking
        initial_total_mass_ = calculateTotalMass();
        previous_concentrations_ = concentrations_;
        
        std::cout << "Starting simulation...\n"
                  << "Initial total mass: " << initial_total_mass_ << std::endl;
        
        while (current_time < total_time_) {
            // Apply boundary conditions before transport step
            applyBoundaryConditions();  // Apply boundary conditions
            applyClayProperties();      // Apply clay properties to transport
            std::cout << "Current time: " << current_time << ", Step: " << step << std::endl;
            conv_status_ = ConvergenceStatus();
            
            // Store pre-step state
            //std::cout << "Before transport_solver_.solve" << std::endl;
            std::vector<float> pre_step_concentrations = concentrations_;
            //std::cout << "After transport_solver_.solve" << std::endl;
            
            // Iterative solution for current timestep
            while (conv_status_.iterations < conv_params_.max_iterations) {
                // Transport step
                transport_solver_.solve(concentrations_);
                flag = checkConvergence(current_time);
                
                // Check convergence
                if (conv_status_.converged) {
                    std::cout << "Converged at iteration: " << conv_status_.iterations << std::endl;
                    dt_ *= 2.0f;
                    break;
                }

                // Handle non-convergence
            
                if (!conv_status_.converged) {
                    if (dt_ <= MIN_DT) {
                        std::cout << "Error: Failed to converge even at minimum timestep" << std::endl;
                        flag = false;
                        break;  // Exit the simulation
                    }
                    concentrations_ = pre_step_concentrations;
                    dt_ *= 0.5f;
                    dt_ = std::max(dt_,MIN_DT);
                }
                
                conv_status_.iterations++;
            }
            
            current_time += dt_;
            step++;
            
            // Save and print results periodically
            if (step % save_interval == 0) {
                writer_->writeTimestep(concentrations_, current_time);
                printStats(current_time);
                printConvergenceInfo();
                
                // Potentially increase timestep if convergence is good
                if (conv_status_.iterations < conv_params_.max_iterations/4 &&
                    dt_ < initial_dt_) {
                    dt_ = std::min(dt_ * 1.2f, initial_dt_);
                    std::cout << "Increasing timestep to " << dt_ << std::endl;
                }
            }
        }
        
        writer_->createXDMF("reactive_transport.xmf");
        std::cout << "Simulation completed." << std::endl;
    }
    
    void setVelocityField(float vx, float vy) {
        transport_solver_.setVelocity(vx, vy);
    }
    
    void setUniformDiffusionCoefficients(float dx, float dy) {
        transport_solver_.setDiffusion(dx, dy);
    }

    void setMask(const std::vector<int>& mask) {
        transport_solver_.setMask(mask);
    }

    void setInterfaceCells(const std::vector<int>& interface_cells) {
        interface_cells_ = interface_cells;
    }

    TransportSolver2D& getTransportSolver() { 
        return transport_solver_;             
    }

    void setModifiedDiffusion(const std::vector<float>& modified_diffusion) {
        transport_solver_.setModifiedDiffusion(modified_diffusion);
    }

    std::vector<float> getDiffusionCoefficients() const {
        return transport_solver_.getDiffusionCoefficients();
    }

    void setCellVolumes(const std::vector<float>& cell_volumes) {
        transport_solver_.setCellVolumes(cell_volumes);
    }

    // Helper function to set up diffusion for clay and water regions
    void setupDiffusionCoefficients(
        const std::vector<int>& clay_cells,
        const std::vector<int>& active_cells,
        float water_diffusion = 2.0e-9,
        float clay_porosity = 0.3,
        float clay_tortuosity = 0.5) {
        
        std::vector<float> modified_diffusion(nx_ * ny_);
        
        for (int i = 0; i < nx_ * ny_; i++) {
            if (!active_cells[i]) {
                // Inactive cells (non-porous regions)
                modified_diffusion[i] = 0.0f;
            }
            else if (clay_cells[i]) {
                // Clay pores: Apply porosity and tortuosity factors
                modified_diffusion[i] = water_diffusion * clay_porosity * clay_tortuosity;
            }
            else {
                // Water pores: Use regular diffusion coefficient
                modified_diffusion[i] = water_diffusion;
            }
        }
        
        // Set the modified diffusion coefficients in the transport solver
        transport_solver_.setModifiedDiffusion(modified_diffusion);
    }

private:
    // Grid parameters
    int nx_, ny_;
    float dx_, dy_;
    float dt_;
    float initial_dt_;
    float total_time_;
    int num_species_;

    float Hs_co2_;  // Henry's coefficient for CO2
    std::vector<int> interface_cells_; // Store interface cells
    float co2_saturation_conc_; // Saturation concentration of CO2 in water

    std::vector<int> clay_cells_;     // Store clay cell locations
    float clay_porosity_;             // Porosity of clay cells
    float clay_tortuosity_;           // Tortuosity factor for clay cells

    float accumulated_interface_mass_; // Track total mass flux at interfaces
    std::vector<float> interface_mass_flux_; // Track mass flux at each interface cell

    // Solvers and data
    TransportSolver2D transport_solver_;
    ReactionParameters reaction_params_;
    std::vector<float> concentrations_;
    std::vector<float> previous_concentrations_;
    
    // Convergence handling
    ConvergenceParams conv_params_;
    ConvergenceStatus conv_status_;
    float initial_total_mass_;
    
    // Output handling
    std::unique_ptr<IOUtils::HDF5Writer> writer_;

    bool checkConvergence(float current_time) {
        // Check mass conservation
        float total_mass = calculateTotalMass();
        float mass_with_interface = total_mass + accumulated_interface_mass_;
        
        conv_status_.mass_error = std::abs(mass_with_interface - initial_total_mass_) / initial_total_mass_;
        std::cout << "Mass error (including interface flux): " << conv_status_.mass_error << std::endl;
        std::cout << "Accumulated interface mass flux: " << accumulated_interface_mass_ << std::endl;
        
        if (conv_status_.mass_error > conv_params_.mass_tol) {
            conv_status_.divergence_reason = "Mass conservation violated";
            return false;
        }
        
        // Check solution change
        float max_relative_change = 0.0f;
        float max_absolute_change = 0.0f;
        
        for (size_t i = 0; i < concentrations_.size(); ++i) {
            float abs_change = std::abs(concentrations_[i] - previous_concentrations_[i]);
            float rel_change = abs_change / 
                (std::abs(previous_concentrations_[i]) + conv_params_.absolute_tol);
                
            max_absolute_change = std::max(max_absolute_change, abs_change);
            max_relative_change = std::max(max_relative_change, rel_change);
        }
        
        conv_status_.max_residual = max_relative_change;
        std::cout << "Max relative change: " << max_relative_change << ", Max absolute change: " << max_absolute_change << std::endl;
        
        // Store current solution for next iteration
        previous_concentrations_ = concentrations_;
        
        // Check convergence criteria (disabled sepatated concentration check)
        conv_status_.converged = true;
        return true;

        /*
        if (max_relative_change < conv_params_.relative_tol &&
            max_absolute_change < conv_params_.absolute_tol) {
            conv_status_.converged = true;
            return true;
        }
        */

        // Check if solution is bounded
        if (!checkSolutionBounds()) {
            conv_status_.divergence_reason = "Solution out of bounds";
            return false;
        }
        
        return true;
    }
    
    float calculateTotalMass() const {
        float total_mass = 0.0f;
        for (size_t i = 0; i < concentrations_.size(); ++i) {
            total_mass += concentrations_[i];
        }
        return total_mass * dx_ * dy_;  // Account for cell area
    }

    float calculateInterfaceMassFlux() {
        float total_flux = 0.0f;
        
        // Initialize or resize if needed
        if (interface_mass_flux_.size() != nx_ * ny_) {
            interface_mass_flux_.resize(nx_ * ny_, 0.0f);
        }

        // Calculate mass flux for each interface cell
        for (int i = 0; i < nx_ * ny_; i++) {
            if (interface_cells_[i]) {
                // Calculate the mass change due to enforcing constant concentration
                float prev_mass = previous_concentrations_[i * num_species_] * dx_ * dy_;
                float current_mass = co2_saturation_conc_ * dx_ * dy_;
                float mass_flux = (current_mass - prev_mass) / dt_;
                
                interface_mass_flux_[i] = mass_flux;
                total_flux += mass_flux;
            }
        }
        
        return total_flux;
    }
    
    bool checkSolutionBounds() const {
        for (float c : concentrations_) {
            if (std::isnan(c) || std::isinf(c) || c < -conv_params_.absolute_tol) {
                return false;  // Return false if solution is out of bounds
            }
        }
        return true;  // Return true if solution is within bounds
    }
    
    void printConvergenceInfo() const {
        std::cout << "\nConvergence Status:\n"
                  << "Iterations: " << conv_status_.iterations << "\n"
                  << "Max Residual: " << conv_status_.max_residual << "\n"
                  << "Mass Error: " << conv_status_.mass_error << "\n"
                  << "Status: " << (conv_status_.converged ? "Converged" : "Not Converged");
        
        if (!conv_status_.converged && !conv_status_.divergence_reason.empty()) {
            std::cout << "\nDivergence Reason: " << conv_status_.divergence_reason;
        }
        std::cout << std::endl;
    }
    
    void printStats(float time) {
        float total_A = 0.0f, total_B = 0.0f, total_C = 0.0f;
        float max_A = 0.0f, max_B = 0.0f, max_C = 0.0f;

        float total_interface_flux = 0.0f;
        float max_interface_flux = 0.0f;
        
        for (int i = 0; i < nx_ * ny_; i++) {
            float A = concentrations_[i * num_species_ + 0];
            float B = concentrations_[i * num_species_ + 1];
            float C = concentrations_[i * num_species_ + 2];
            
            total_A += A;
            total_B += B;
            total_C += C;
            
            max_A = std::max(max_A, A);
            max_B = std::max(max_B, B);
            max_C = std::max(max_C, C);

            if (interface_cells_[i]) {
                float flux = interface_mass_flux_[i * num_species_];  // CO2 flux
                total_interface_flux += flux;
                max_interface_flux = std::max(max_interface_flux, std::abs(flux));
            }
        }
        
        std::cout << "Time: " << time << "\n"
                  << "Total mass - A: " << total_A << ", B: " << total_B 
                  << ", C: " << total_C << "\n"
                  << "Max values - A: " << max_A << ", B: " << max_B 
                  << ", C: " << max_C << std::endl;
        
        std::cout << "Interface mass flux - Total: " << total_interface_flux 
                  << ", Max: " << max_interface_flux 
                  << ", Accumulated: " << accumulated_interface_mass_ << std::endl;
    }

    void applyBoundaryConditions() {
        // Calculate and store mass changes from interface condition
        for (int i = 0; i < nx_ * ny_; i++) {
            if (interface_cells_[i]) {
                for (int s = 0; s < num_species_; s++) {
                    int idx = i * num_species_ + s;
                    float old_concentration = concentrations_[idx];
                    
                    // Apply boundary condition (only for CO2 - species 0)
                    if (s == 0) {
                        concentrations_[idx] = co2_saturation_conc_;
                        
                        // Calculate mass change
                        float mass_change = (concentrations_[idx] - old_concentration) * dx_ * dy_;
                        if (!clay_cells_[i]) {
                            interface_mass_flux_[idx] = mass_change / dt_; // Store flux
                            accumulated_interface_mass_ += mass_change;     // Accumulate total mass change
                        } else {
                            // For clay cells, account for porosity
                            interface_mass_flux_[idx] = mass_change * clay_porosity_ / dt_;
                            accumulated_interface_mass_ += mass_change * clay_porosity_;
                        }
                    }
                }
            }
        }
    }

    void applyClayProperties() {
        // Modify transport parameters in clay cells if needed
        // This could include adjusting diffusion coefficients, reaction rates, etc.
        for (int i = 0; i < nx_ * ny_; i++) {
            if (clay_cells_[i]) {
                // Apply clay-specific modifications to transport/reaction parameters
                // For example, different reaction rates in clay vs water
                for (int s = 0; s < num_species_; s++) {
                    int idx = i * num_species_ + s;
                    // Apply clay porosity to concentrations if needed
                    concentrations_[idx] *= clay_porosity_;
                }
            }
        }
    }
};

int main() {
    // Simulation parameters
    const int nx = 200;
    const int ny = 200;
    const float dx = 0.01f;
    const float dy = 0.01f;
    const float dt = 0.00005f;
    const float total_time = 1.0f; // time will be counted as hour in the simulation
    const int num_species = 3; 

    const float Hs_co2 = 0.034f; // Henry's coefficient for CO2 in water mol/(L⋅atm) at 25°C 
    const float clay_porosity = 0.3f; // Typical clay porosity
    const float clay_tortuosity = 0.5f; // Typical clay tortuosity

    const std::string mask_file = "../Wallula_2810_pore1_final_slice73.raw";


    
    try {
        auto mask_data = IOUtils::MaskReader::loadRawMask(
        mask_file, nx, ny, 0, 1, 2);  // clay_label=0, water_label=1, co2_label=2

        // Create solver
        ReactiveTransportSolver solver(
            nx, ny, dx, dy, dt, total_time, num_species, Hs_co2,
            clay_porosity, clay_tortuosity);

        std::vector<int> active_cells = mask_data.active_cells;
        std::vector<float> modified_diffusion(nx * ny);
        std::vector<int> clay_cells = mask_data.clay_cells;
        std::vector<float> cell_volumes(nx * ny);

        for (int i = 0; i < nx * ny; ++i) {
            if (clay_cells[i]==1) {
                cell_volumes[i] = clay_porosity * dx * dy;  // Apply clay porosity
            } else {
                cell_volumes[i] = dx * dy;  // Full volume for water phase
            }
        }
        
        solver.getTransportSolver().setMask(mask_data.active_cells);
        solver.setInterfaceCells(mask_data.interface_cells);
        solver.setCellVolumes(cell_volumes);

        // Set convergence parameters
        ReactiveTransportSolver::ConvergenceParams conv_params;
        conv_params.relative_tol = 1e-4f;
        conv_params.absolute_tol = 1e-6f;
        conv_params.max_iterations = 50;
        conv_params.mass_tol = 1e-3f;
        solver.setConvergenceParams(conv_params);
        
        // Set physical parameters
        solver.setVelocityField(0.0f, 0.0f);
        //solver.setUniformDiffusionCoefficients(0.02f, 0.02f);
        solver.setupDiffusionCoefficients(
            clay_cells, active_cells,
            4.0e-2,  // water diffusion coefficient
            0.3,     // clay porosity
            0.5      // clay tortuosity
            );
        
        // Initialize and run
        solver.initialize();
        solver.solve();
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}