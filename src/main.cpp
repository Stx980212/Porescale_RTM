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
        double relative_tol;     // Relative tolerance for convergence
        double absolute_tol;     // Absolute tolerance for convergence
        int max_iterations;     // Maximum iterations per timestep
        double mass_tol;         // Tolerance for mass conservation
        
        ConvergenceParams()
            : relative_tol(1e-5f)
            , absolute_tol(1e-7f)
            , max_iterations(50)
            , mass_tol(1e-5f)
        {}
    };
    
    struct ConvergenceStatus {
        bool converged;
        double max_residual;
        double mass_error;
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
        double accumulated_mass;
        std::vector<double> local_flux;
    };

    InterfaceFluxInfo getInterfaceFluxInfo() const {
        InterfaceFluxInfo info;
        info.accumulated_mass = accumulated_interface_mass_;
        info.local_flux = interface_mass_flux_;
        return info;
    }

    ReactiveTransportSolver(
        int nx, int ny,
        double dx, double dy,
        double dt,
        double total_time,
        int num_species, 
        double Hs_co2,
        double clay_porosity = 0.3f,    // Default clay porosity
        double clay_tortuosity = 0.5f  // Default clay tortuosity
    ): nx_(nx), ny_(ny), dx_(dx), dy_(dy), dt_(dt), initial_dt_(dt),
        total_time_(total_time), num_species_(num_species),
        transport_solver_(nx, ny, dx, dy, dt, num_species),
        Hs_co2_(Hs_co2),
        clay_porosity_(clay_porosity),
        clay_tortuosity_(clay_tortuosity) {
        
        // Initialize concentrations
        concentrations_.resize(nx * ny * num_species);
        previous_concentrations_.resize(nx * ny * num_species);
        interface_cells_.resize(nx * ny, 0);
        clay_cells_.resize(nx * ny, 0);
        cell_volumes_.resize(nx * ny);
        porosity_.resize(nx * ny);

        const double P_co2 = 1.0f; // Partial pressure of CO2 in atmospheres
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
    const double cx = nx_ / 2.0f;
    const double cy = ny_ / 2.0f;
    const double radius = nx_ / 10.0f;
    
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
            double dx = (i - cx);
            double dy = (j - cy);
            double r2 = (dx*dx + dy*dy)/(radius*radius);
            
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
   // Get the current cell volumes from transport solver
    porosity_ = transport_solver_.getPorosity();
    cell_volumes_ = transport_solver_.getCellVolumes();

    std::fill(concentrations_.begin(), concentrations_.end(), 0.0f);
    
    // Save initial state
    writer_->writeTimestep(concentrations_, cell_volumes_, porosity_, 0.0f);
    
    // Create XDMF file after first timestep is written
    writer_->createXDMF("reactive_transport.xmf");
    }
    
    void solve() {
        double current_time = 0.0f;
        int step = 0;
        const int save_interval = 100;
        const double MIN_DT = 1e-5f;  // minimum dt
        const double MAX_DT = 1e-5f;  // minimum dt
        bool flag = true; // to test if there is any error reported in the simulation
        
        // Store initial mass for conservation checking
        initial_total_mass_ = calculateTotalMass();
        previous_concentrations_ = concentrations_;
        
        std::cout << "Starting simulation...\n"
                  << "Initial total mass: " << initial_total_mass_ << std::endl;
        
        while (current_time < total_time_) {
            // Apply boundary conditions before transport step
            applyBoundaryConditions();  // Apply boundary conditions
            std::cout << "Current time: " << current_time << ", Step: " << step << std::endl;
            conv_status_ = ConvergenceStatus();
            
            // Store pre-step state
            //std::cout << "Before transport_solver_.solve" << std::endl;
            std::vector<double> pre_step_concentrations = concentrations_;

            //std::cout << "After transport_solver_.solve" << std::endl;
            
            // Iterative solution for current timestep
            while (conv_status_.iterations < conv_params_.max_iterations) {
                // Transport step
                checkConvergence(current_time);
                transport_solver_.solve(concentrations_);
                flag = checkConvergence(current_time);
                
                // Check convergence
                if (conv_status_.converged) {
                    std::cout << "Converged at iteration: " << conv_status_.iterations << std::endl;
                    dt_ *= 2.0f;
                    dt_ = std::min(dt_,MAX_DT);
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
                porosity_ = transport_solver_.getPorosity();
                cell_volumes_ = transport_solver_.getCellVolumes();
                writer_->writeTimestep(concentrations_, cell_volumes_, porosity_, current_time);
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
    
    void setVelocityField(double vx, double vy) {
        transport_solver_.setVelocity(vx, vy);
    }
    
    void setUniformDiffusionCoefficients(double dx, double dy) {
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

    void setModifiedDiffusion(const std::vector<double>& modified_diffusion) {
        transport_solver_.setModifiedDiffusion(modified_diffusion);
    }

    std::vector<double> getDiffusionCoefficients() const {
        return transport_solver_.getDiffusionCoefficients();
    }

    void setCellVolumes(const std::vector<double>& cell_volumes) {
        transport_solver_.setCellVolumes(cell_volumes);
    }

    // Helper function to set up diffusion for clay and water regions
    void setupDiffusionCoefficients(
        const std::vector<int>& clay_cells,
        const std::vector<int>& active_cells,
        double water_diffusion = 2.0e-9,
        double clay_porosity = 0.3,
        double clay_tortuosity = 0.5) {
        
        std::vector<double> modified_diffusion(nx_ * ny_);
        
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

    void updatePorosity() {
        for (int i = 0; i < nx_ * ny_; i++) {
            if (clay_cells_[i]) {
                // Update porosity based on your porosity evolution model
                // This is just a placeholder - implement your actual porosity update logic
                porosity_[i] = clay_porosity_;  // or your updated value
            }
        }
    }

    void setPorosity(const std::vector<double>& porosity) {
        transport_solver_.setPorosity(porosity);
    }

private:
    // Grid parameters
    int nx_, ny_;
    double dx_, dy_;
    double dt_;
    double initial_dt_;
    double total_time_;
    int num_species_;

    double Hs_co2_;  // Henry's coefficient for CO2
    std::vector<int> interface_cells_; // Store interface cells
    double co2_saturation_conc_; // Saturation concentration of CO2 in water

    std::vector<int> clay_cells_;     // Store clay cell locations
    double clay_porosity_;             // Porosity of clay cells
    double clay_tortuosity_;           // Tortuosity factor for clay cells

    double accumulated_interface_mass_; // Track total mass flux at interfaces
    std::vector<double> interface_mass_flux_; // Track mass flux at each interface cell

    std::vector<double> cell_volumes_;
    std::vector<double> porosity_;

    // Solvers and data
    TransportSolver2D transport_solver_;
    ReactionParameters reaction_params_;
    std::vector<double> concentrations_;
    std::vector<double> previous_concentrations_;
    std::vector<double> previous_CO2_mass_;
    
    // Convergence handling
    ConvergenceParams conv_params_;
    ConvergenceStatus conv_status_;
    double initial_total_mass_;
    
    // Output handling
    std::unique_ptr<IOUtils::HDF5Writer> writer_;

    bool checkConvergence(double current_time) {
        // Check mass conservation
        double total_mass = calculateTotalMass();
        double expected_mass = initial_total_mass_ + accumulated_interface_mass_;
        conv_status_.mass_error = std::abs(total_mass - expected_mass) / 
                                (expected_mass + 1e-10);
        
        validateMassConservation();  // Add detailed mass conservation info
        
        if (conv_status_.mass_error > conv_params_.mass_tol) {
            conv_status_.divergence_reason = "Mass conservation violated";
            return false;
        }
        
        // Check solution change
        double max_relative_change = 0.0f;
        double max_absolute_change = 0.0f;
        
        for (size_t i = 0; i < concentrations_.size(); ++i) {
            double abs_change = std::abs(concentrations_[i] - previous_concentrations_[i]);
            double rel_change = abs_change / 
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
    
    double calculateTotalMass() const {
        double total_mass = 0.0f;
        for (size_t i = 0; i < nx_ * ny_; ++i) {
            double effective_volume = dx_ * dy_ * porosity_[i];
            
            for (int s = 0; s < num_species_; s++) {
                double conc = concentrations_[i * num_species_ + s];
                if (std::isfinite(conc)) {
                    total_mass += conc * effective_volume;
                }
            }
        }
        return total_mass;
    }

    double calculateInterfaceMassFlux() {
        double total_flux = 0.0f;
        
        // Initialize or resize if needed
        if (interface_mass_flux_.size() != nx_ * ny_) {
            interface_mass_flux_.resize(nx_ * ny_, 0.0f);
        }

        // Calculate mass flux for each interface cell
        for (int i = 0; i < nx_ * ny_; i++) {
            if (interface_cells_[i]) {
                // Calculate the mass change due to enforcing constant concentration
                double prev_mass = previous_concentrations_[i * num_species_] * dx_ * dy_;
                double current_mass = co2_saturation_conc_ * dx_ * dy_;
                double mass_flux = (current_mass - prev_mass) / dt_;
                
                interface_mass_flux_[i] = mass_flux;
                total_flux += mass_flux;
            }
        }
        
        return total_flux;
    }
    
    bool checkSolutionBounds() const {
        for (double c : concentrations_) {
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
    
    void printStats(double time) {
        double total_A = 0.0f, total_B = 0.0f, total_C = 0.0f;
        double max_A = 0.0f, max_B = 0.0f, max_C = 0.0f;

        double total_interface_flux = 0.0f;
        double max_interface_flux = 0.0f;
        
        for (int i = 0; i < nx_ * ny_; i++) {
            double A = concentrations_[i * num_species_ + 0];
            double B = concentrations_[i * num_species_ + 1];
            double C = concentrations_[i * num_species_ + 2];
            
            total_A += A;
            total_B += B;
            total_C += C;
            
            max_A = std::max(max_A, A);
            max_B = std::max(max_B, B);
            max_C = std::max(max_C, C);

            if (interface_cells_[i]) {
                double flux = interface_mass_flux_[i * num_species_];  // CO2 flux
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
        for (int i = 0; i < nx_ * ny_; i++) {
            if (interface_cells_[i]) {
                for (int s = 0; s < num_species_; s++) {
                    if (s == 0) {  // CO2
                        int idx = i * num_species_ + s;
                        double old_concentration = concentrations_[idx];
                        
                        // Use porosity vector instead of clay_cells check
                        double effective_volume = dx_ * dy_ * porosity_[i];
                        double old_mass = old_concentration * effective_volume;
                        
                        // Set new concentration
                        concentrations_[idx] = co2_saturation_conc_;
                        
                        // Calculate new mass
                        double new_mass = co2_saturation_conc_ * effective_volume;
                        
                        // Calculate mass flux
                        double mass_flux = (new_mass - old_mass) / dt_;
                        interface_mass_flux_[idx] = mass_flux;
                        
                        if (mass_flux > 0) {
                            accumulated_interface_mass_ += mass_flux * dt_;
                        }
                    }
                }
            }
        }
    }   

    void validateMassConservation() {
    double total_mass = calculateTotalMass();
    double expected_mass = initial_total_mass_ + accumulated_interface_mass_;
    double absolute_error = std::abs(total_mass - expected_mass);
    double relative_error = absolute_error / (expected_mass + 1e-10);
    
    std::cout << "\nMass Conservation Details:" << std::endl;
    std::cout << "Initial mass: " << initial_total_mass_ << std::endl;
    std::cout << "Current total mass: " << total_mass << std::endl;
    std::cout << "Accumulated interface mass: " << accumulated_interface_mass_ << std::endl;
    std::cout << "Expected mass: " << expected_mass << std::endl;
    std::cout << "Absolute error: " << absolute_error << std::endl;
    std::cout << "Relative error: " << relative_error << std::endl;
    
    // Print interface flux details
    double interface_water_flux = 0.0f;
    double interface_clay_flux = 0.0f;
    int water_interface_cells = 0;
    int clay_interface_cells = 0;
    
    for (int i = 0; i < nx_ * ny_; i++) {
        if (interface_cells_[i]) {
            if (clay_cells_[i]) {
                interface_clay_flux += interface_mass_flux_[i * num_species_];
                clay_interface_cells++;
            } else {
                interface_water_flux += interface_mass_flux_[i * num_species_];
                water_interface_cells++;
            }
        }
    }
    
    std::cout << "\nInterface Details:" << std::endl;
    std::cout << "Water interface flux: " << interface_water_flux 
              << " (cells: " << water_interface_cells << ")" << std::endl;
    std::cout << "Clay interface flux: " << interface_clay_flux 
              << " (cells: " << clay_interface_cells << ")" << std::endl;
}
};



int main() {
    // Simulation parameters
    const int nx = 200;
    const int ny = 200;
    const double dx = 0.01f;
    const double dy = 0.01f;
    const double dt = 0.00005f;
    const double total_time = 1.0f; // time will be counted as hour in the simulation
    const int num_species = 3; 

    const double Hs_co2 = 0.034f; // Henry's coefficient for CO2 in water mol/(L⋅atm) at 25°C 
    const double clay_porosity = 0.3f; // Typical clay porosity
    const double clay_tortuosity = 0.5f; // Typical clay tortuosity

    const std::string mask_file = "../Wallula_2810_pore1_final_slice73.raw";

    try {
        auto mask_data = IOUtils::MaskReader::loadRawMask(
        mask_file, nx, ny, 0, 1, 2);  // clay_label=0, water_label=1, co2_label=2

        // Create solver
        ReactiveTransportSolver solver(
            nx, ny, dx, dy, dt, total_time, num_species, Hs_co2,
            clay_porosity, clay_tortuosity);

        std::vector<int> active_cells = mask_data.active_cells;
        std::vector<double> modified_diffusion(nx * ny);
        std::vector<int> clay_cells = mask_data.clay_cells;
        std::vector<double> cell_volumes(nx * ny);
        std::vector<double> porosity(nx * ny);

        for (int i = 0; i < nx * ny; ++i) {
            if (!active_cells[i]) {
                cell_volumes[i] = 0.0f;
                porosity[i] = 0.0f;
            } else if (clay_cells[i]) {
                cell_volumes[i] = dx * dy * clay_porosity;
                porosity[i] = clay_porosity;
            } else {
                cell_volumes[i] = dx * dy;
                porosity[i] = 1.0f;
            }
        }
        
        solver.getTransportSolver().setMask(mask_data.active_cells);
        solver.setInterfaceCells(mask_data.interface_cells);
        solver.setCellVolumes(cell_volumes);
        solver.setPorosity(porosity);  
        
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