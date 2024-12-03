#pragma once
#include <vector>
#include <cuda_runtime.h>

class TransportSolver2D {
public:
    TransportSolver2D(int nx, int ny, double dx, double dy, double dt, int num_species);
    ~TransportSolver2D();
    void solve(std::vector<double>& concentrations);
    void setVelocity(double vx, double vy);
    void setDiffusion(double dx, double dy);
    void setMask(const std::vector<int>& mask);
    void setCellVolumes(const std::vector<double>& volumes); // apply the fluid volume of each cell accroding to the porosity
    void checkCFLCondition();
    void checkBoundaryFluxes();
    double getTotalMass() const;

    std::vector<double> getDiffusionCoefficients() const;
    void setModifiedDiffusion(const std::vector<double>& modified_diffusion);
    std::vector<int> getMask() const;
    std::vector<double> getCellVolumes() const;
    std::vector<double> getPorosity() const;
    void setPorosity(const std::vector<double>& porosity);

    void setInterfaceConcentration(double co2_interface_conc) {
        interface_concentration_ = co2_interface_conc;
    }

private:
    int nx_, ny_;
    double dx_, dy_;
    double dt_;
    int num_species_;
    double* d_concentrations_;
    double* d_concentrations_new_;
    double* d_fluxes_x_;      // Fluxes at x-interfaces
    double* d_fluxes_y_;      // Fluxes at y-interfaces
    double* d_cell_volumes_;
    double2 velocity_;
    double2 diffusion_;
    int* d_mask_;     
    double* d_modified_diffusion_;  // Add new member for modified diffusion coefficients
    bool has_modified_diffusion_;  // Flag to track if modified diffusion is being used   
    bool has_mask_;

    double interface_concentration_; // Concentration at scCO2-water interface
    double* d_interface_flux_;      // Store interface flux for each cell
    double* d_porosity_;
};
