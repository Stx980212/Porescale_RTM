#pragma once
#include <vector>
#include <cuda_runtime.h>

class TransportSolver2D {
public:
    TransportSolver2D(int nx, int ny, float dx, float dy, float dt, int num_species);
    ~TransportSolver2D();
    void solve(std::vector<float>& concentrations);
    void setVelocity(float vx, float vy);
    void setDiffusion(float dx, float dy);
    void setMask(const std::vector<int>& mask);
    void setCellVolumes(const std::vector<float>& volumes); // apply the fluid volume of each cell accroding to the porosity
    void checkCFLCondition();
    void checkBoundaryFluxes();
    float getTotalMass() const;

    std::vector<float> getDiffusionCoefficients() const;
    void setModifiedDiffusion(const std::vector<float>& modified_diffusion);
    std::vector<int> getMask() const;
    std::vector<float> getCellVolumes() const;

private:
    int nx_, ny_;
    float dx_, dy_;
    float dt_;
    int num_species_;
    float* d_concentrations_;
    float* d_concentrations_new_;
    float* d_fluxes_x_;      // Fluxes at x-interfaces
    float* d_fluxes_y_;      // Fluxes at y-interfaces
    float* d_cell_volumes_;
    float2 velocity_;
    float2 diffusion_;
    int* d_mask_;     
    float* d_modified_diffusion_;  // Add new member for modified diffusion coefficients
    bool has_modified_diffusion_;  // Flag to track if modified diffusion is being used   
    bool has_mask_;
};
