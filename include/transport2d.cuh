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
    void checkCFLCondition();
    void checkBoundaryFluxes();


private:
    int nx_, ny_;
    float dx_, dy_;
    float dt_;
    int num_species_;
    float* d_concentrations_;
    float* d_concentrations_new_;
    float* d_fluxes_x_;      // Fluxes at x-interfaces
    float* d_fluxes_y_;      // Fluxes at y-interfaces
    float2 velocity_;
    float2 diffusion_;
};
