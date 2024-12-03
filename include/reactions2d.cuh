#pragma once
#include <cuda_runtime.h>

struct ReactionParameters {
    double k_forward;
    double k_backward;
    double equilibrium_K;
};

enum class ReactionType {
    A_PLUS_B_TO_C,
    NONLINEAR,
    EQUILIBRIUM
};

__global__ void computeReactions2D(
    double* concentrations,
    int nx, int ny,
    double dt,
    int num_species,
    ReactionParameters params
);