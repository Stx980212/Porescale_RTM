#pragma once
#include <cuda_runtime.h>

struct ReactionParameters {
    float k_forward;
    float k_backward;
    float equilibrium_K;
};

enum class ReactionType {
    A_PLUS_B_TO_C,
    NONLINEAR,
    EQUILIBRIUM
};

__global__ void computeReactions2D(
    float* concentrations,
    int nx, int ny,
    float dt,
    int num_species,
    ReactionParameters params
);