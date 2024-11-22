#include "reactions2d.cuh"
#include "fvm_utils.cuh"
#include "cuda_utils.cuh"  // Include the new header
#include <iostream>

__device__ void computeReactionRates(
    float* rates,
    const float* concentrations,
    ReactionType reaction_type,
    const ReactionParameters& params
) {
    float cA = concentrations[0];  // Species A concentration
    float cB = concentrations[1];  // Species B concentration
    float cC = concentrations[2];  // Species C concentration
    
    switch(reaction_type) {
        case ReactionType::A_PLUS_B_TO_C:
            // Simple second-order reaction: A + B -> C
            rates[0] = -params.k_forward * cA * cB;                    // Rate for A
            rates[1] = -params.k_forward * cA * cB;                    // Rate for B
            rates[2] = params.k_forward * cA * cB;                     // Rate for C
            break;
            
        case ReactionType::NONLINEAR:
            // Nonlinear reaction with backward reaction: A + B <-> C
            float forward_rate = params.k_forward * cA * cB;
            float backward_rate = params.k_backward * cC;
            rates[0] = -forward_rate + backward_rate;                  // Rate for A
            rates[1] = -forward_rate + backward_rate;                  // Rate for B
            rates[2] = forward_rate - backward_rate;                   // Rate for C
            break;
            
        case ReactionType::EQUILIBRIUM:
            // Fast equilibrium approximation
            float deviation = cA * cB - params.equilibrium_K * cC;
            float relaxation_rate = params.k_forward * deviation;
            rates[0] = -relaxation_rate;                              // Rate for A
            rates[1] = -relaxation_rate;                              // Rate for B
            rates[2] = relaxation_rate;                               // Rate for C
            break;
    }
}

__global__ void computeReactions2D(
    float* concentrations,
    int nx,
    int ny,
    float dt,
    int num_species,
    ReactionType reaction_type,
    ReactionParameters params
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i >= nx || j >= ny) return;
    
    // Get index for current cell
    int idx = (j * nx + i) * num_species;
    
    // Local array for species concentrations
    float local_conc[3];
    float reaction_rates[3];
    
    // Load concentrations
    for (int s = 0; s < num_species; s++) {
        local_conc[s] = concentrations[idx + s];
    }
    
    // Compute reaction rates
    computeReactionRates(reaction_rates, local_conc, reaction_type, params);
    
    // Update concentrations using semi-implicit scheme
    for (int s = 0; s < num_species; s++) {
        float new_conc = local_conc[s] + dt * reaction_rates[s];
        
        // Ensure positivity
        new_conc = fmaxf(0.0f, new_conc);
        
        // Write back to global memory
        concentrations[idx + s] = new_conc;
    }
    
    // Add stability check for debugging
    #ifdef DEBUG
    for (int s = 0; s < num_species; s++) {
        if (isnan(concentrations[idx + s]) || isinf(concentrations[idx + s])) {
            printf("Warning: Invalid concentration at (%d,%d) species %d\n", i, j, s);
        }
    }
    #endif
}
