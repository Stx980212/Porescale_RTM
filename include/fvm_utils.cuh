#pragma once
#include <cuda_runtime.h>

namespace FVMUtils {
    struct FluxLimiter {
        // van Leer limiter
        __device__ static float vanLeer(float r) {
            return (r + fabsf(r))/(1.0f + fabsf(r));
        }
        
        // minmod limiter
        __device__ static float minmod(float r) {
            return fmaxf(0.0f, fminf(1.0f, r));
        }
        
        // superbee limiter
        __device__ static float superbee(float r) {
            return fmaxf(0.0f, fmaxf(fminf(2.0f*r, 1.0f), fminf(r, 2.0f)));
        }
    };

    // Reconstruction methods
    struct Reconstruction {
        __device__ static void muscl(
            float cL, float cR,     // Left and right cell averages
            float dx,               // Cell size
            float& cL_interface,    // Output: reconstructed left state
            float& cR_interface     // Output: reconstructed right state
        ) {
            float r = (cR - cL)/(cL + 1e-10f);  // Slope ratio
            float phi = FluxLimiter::vanLeer(r);
            
            cL_interface = cL + 0.5f * phi * (cR - cL);
            cR_interface = cR - 0.5f * phi * (cR - cL);
        }
    };

    // Numerical fluxes
    struct NumericalFlux {
        __device__ static float upwind(
            float cL, float cR,     // Left and right states
            float velocity          // Velocity at interface
        ) {
            return (velocity >= 0.0f) ? velocity * cL : velocity * cR;
        }

        __device__ static float diffusive(
            float cL, float cR,     // Left and right states
            float dx,               // Cell size
            float diffusion_coeff   // Diffusion coefficient
        ) {
            return -diffusion_coeff * (cR - cL)/dx;
        }
    };
}
