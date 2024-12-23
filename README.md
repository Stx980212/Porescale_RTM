# Pore-scale RTM
CUDA-based 2D finite-volume pore-scale reactive transport simulation\
Tianxiao Shen, Shaina Kelly* \
Department of Earth and Environmental Engineering, Columbia University

## Project structure tree
├── cache\
│   └── cmake-3.31.1-tutorial-source.zip\
├── CMakeLists.txt             // CMake configuration\
├── include\
│   ├── cuda_utils.cuh          // CUDA utility functions\
│   ├── fvm_utils.cuh           // Finite Volume Method utility functions\
│   ├── io_utils.hpp            // Input/Output operations\
│   ├── reactions2d.cuh         // Head file for reaction\
│   └── transport2d.cuh          // Head file for transport\
├── README.md\
├── scripts\
│   ├── build.sh                // Build executable with cmake\
│   └── delete_build_run.sh     // Re-build with cmake and run simulation\ 
├── src\
│   ├── io_utils.cpp\
│   ├── main.cpp                // Main program entry point\
│   ├── reactions2d.cu          // CUDA kernels for geochemical reactions\
│   └── transport2d.cu          // CUDA kernels for transport processes\
├── Wallula_2810_pore1_final_slice73.raw    // Raw geometry mask file\
└── Wallula_2810_pore1_final_slice73.tif    // Visualized geometry

## Definition
### Labels in mask file 
basalt matrix: -1\
nanoporous pore-lining clay: 0, with the assumption of a 0.3 porosity and a 0.5 torturosity\
water (brine): 1\
scCO2: 2\
carbonate precipitants: 3

### Lables of chemical species in pore water (water and pore-lining clay)
dissolved CO2: 0

### Boundary condition
dissolved CO2: constant concentration boundary by Henry's law at water/scCO2 interface, closed boundary otherwise