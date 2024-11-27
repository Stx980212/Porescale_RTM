# Porescale_RTM

## Project Structure
├── cache
│   └── cmake-3.31.1-tutorial-source.zip
├── CMakeLists.txt
├── include
│   ├── cuda_utils.cuh		// utility functions for cuda
│   ├── fvm_utils.cuh		// utility functions for finite volume method
│   ├── io_utils.hpp		// head file for input/output
│   ├── reactions2d.cuh		// head file for reaction
│   └── transport2d.cuh		// head file for transport 
├── README.md
├── scripts
│   ├── build.sh		// build project with cmake
│   └── delete_build_run.sh	// rebuild the project and run simulation
├── src
│   ├── io_utils.cpp		// utility functions for input/output
│   ├── main.cpp		// main script
│   ├── reactions2d.cu		// cuda kernels for reaction
│   └── transport2d.cu		// cuda kernels for transport 
├── Wallula_2810_pore1_final_slice73.raw	// input mask file for geometry
└── Wallula_2810_pore1_final_slice73.tif	// visualized geometry

